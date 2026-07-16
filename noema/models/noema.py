"""Noema: a coupled, action-conditioned world model of neural population activity."""

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .adversary import SessionAdversary
from .coupling import SensoryEncoder
from .encoder import SpatioTemporalEncoder, TemporalEncoder
from .heads import BehaviorHead, CrossReadout
from .tokenizer import PopulationTokenizer
from .world_model import WorldModel


def poisson_nll(log_rate, counts, mask=None):
    loss = torch.exp(log_rate) - counts * log_rate
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def latent_prediction_loss(pred, target):
    # cosine distance against a stop-grad (EMA) target — a joint-embedding objective
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return (1.0 - (pred * target).sum(-1)).mean()


class Noema(nn.Module):
    def __init__(self, dim=256, enc_depth=6, wm_depth=3, heads=8, max_units=8192,
                 action_dim=0, behavior_dim=0, context_dim=0, sessions=0, mask_ratio=0.25,
                 adv_weight=1.0, ema=0.996, spatial=False, neuron_mask_ratio=0.0, cross=False):
        super().__init__()
        self.spatial = spatial
        self.neuron_mask_ratio = neuron_mask_ratio
        self.tokenizer = PopulationTokenizer(dim, max_units)
        self.encoder = (SpatioTemporalEncoder if spatial else TemporalEncoder)(dim, enc_depth, heads)
        self.world = WorldModel(dim, wm_depth, heads, action_dim)
        self.behavior = BehaviorHead(dim, behavior_dim) if behavior_dim else None
        self.cross = CrossReadout(dim, heads) if (spatial and cross) else None  # per-unit co-smoothing
        self.sensory = SensoryEncoder(context_dim, dim, max(2, wm_depth), heads) if context_dim else None
        self.adversary = SessionAdversary(dim, sessions, adv_weight) if sessions else None
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.mask_ratio = mask_ratio
        self.ema = ema

    def _represent(self, counts, unit_ids):
        # returns (per-unit tokens or None, pooled latent z [B,T,dim])
        if self.spatial:
            tokens = self.encoder(self.tokenizer.encode_units(counts, unit_ids))
            return tokens, tokens.mean(2)  # pool over units into the shared latent
        return None, self.encoder(self.tokenizer.encode(counts, unit_ids))

    def encode(self, counts, unit_ids):
        return self._represent(counts, unit_ids)[1]

    def _teacher_target(self, counts, unit_ids):
        if self.spatial:
            return self.teacher(self.tokenizer.encode_units(counts, unit_ids)).mean(2)
        return self.teacher(self.tokenizer.encode(counts, unit_ids))

    def cosmooth(self, counts, unit_ids, target_unit_ids):
        """Log-rates for held-out (co-smoothing) units, given only held-in spikes."""
        if self.cross is not None:
            tokens = self.encoder(self.tokenizer.encode_units(counts, unit_ids))
            return self.cross(tokens, self.tokenizer.readout(target_unit_ids))
        return self.tokenizer.decode(self.encode(counts, unit_ids), target_unit_ids)

    def _cosmooth_from(self, tokens, z, target_unit_ids):
        if self.cross is not None:
            return self.cross(tokens, self.tokenizer.readout(target_unit_ids))
        return self.tokenizer.decode(z, target_unit_ids)

    def forward(self, counts, unit_ids, actions=None, behavior=None,
                target_counts=None, target_unit_ids=None, session=None, context=None):
        # Coordinated dropout: hide a fraction of spikes and reconstruct only those,
        # which blocks the trivial copy solution and forces use of population structure.
        loss_mask, observed = None, counts
        if self.training and self.mask_ratio > 0:
            loss_mask = (torch.rand_like(counts) < self.mask_ratio).float()
            # BERT 80/10/10 corruption of masked cells: zeroing all of them would make
            # a masked cell indistinguishable from a true zero-count cell (log1p(0)=0),
            # leaking the mask. 80% -> 0, 10% -> a random plausible count, 10% kept.
            mode = torch.rand_like(counts)
            rand = loss_mask * ((mode >= 0.8) & (mode < 0.9)).float()
            zero = loss_mask * (mode < 0.8).float()
            shuffled = counts.reshape(-1)[torch.randperm(counts.numel(), device=counts.device)].reshape_as(counts)
            observed = counts * (1 - zero - rand) + shuffled * rand

        tokens, z = self._represent(observed, unit_ids)
        rate = self.tokenizer.decode_units(tokens, unit_ids) if self.spatial else self.tokenizer.decode(z, unit_ids)
        out = {"z": z, "rate": rate}
        out["loss_rate"] = poisson_nll(out["rate"], counts, loss_mask)

        # Co-smoothing: infer the firing of held-out units the encoder never saw.
        if target_counts is not None:
            out["loss_cosmooth"] = poisson_nll(self._cosmooth_from(tokens, z, target_unit_ids), target_counts)

        # Random-neuron co-smoothing: hide a random subset of input units and predict
        # them from the rest, so the metric objective (predict any held-out unit) is
        # trained on every split, not just the fixed one. Zeroing drops a unit from the
        # summed token entirely (log1p(0)=0), so the encoder genuinely never sees it.
        if self.training and self.neuron_mask_ratio > 0:
            held = torch.rand(counts.size(-1), device=counts.device) < self.neuron_mask_ratio
            if held.any():
                hidden = counts.clone()
                hidden[..., held] = 0
                pred = self.tokenizer.decode(self.encode(hidden, unit_ids), unit_ids[held])
                out["loss_ncosmooth"] = poisson_nll(pred, counts[..., held])

        # Forecast the next latent; the target comes from the clean, unmasked view.
        target = self._teacher_target(counts, unit_ids)
        pred = self.world(z, actions)
        out["loss_jepa"] = latent_prediction_loss(pred[:, :-1], target[:, 1:])

        # Anchor the forecast in observation space: the predicted next latent must
        # decode to the next bin's firing. This is what makes rollouts faithful.
        out["loss_forecast"] = poisson_nll(self.tokenizer.decode(pred[:, :-1], unit_ids), counts[:, 1:])

        if self.behavior is not None and behavior is not None:
            out["loss_behavior"] = F.mse_loss(self.behavior(z), behavior)

        # Sensory coupling: predict the population's firing from the stimulus alone,
        # read out by the shared per-unit decoder.
        if self.sensory is not None and context is not None:
            out["loss_sensory"] = poisson_nll(self.tokenizer.decode(self.sensory(context), unit_ids), counts)

        if self.adversary is not None and session is not None:
            out["loss_session"] = self.adversary(z, session)
        return out

    @torch.no_grad()
    def predict_response(self, context, unit_ids):
        """Firing rates a population would produce in response to a stimulus."""
        self.eval()
        return self.tokenizer.decode(self.sensory(context), unit_ids).exp()

    @torch.no_grad()
    def update_teacher(self):
        for online, target in zip(self.encoder.parameters(), self.teacher.parameters()):
            target.lerp_(online, 1.0 - self.ema)

    @torch.no_grad()
    def rollout(self, seed_counts, unit_ids, future_actions, seed_actions=None):
        """Imagine firing rates forward from a seed window under a plan of actions.

        Latents and actions stay index-aligned: the prediction at position t uses
        the action at t, so `seed_actions` must cover the seed window when the model
        is action-conditioned.
        """
        z = self.encode(seed_counts, unit_ids)
        a = seed_actions
        for t in range(future_actions.size(1)):
            z = torch.cat([z, self.world(z, a)[:, -1:]], dim=1)
            if a is not None:
                a = torch.cat([a, future_actions[:, t : t + 1]], dim=1)
        return self.tokenizer.decode(z, unit_ids)
