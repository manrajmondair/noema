"""Noema: a coupled, action-conditioned world model of neural population activity."""

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .adversary import SessionAdversary
from .coupling import SensoryEncoder
from .encoder import SpatioTemporalEncoder, TemporalEncoder
from .heads import AttentionPool, BehaviorHead, CrossReadout, FiLMReadout
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


def contrastive_loss(pred, target, temp=0.1):
    # InfoNCE across the batch at each timestep (the SimCLR/STNDT form): a trial's masked
    # online latent must match its own clean EMA-teacher latent, against the OTHER trials'
    # latents at the same time as negatives. Contrasting trials (which genuinely differ in
    # firing) aligns with the task's condition structure — unlike contrasting timesteps
    # within a trial, which forces apart correlated states and fights the linear rate readout.
    pred = F.normalize(pred, dim=-1)              # [B,T,D]
    target = F.normalize(target.detach(), dim=-1)
    batch = pred.size(0)
    logits = torch.einsum("btd,ctd->tbc", pred, target) / temp  # [T,B,B]
    labels = torch.arange(batch, device=pred.device).expand(pred.size(1), batch).reshape(-1)
    return F.cross_entropy(logits.reshape(-1, batch), labels)


class Noema(nn.Module):
    def __init__(self, dim=256, enc_depth=6, wm_depth=3, heads=8, max_units=8192,
                 action_dim=0, behavior_dim=0, context_dim=0, sessions=0, mask_ratio=0.25,
                 adv_weight=1.0, ema=0.996, spatial=False, neuron_mask_ratio=0.0, cross=False,
                 multistep=0, attn_pool=False, contrastive=False, contrastive_temp=0.1, ssm=False, ssm_state=128, hybrid=False,
                 ssm_dt=False, film=False, graft=False):
        super().__init__()
        self.spatial = spatial
        self.neuron_mask_ratio = neuron_mask_ratio
        self.multistep = multistep  # >1 adds a multi-step rollout loss (open-loop drift resistance)
        self.contrastive = contrastive  # InfoNCE representation loss (STNDT-style)
        self.contrastive_temp = contrastive_temp
        # SimCLR/STNDT projection head: the contrastive loss acts on proj(z), NOT z itself,
        # so shaping the representation to be contrastively discriminative does not distort
        # the clean z that the linear rate readout decodes.
        self.contrastive_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) if contrastive else None
        self.tokenizer = PopulationTokenizer(dim, max_units, graft)
        if spatial:
            self.encoder = SpatioTemporalEncoder(dim, enc_depth, heads)
        elif ssm:  # diagonal state-space temporal encoder (S5/LRU-style)
            from .ssm import SSMEncoder
            self.encoder = SSMEncoder(dim, enc_depth, heads, ssm_state, hybrid, learn_dt=ssm_dt)
        else:
            self.encoder = TemporalEncoder(dim, enc_depth, heads)
        self.world = WorldModel(dim, wm_depth, heads, action_dim)
        self.behavior = BehaviorHead(dim, behavior_dim) if behavior_dim else None
        self.cross = CrossReadout(dim, heads) if (spatial and cross) else None  # per-unit co-smoothing
        self.film = FiLMReadout(dim) if film else None  # nonlinear held-out readout (vs linear decode)
        # Content-weighted pool of the per-unit tokens into the shared latent (spatial only);
        # falls back to a mean when off. The teacher mirrors it so the JEPA target lives in
        # the same latent space the world model predicts.
        self.pooler = AttentionPool(dim, heads) if (spatial and attn_pool) else None
        self.sensory = SensoryEncoder(context_dim, dim, max(2, wm_depth)) if context_dim else None
        self.adversary = SessionAdversary(dim, sessions, adv_weight) if sessions else None
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.teacher_pooler = copy.deepcopy(self.pooler).requires_grad_(False) if self.pooler else None
        self.mask_ratio = mask_ratio
        self.ema = ema
        # Head count can't be read back from weight shapes (qkv is dim->k*dim regardless),
        # so persist it: a checkpoint reloaded with the wrong head count silently scrambles
        # attention. Lets build_from_state reconstruct the exact model.
        self.register_buffer("num_heads", torch.tensor(heads), persistent=True)

    def _represent(self, counts, unit_ids):
        # returns (per-unit tokens or None, pooled latent z [B,T,dim])
        if self.spatial:
            tokens = self.encoder(self.tokenizer.encode_units(counts, unit_ids))
            z = self.pooler(tokens) if self.pooler is not None else tokens.mean(2)
            return tokens, z
        return None, self.encoder(self.tokenizer.encode(counts, unit_ids))

    def encode(self, counts, unit_ids):
        return self._represent(counts, unit_ids)[1]

    def _teacher_target(self, counts, unit_ids):
        if self.spatial:
            tok = self.teacher(self.tokenizer.encode_units(counts, unit_ids))
            return self.teacher_pooler(tok) if self.teacher_pooler is not None else tok.mean(2)
        return self.teacher(self.tokenizer.encode(counts, unit_ids))

    def _film(self, z, target_unit_ids):
        return self.film(z, self.tokenizer.readout(target_unit_ids),
                         self.tokenizer.bias(target_unit_ids).squeeze(-1))

    def cosmooth(self, counts, unit_ids, target_unit_ids):
        """Log-rates for held-out (co-smoothing) units, given only held-in spikes."""
        if self.cross is not None:
            tokens = self.encoder(self.tokenizer.encode_units(counts, unit_ids))
            return self.cross(tokens, self.tokenizer.readout(target_unit_ids))
        z = self.encode(counts, unit_ids)
        return self._film(z, target_unit_ids) if self.film is not None else self.tokenizer.decode(z, target_unit_ids)

    @torch.no_grad()
    def cosmooth_tta(self, counts, unit_ids, target_unit_ids, k=30):
        """Test-time augmentation: average the held-out RATE over k coordinated-dropout masks.
        Marginalizing the input mask reduces single-model estimator variance (the same variance
        ensembling reduces).
        Returns a rate (exp of log-rate), not a log-rate."""
        if k <= 0:
            return self.cosmooth(counts, unit_ids, target_unit_ids).exp()
        r = 0.0
        for _ in range(k):
            obs = counts.masked_fill(torch.rand_like(counts) < self.mask_ratio, 0.0)
            r = r + self.cosmooth(obs, unit_ids, target_unit_ids).exp()
        return r / k

    def _cosmooth_from(self, tokens, z, target_unit_ids):
        if self.cross is not None:
            return self.cross(tokens, self.tokenizer.readout(target_unit_ids))
        return self._film(z, target_unit_ids) if self.film is not None else self.tokenizer.decode(z, target_unit_ids)

    @torch.no_grad()
    def cosmooth_dynamical(self, counts, unit_ids, target_unit_ids, alpha=0.5):
        """Co-smooth held-out rates after blending the encoder's per-bin latent with the world
        model's one-step dynamical prediction. The encoder denoises with a fixed temporal filter;
        the world model denoises by projecting onto its learned vector field (trajectory shape),
        a different inference channel -> its errors can decorrelate from the encoder's. alpha=1
        recovers the plain estimate; alpha<1 mixes in the dynamics prior."""
        z = self.encode(counts, unit_ids)
        zf = torch.cat([z[:, :1], self.world(z)[:, :-1]], dim=1)  # world(z)[:,t] predicts z[:,t+1]
        zs = alpha * z + (1 - alpha) * zf
        return self.tokenizer.decode(zs, target_unit_ids)

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

        # Contrastive representation loss: match the masked online latent to its clean
        # teacher latent at the same timestep, repelling the trial's other timesteps.
        if self.contrastive:
            out["loss_contrastive"] = contrastive_loss(
                self.contrastive_proj(z), self.contrastive_proj(target), self.contrastive_temp)

        # Anchor the forecast in observation space: the predicted next latent must
        # decode to the next bin's firing. This is what makes rollouts faithful.
        out["loss_forecast"] = poisson_nll(self.tokenizer.decode(pred[:, :-1], unit_ids), counts[:, 1:])

        # Multi-step rollout: keep predicting from the model's OWN predicted latents and
        # match each future bin's firing. Training on its own rollout (not the clean
        # encoding) is what teaches open-loop rollouts to resist drift over the horizon.
        # zk[t] ~ z[t+k] after k steps; when action-conditioned, the action that drives the
        # step into bin t+k is a_{t+k-1}, so the action stream is shifted and zk truncated.
        if self.multistep > 1 and counts.size(1) > self.multistep:
            zk, ms, nk = pred, 0.0, 0
            for k in range(2, self.multistep + 1):
                a = actions[:, k - 1:] if actions is not None else None
                if a is not None:
                    zk = zk[:, :a.size(1)]
                zk = self.world(zk, a)
                L = counts.size(1) - k
                if L <= 0:
                    break
                ms = ms + poisson_nll(self.tokenizer.decode(zk[:, :L], unit_ids), counts[:, k:k + L])
                nk += 1
            if nk:
                out["loss_multistep"] = ms / nk

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
        if self.teacher_pooler is not None:
            for online, target in zip(self.pooler.parameters(), self.teacher_pooler.parameters()):
                target.lerp_(online, 1.0 - self.ema)
