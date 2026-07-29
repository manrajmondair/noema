"""The demo's split discipline.

The project has already shipped one number that was fitted on the data it was scored
against, because two split directories shared rows. These assertions are the guard.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.tapes import _contains, assert_disjoint, sessions, tape_paths, training_paths  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "000954"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="FALCON dandiset not downloaded")


def test_identical_recording_is_rejected():
    a = np.arange(60, dtype=np.uint8).reshape(20, 3)
    with pytest.raises(AssertionError, match="byte-identical"):
        assert_disjoint([a], [a])


def test_a_tape_contained_in_a_training_recording_is_rejected():
    # This is the exact shape of the defect that cost the held-in number: a shorter
    # recording that is a contiguous run of rows inside a longer one.
    train = np.arange(300, dtype=np.uint8).reshape(100, 3)
    with pytest.raises(AssertionError, match="appears inside"):
        assert_disjoint([train[20:50]], [train])


def test_genuinely_disjoint_recordings_pass():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 5, (100, 3), dtype=np.uint8)
    tape = rng.integers(200, 255, (30, 3), dtype=np.uint8)
    assert assert_disjoint([tape], [train])


def test_contains_is_row_wise_not_elementwise():
    hay = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.uint8)
    assert _contains(hay, np.array([[2, 2], [3, 3]], dtype=np.uint8))
    # the rows exist individually but never adjacently in this order
    assert not _contains(hay, np.array([[3, 3], [2, 2]], dtype=np.uint8))


@needs_data
def test_every_day_can_train_on_one_session_and_ship_another():
    days = sessions()
    assert len(days) == 13, f"expected 13 recording days, found {len(days)}"
    assert all(len(paths) >= 2 for _, _, paths in days), "a day has no second session to ship"
    assert [offset for offset, _, _ in days] == [0, 7, 12, 14, 18, 19, 25, 26, 28, 32, 33, 36, 39]


@needs_data
def test_shipped_tapes_do_not_overlap_the_training_recordings():
    # The unit tests above cover the comparison logic; this runs it on the real bytes,
    # which is the only version of the guarantee that can actually fail in production.
    from demo.tapes import extract
    from noema.eval.falcon import load_sessions

    tapes = extract()
    assert len(tapes) == 13
    assert all(a.dtype == np.uint8 and a.shape[1] == 176 for *_, a in tapes)
    training = [n for path in training_paths() for _, n, _, _ in load_sessions(path)]
    assert assert_disjoint([a for *_, a in tapes], training)


@needs_data
def test_no_shipped_file_is_ever_a_training_file():
    shipped = {path for _, _, path in tape_paths()}
    assert shipped.isdisjoint(set(training_paths()))
    assert len(shipped) == 13
