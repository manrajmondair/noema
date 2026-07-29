"""Which FALCON recordings train the demo model, and which are shipped to the browser.

Every recording day holds two or three session files. The first session of each day
trains; a later session of the same day is shipped as a tape. So a tape is always from
a file the model never read, on held-in and held-out days alike, which makes the
thirteen-day calendar one comparable curve instead of two protocols.

This is the failure class that cost the project its held-in number — two split
directories that turned out to share rows — so disjointness is asserted over the actual
bytes rather than inferred from filenames.
"""

import glob
import hashlib
import re
from datetime import datetime

import numpy as np

PATTERN = "data/000954/*-calib/*.nwb"


def sessions(root="data/000954"):
    """Every calibration session, grouped by recording day and ordered within the day.

    Returns a list of days, each `(day_offset, split, [paths...])`, ordered by date.
    `day_offset` counts days from the first recording, which is what the calendar plots.
    """
    found = {}
    for path in sorted(glob.glob(PATTERN.replace("data/000954", root))):
        stamp = re.search(r"ses-(\d{8})T(\d{6})", path)
        if not stamp:
            continue
        split = "held-in" if "held-in" in path else "held-out"
        found.setdefault((stamp.group(1), split), []).append((stamp.group(2), path))

    days = sorted(found)
    origin = datetime.strptime(days[0][0], "%Y%m%d")
    out = []
    for date, split in days:
        offset = (datetime.strptime(date, "%Y%m%d") - origin).days
        out.append((offset, split, [p for _, p in sorted(found[(date, split)])]))
    return out


def training_paths(root="data/000954"):
    """Session one of each HELD-IN day. These are the only files the model may read."""
    return [paths[0] for _, split, paths in sessions(root) if split == "held-in"]


def tape_paths(root="data/000954"):
    """Session two of every day, held-in and held-out. These are shipped."""
    return [(offset, split, paths[1]) for offset, split, paths in sessions(root) if len(paths) > 1]


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def assert_disjoint(tapes, training):
    """Fail if any shipped slice appears in the training set, compared over bytes.

    `tapes` and `training` are sequences of arrays. Filenames are not evidence: the
    withdrawn number came from two differently-named files holding identical rows.
    """
    seen = {_digest(a) for a in training}
    for i, tape in enumerate(tapes):
        if _digest(tape) in seen:
            raise AssertionError(f"tape {i} is byte-identical to a training recording")
        # A tape could also be a prefix or suffix of a training recording, which is
        # exactly how minival overlapped calib, so check containment on the row axis.
        for j, train in enumerate(training):
            if len(tape) <= len(train) and _contains(train, tape):
                raise AssertionError(f"tape {i} appears inside training recording {j}")
    return True


def extract(root="data/000954", bins=2250, task="h1"):
    """The thirteen shipped tapes, one per recording day.

    Returns `[(day_offset, split, session_id, counts), ...]` where counts is
    `[bins, channels]` uint8. Spike counts in this dataset already fit in a byte, which
    is asserted rather than assumed — silently clipping a real count would fabricate
    data the reader is being invited to check the model against.
    """
    from noema.eval.falcon import load_sessions

    out = []
    for offset, split, path in tape_paths(root):
        (_, neural, _, _), = load_sessions(path, task)
        # Take the session id from the filename rather than load_sessions' fixed-offset
        # slice, which lands mid-word and puts "alib_ses-..." in the provenance line.
        name = re.search(r"ses-(\d{8}T\d{6})", path).group(1)
        window = neural[:bins]
        peak = window.max()
        if peak > 255:
            raise AssertionError(f"{name}: spike count {peak} does not fit in uint8")
        out.append((offset, split, name, window.astype(np.uint8)))
    return out


def _contains(haystack, needle):
    """True if `needle` appears as a contiguous run of rows inside `haystack`."""
    if len(needle) == 0 or len(needle) > len(haystack):
        return False
    first = needle[0]
    for start in np.flatnonzero((haystack == first).all(axis=1)):
        if start + len(needle) <= len(haystack) and np.array_equal(
                haystack[start:start + len(needle)], needle):
            return True
    return False
