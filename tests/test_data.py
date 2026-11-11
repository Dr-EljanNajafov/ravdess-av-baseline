"""Tests for filename parsing and fold construction (no real data needed)."""

import itertools

import numpy as np
import pytest

from ravdess_av.data import (
    ALL_ACTORS,
    actor_independent_splits,
    actor_sex,
    av_speech_members,
    make_actor_folds,
    parse_filename,
    random_stratified_splits,
    select_av_speech,
)


def synthetic_av_names(actors=ALL_ACTORS):
    """All 60 AV speech filenames per actor, exactly as RAVDESS defines them."""
    names = []
    for actor in actors:
        for emo, inten, stmt, rep in itertools.product(range(1, 9), (1, 2), (1, 2), (1, 2)):
            if emo == 1 and inten == 2:
                continue  # neutral has no strong intensity
            names.append(f"01-01-{emo:02d}-{inten:02d}-{stmt:02d}-{rep:02d}-{actor:02d}.mp4")
    return names


# --- parsing -----------------------------------------------------------------

def test_parse_known_example():
    c = parse_filename("/data/Actor_12/01-01-03-02-01-01-12.mp4")
    assert c.emotion_name == "happy"
    assert c.label == 2
    assert c.intensity == 2 and c.statement == 1 and c.repetition == 1
    assert c.actor == 12 and c.sex == "F"
    assert c.is_av_speech


def test_sex_convention():
    assert actor_sex(1) == "M"
    assert actor_sex(2) == "F"
    assert actor_sex(23) == "M"
    assert actor_sex(24) == "F"


@pytest.mark.parametrize("bad", [
    "._01-01-03-02-01-01-12.mp4",   # macOS metadata file
    "01-01-03-02-01-12.mp4",        # too few fields
    "01-01-09-01-01-01-12.mp4",     # emotion out of range
    "01-01-03-01-01-01-25.mp4",     # actor out of range
    "01-01-01-02-01-01-01.mp4",     # strong neutral does not exist
    "readme.txt",
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_filename(bad)


def test_select_av_speech_drops_video_only():
    names = ["02-01-03-01-01-01-05.mp4", "01-01-03-01-01-01-05.mp4"]
    kept = select_av_speech(names)
    assert [c.filename for c in kept] == ["01-01-03-01-01-01-05.mp4"]


def test_av_speech_members_from_zip_listing():
    names = ["Actor_03/"]
    names += [f"Actor_03/{n}" for n in synthetic_av_names([3])]
    names += [f"Actor_03/02{n[2:]}" for n in synthetic_av_names([3])]   # video-only copies
    names += ["__MACOSX/Actor_03/._01-01-03-01-01-01-03.mp4", "Actor_03/._x.mp4"]
    members = av_speech_members(names)
    assert len(members) == 60
    assert all(m.startswith("Actor_03/01-01-") for m, _ in members)
    assert {c.actor for _, c in members} == {3}


def test_full_dataset_counts():
    clips = select_av_speech(synthetic_av_names())
    assert len(clips) == 1440
    labels = np.array([c.label for c in clips])
    counts = np.bincount(labels, minlength=8)
    assert counts[0] == 96            # neutral: 24 actors x 4
    assert all(counts[1:] == 192)     # others: 24 actors x 8


# --- folds -------------------------------------------------------------------

def test_actor_folds_partition_and_balance():
    folds = make_actor_folds(n_folds=6, seed=42)
    flat = [a for f in folds for a in f]
    assert sorted(flat) == ALL_ACTORS                # each actor tested exactly once
    for f in folds:
        assert len(f) == 4
        assert sum(actor_sex(a) == "M" for a in f) == 2


def test_actor_folds_reproducible_and_seed_dependent():
    assert make_actor_folds(seed=42) == make_actor_folds(seed=42)
    assert make_actor_folds(seed=42) != make_actor_folds(seed=7)


def test_actor_independent_no_leakage():
    clips = select_av_speech(synthetic_av_names())
    actors = np.array([c.actor for c in clips])
    splits = actor_independent_splits(actors)
    tested = np.zeros(len(clips), dtype=int)
    for train_idx, test_idx in splits:
        assert not set(actors[train_idx]) & set(actors[test_idx])
        assert len(test_idx) == 240                  # 4 actors x 60 clips
        tested[test_idx] += 1
    assert (tested == 1).all()


def test_random_splits_do_leak_actors():
    """Sanity check of the comparison protocol: actors DO overlap."""
    clips = select_av_speech(synthetic_av_names())
    actors = np.array([c.actor for c in clips])
    labels = np.array([c.label for c in clips])
    for train_idx, test_idx in random_stratified_splits(labels):
        assert set(actors[test_idx]) <= set(actors[train_idx])
