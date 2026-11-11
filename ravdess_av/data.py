"""RAVDESS filename parsing and cross-validation folds.

RAVDESS encodes all labels in the filename as seven two-digit fields:

    modality-vocalchannel-emotion-intensity-statement-repetition-actor.mp4
    e.g. 01-01-03-02-01-01-12.mp4

    modality      01 = audio-video, 02 = video-only, 03 = audio-only
    vocal channel 01 = speech, 02 = song
    emotion       01..08 (see EMOTIONS)
    intensity     01 = normal, 02 = strong (neutral has no strong intensity)
    statement     01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
    repetition    01 or 02
    actor         01..24; odd = male, even = female

This module has no heavy dependencies on purpose: it is imported both by the
GPU feature extractor and by the CPU evaluation code, and it is unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Emotion codes as defined by the dataset authors (Livingstone & Russo, 2018).
EMOTIONS = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
EMOTION_NAMES: List[str] = [EMOTIONS[i] for i in range(1, 9)]  # index = class id

MODALITY_AV = 1
VOCAL_SPEECH = 1
N_ACTORS = 24
ALL_ACTORS = list(range(1, N_ACTORS + 1))


@dataclass(frozen=True)
class ClipInfo:
    """Labels decoded from one RAVDESS filename."""

    filename: str
    modality: int
    vocal_channel: int
    emotion: int  # dataset code 1..8
    intensity: int
    statement: int
    repetition: int
    actor: int

    @property
    def label(self) -> int:
        """Zero-based class index (0..7) for classifiers."""
        return self.emotion - 1

    @property
    def emotion_name(self) -> str:
        return EMOTIONS[self.emotion]

    @property
    def sex(self) -> str:
        return actor_sex(self.actor)

    @property
    def is_av_speech(self) -> bool:
        return self.modality == MODALITY_AV and self.vocal_channel == VOCAL_SPEECH


def actor_sex(actor: int) -> str:
    """RAVDESS convention: odd actor ids are male, even are female."""
    if not 1 <= actor <= N_ACTORS:
        raise ValueError(f"actor id out of range: {actor}")
    return "M" if actor % 2 == 1 else "F"


# Allowed values per field, used to reject malformed or foreign files early.
_VALID = {
    "modality": {1, 2, 3},
    "vocal_channel": {1, 2},
    "emotion": set(EMOTIONS),
    "intensity": {1, 2},
    "statement": {1, 2},
    "repetition": {1, 2},
    "actor": set(ALL_ACTORS),
}
_FIELDS = list(_VALID)


def parse_filename(path: str | Path) -> ClipInfo:
    """Decode a RAVDESS filename (path or bare name) into a ClipInfo.

    Raises ValueError for anything that is not a well-formed RAVDESS name,
    so that a stray file (e.g. macOS '._' metadata) never silently becomes
    a training sample with a wrong label.
    """
    name = Path(path).name
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("-")
    if len(parts) != 7 or not all(len(p) == 2 and p.isdigit() for p in parts):
        raise ValueError(f"not a RAVDESS filename: {name!r}")

    values = dict(zip(_FIELDS, (int(p) for p in parts)))
    for field, value in values.items():
        if value not in _VALID[field]:
            raise ValueError(f"{name!r}: invalid {field}={value}")
    if values["emotion"] == 1 and values["intensity"] == 2:
        raise ValueError(f"{name!r}: neutral has no strong intensity in RAVDESS")

    return ClipInfo(filename=name, **values)


def select_av_speech(paths: Iterable[str | Path]) -> List[ClipInfo]:
    """Keep only audio-visual speech clips (modality 01, vocal channel 01).

    Video_Speech_Actor_XX.zip contains both 01 (AV) and 02 (video-only)
    files; only the AV ones carry the audio track we need.
    """
    clips = [parse_filename(p) for p in paths]
    return sorted((c for c in clips if c.is_av_speech), key=lambda c: c.filename)


def av_speech_members(zip_names: Iterable[str]) -> List[Tuple[str, ClipInfo]]:
    """Pick AV speech clips out of the member list of an actor's zip archive.

    Archives contain paths like 'Actor_01/01-01-03-01-01-01-01.mp4'.
    Directory entries and macOS metadata ('__MACOSX/', '._*') are skipped;
    every remaining .mp4 is parsed strictly, so an unexpected name fails loudly.
    Returns (member_name, ClipInfo) pairs sorted by filename.
    """
    out = []
    for member in zip_names:
        base = member.rstrip("/").split("/")[-1]
        if member.endswith("/") or "__MACOSX" in member or base.startswith("._"):
            continue
        if not base.lower().endswith(".mp4"):
            continue
        info = parse_filename(base)
        if info.is_av_speech:
            out.append((member, info))
    return sorted(out, key=lambda pair: pair[1].filename)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def make_actor_folds(n_folds: int = 6, seed: int = 42,
                     actors: Sequence[int] = ALL_ACTORS) -> List[List[int]]:
    """Split actors into sex-balanced, disjoint test groups.

    With 24 actors and 6 folds each fold holds 2 male + 2 female actors.
    Shuffling within each sex with a fixed seed makes the split random but
    reproducible.
    """
    males = sorted(a for a in actors if actor_sex(a) == "M")
    females = sorted(a for a in actors if actor_sex(a) == "F")
    if len(males) % n_folds or len(females) % n_folds:
        raise ValueError(
            f"cannot balance {len(males)}M/{len(females)}F actors over {n_folds} folds")

    rng = np.random.default_rng(seed)
    males = list(rng.permutation(males))
    females = list(rng.permutation(females))
    m, f = len(males) // n_folds, len(females) // n_folds
    return [sorted(int(a) for a in males[i * m:(i + 1) * m] + females[i * f:(i + 1) * f])
            for i in range(n_folds)]


def actor_independent_splits(clip_actors: Sequence[int], n_folds: int = 6,
                             seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs where no actor appears on both sides."""
    clip_actors = np.asarray(clip_actors)
    present = sorted(set(int(a) for a in clip_actors))
    splits = []
    for test_actors in make_actor_folds(n_folds, seed, present):
        test_mask = np.isin(clip_actors, test_actors)
        splits.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return splits


def random_stratified_splits(labels: Sequence[int], n_folds: int = 6,
                             seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Comparison protocol: clip-level stratified K-fold, actors overlap.

    This mirrors the random splits common in the literature and is expected
    to overestimate accuracy on unseen people.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))
