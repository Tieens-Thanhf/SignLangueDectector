"""Shared hand landmark feature extraction.

Keep feature extraction consistent between dataset creation and realtime inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


LANDMARK_COUNT = 21


@dataclass(frozen=True)
class FeatureConfig:
    normalize: bool = True
    include_z: bool = True

    @property
    def feature_dim(self) -> int:
        return LANDMARK_COUNT * (3 if self.include_z else 2)


def _flatten_landmarks(hand_landmarks, include_z: bool) -> Tuple[List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for lm in hand_landmarks.landmark:
        xs.append(float(lm.x))
        ys.append(float(lm.y))
        if include_z:
            zs.append(float(lm.z))
    return xs, ys, zs


def extract_hand_features(hand_landmarks, config: FeatureConfig = FeatureConfig()) -> Optional[List[float]]:
    """Return a fixed-length feature vector from a single hand's landmarks.

    Landmarks are in normalized image coordinates (x/y in [0..1], z in roughly [-?..?]).
    We make them more translation/scale invariant by subtracting min(x/y[/z]) and
    dividing by the max span (x_span or y_span).
    """

    xs, ys, zs = _flatten_landmarks(hand_landmarks, include_z=config.include_z)

    if len(xs) != LANDMARK_COUNT or len(ys) != LANDMARK_COUNT:
        return None
    if config.include_z and len(zs) != LANDMARK_COUNT:
        return None

    if config.normalize:
        min_x, min_y = min(xs), min(ys)
        xs = [x - min_x for x in xs]
        ys = [y - min_y for y in ys]

        span_x = max(xs) - min(xs) if xs else 0.0
        span_y = max(ys) - min(ys) if ys else 0.0
        scale = max(span_x, span_y)
        if scale <= 1e-9:
            scale = 1.0

        xs = [x / scale for x in xs]
        ys = [y / scale for y in ys]

        if config.include_z:
            min_z = min(zs)
            zs = [z - min_z for z in zs]
            # Keep z roughly in the same scale as x/y.
            zs = [z / scale for z in zs]

    features: List[float] = []
    for i in range(LANDMARK_COUNT):
        features.append(xs[i])
        features.append(ys[i])
        if config.include_z:
            features.append(zs[i])

    if len(features) != config.feature_dim:
        return None

    return features
