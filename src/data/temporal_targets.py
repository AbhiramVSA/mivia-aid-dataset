from __future__ import annotations

import torch


TEMPORAL_BIN_IGNORE_INDEX = -100


def build_temporal_distance_bins(
    timestamps_s: torch.Tensor,
    onset_s: float | None,
    *,
    bin_edges_s: tuple[float, float, float],
) -> torch.Tensor:
    """Return 4-bin temporal-distance targets for positive videos.

    Bins:
    0: far_before   (< edge0)
    1: near_before  ([edge0, edge1))
    2: near_after   ([edge1, edge2])
    3: far_after    (> edge2)

    Negative videos or missing onsets are returned as ignore targets.
    """

    targets = torch.full_like(timestamps_s, TEMPORAL_BIN_IGNORE_INDEX, dtype=torch.long)
    if onset_s is None:
        return targets

    edge0, edge1, edge2 = bin_edges_s
    distance = timestamps_s - float(onset_s)
    targets[distance < edge0] = 0
    targets[(distance >= edge0) & (distance < edge1)] = 1
    targets[(distance >= edge1) & (distance <= edge2)] = 2
    targets[distance > edge2] = 3
    return targets
