from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ..models import TimePair


def pair_nearest_times(
    left_times: list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    right_times: list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    max_delta: timedelta,
) -> list[TimePair]:
    candidates: list[tuple[timedelta, int, int]] = []
    for left_index, left_time in enumerate(left_times):
        for right_index, right_time in enumerate(right_times):
            delta = abs(left_time - right_time)
            if delta <= max_delta:
                candidates.append((delta, left_index, right_index))

    candidates.sort(key=lambda item: (item[0], left_times[item[1]], right_times[item[2]]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[TimePair] = []
    for delta, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pairs.append(
            TimePair(
                left_time=left_times[left_index],
                right_time=right_times[right_index],
                delta=delta,
                left_index=left_index,
                right_index=right_index,
            )
        )
    return pairs
