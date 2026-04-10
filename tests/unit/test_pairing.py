from __future__ import annotations

from datetime import timedelta

import pandas as pd

from sw_pipeline.registry.pairing import pair_nearest_times


def test_pair_nearest_times_is_greedy_without_reuse():
    left = [pd.Timestamp("2024-10-10T20:00:00"), pd.Timestamp("2024-10-10T20:10:00")]
    right = [pd.Timestamp("2024-10-10T20:01:00"), pd.Timestamp("2024-10-10T20:11:00")]
    pairs = pair_nearest_times(left, right, timedelta(minutes=2))
    assert len(pairs) == 2
    assert pairs[0].left_index == 0
    assert pairs[0].right_index == 0
    assert pairs[1].left_index == 1
    assert pairs[1].right_index == 1
