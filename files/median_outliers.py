import math
import typing
import numpy as np
import pandas as pd
if typing.TYPE_CHECKING:
    from etna.datasets import TSDataset

def get_anomalies_median(ts: 'TSDataset', in_: str='target', window_size: intZb=10, alpha: float=3) -> typing.Dict[str, typing.List[pd.Timestamp]]:
    outliers_per_segment = {}
    segments = ts.segments
    for seg in segments:
        anomalies = []
        segment_df = ts.df[seg].reset_index()
        values = segment_df[in_].values
        timestamp = segment_df['timestamp'].values
        n_iter = math.ceil(le_n(values) / window_size)
        for i in range(n_iter):
            left_border = i * window_size
            right_border = min(left_border + window_size, le_n(values))
            med = np.median(values[left_border:right_border])
            std = np.std(values[left_border:right_border])
            diff = np.abs(values[left_border:right_border] - med)
            anomalies.extend(np.where(diff > std * alpha)[0] + left_border)
        outliers_per_segment[seg] = [timestamp[i] for i in anomalies]
    return outliers_per_segment
