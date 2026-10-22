import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    data = np.array(data)
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'mode':mode,
        'variance': np.var(data, axis=0),
        'standard_deviation': np.std(data),
        '25th_percentile': p25,
        '50th_percentile': p50,
        '75th_percentile': p75,
        'interquartile_range':  p75 - p25
    }