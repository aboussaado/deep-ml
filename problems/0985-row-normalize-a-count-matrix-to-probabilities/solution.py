
import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    counts = np.array(counts, dtype=float)
    row_sums = np.sum(counts, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    counts /= row_sums
    return counts.tolist()