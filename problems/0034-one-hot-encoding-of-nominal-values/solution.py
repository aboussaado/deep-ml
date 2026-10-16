import numpy as np

def to_categorical(x, n_col=None):
    m = max(x)
    n = len(x)

    if n_col is None:
        n_col = m + 1

    L = np.zeros((n, n_col))

    for i in range(n):
        L[i][x[i]] = 1.

    return L.tolist()