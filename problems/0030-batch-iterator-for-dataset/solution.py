import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    n = len(X)
    L = []
    i = 0

    while i < n:
        if y is not None:
            L.append([X[i:i+batch_size], y[i:i+batch_size]])
        else:
            L.append(X[i:i+batch_size])

        i += batch_size

    return L