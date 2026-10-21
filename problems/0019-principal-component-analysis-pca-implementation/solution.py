import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    M = data - np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    sd[sd == 0] = 1
    M = M / sd
    cov = M.T @ M / (len(data) - 1)
    val,vect = np.linalg.eigh(cov)
    indices = np.argsort(val)[::-1][:k]
    components = vect[:, indices].copy()
    for i in range(k):
        v = components[:, i]
        nonzero = np.where(np.abs(v) > 1e-12)[0]
        if len(nonzero) > 0 and v[nonzero[0]] < 0:
            components[:, i] *= -1
    return np.round(components, 4)
