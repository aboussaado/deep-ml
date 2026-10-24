import numpy as np
def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    M = A.T @ A
    theta = 0.5 * np.arctan2(2 * M[0, 1],M[0, 0] - M[1, 1])
    c = np.cos(theta)
    s = np.sin(theta)
    V = np.array([[c, -s],[s,  c]])
    B = A @ V
    S = np.sqrt(np.sum(B**2, axis=0))
    indices = np.argsort(S)[::-1]
    S = S[indices]
    V = V[:, indices]
    B = B[:, indices]
    U = np.zeros((2, 2))
    if S[0] < 1e-12:
        U = np.eye(2)
        S[:] = 0
    else:
        U[:, 0] = B[:, 0] / S[0]
        if S[1] > 1e-12:
            U[:, 1] = B[:, 1] / S[1]
        else:
            S[1] = 0
            U[:, 1] = [-U[1, 0], U[0, 0]]
    return U, S, V.T
