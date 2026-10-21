import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	m = len(b)
	x = np.zeros(m)
	for k in range(n):
		x_new = np.zeros(m)
		for i in range(m):
			s = sum(A[i][j] * x[j] for j in range(m) if j != i)
			x_new[i] = (b[i] - s)/A[i][i]
		x = x_new
	return np.round(x,4).tolist()