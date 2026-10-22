import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	A = np.array(A)
	b = np.array(b)
	m = len(b)
	x = np.zeros(m)
	for k in range(n):
		for i in range(m):
			x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i)) - sum(A[i][j] * x[j] for j in range(i+1,m)))/A[i][i]
	return x
