import numpy as np
def rref(matrix):
	A = np.array(matrix, dtype=float)
	m, n = A.shape
	row = 0
	for col in range(n):
		if row == m:
			break
		pivot = row + np.argmax(abs(A[row:, col]))
		if np.isclose(A[pivot, col], 0):
			continue
		A[[row, pivot]] = A[[pivot, row]]
		A[row] /= A[row, col]
		for i in range(m):
			if i != row:
				A[i] -= A[i, col] * A[row]
		row += 1
	A[np.isclose(A, 0)] = 0
	return A

