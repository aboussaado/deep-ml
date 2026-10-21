
import numpy as np

def matrix_image(A):
	A = np.array(A)
	m,n = A.shape
	idx = []
	for i in range(n):
		cols = idx + [i]
		if np.linalg.matrix_rank(A[:,cols]) > len(idx):
			idx.append(i)
	return A[:,idx]