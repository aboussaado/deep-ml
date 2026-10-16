import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	n = X.shape[0]
	perm = np.random.permutation(n)
	Z = np.array([X[perm[i]] for i in range(n)])
	T = np.array([y[perm[i]] for i in range(n)])
	return Z,T
