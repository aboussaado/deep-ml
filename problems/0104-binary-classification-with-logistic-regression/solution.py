import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	N,D = X.shape
	z = X @ weights + bias
	sigma = 1 / (1+np.exp(-z))
	L = []
	for i in range(len(sigma)):
		if sigma[i] >= 0.5:
			L.append(1)
		else:
			L.append(0)
	return L