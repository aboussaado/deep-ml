import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	z = np.array(features) @ np.array(weights) + np.array(bias)
	def sigmoid(x):
		return 1/(1+np.exp(-x))
	probabilities = sigmoid(z)
	mse = np.mean((probabilities - np.array(labels))**2)
	return probabilities.tolist(), mse