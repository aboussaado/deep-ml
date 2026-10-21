import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	p = np.dot(v1,v2)
	m1 = np.sqrt(sum(v1**2))
	m2 = np.sqrt(sum(v2**2))
	return p / (m1 * m2)