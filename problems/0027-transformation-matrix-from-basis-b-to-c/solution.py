import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	return np.linalg.solve(np.array(C), np.array(B)).tolist()