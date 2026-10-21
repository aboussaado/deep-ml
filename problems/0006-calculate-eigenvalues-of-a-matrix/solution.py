import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a, b = matrix[0]
	c, d = matrix[1]
	trace = a + d
	det = a*d - b*c
	delta = trace**2 - 4*det
	if delta < 0:
		raise ValueError("complex eigenvalues")
	l1 = (trace + math.sqrt(delta)) / 2
	l2 = (trace - math.sqrt(delta)) / 2
	return sorted([l1, l2], reverse=True)