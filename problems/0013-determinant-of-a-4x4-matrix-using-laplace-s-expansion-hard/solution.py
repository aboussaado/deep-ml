def determinant_4x4(matrix: list[list[int|float]]) -> float:
	n = len(matrix)
	if n == 1: return matrix[0][0]
	det = 0
	for j in range(n):
		minor = [row[:j] + row[j+1:] for row in matrix[1:]]
		det += (-1)**j * matrix[0][j] * determinant_4x4(minor)
	return det
