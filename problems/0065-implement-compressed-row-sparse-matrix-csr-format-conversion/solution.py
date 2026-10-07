import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	n,m = len(dense_matrix), len(dense_matrix[0])
	value = []
	column_index = []
	row_pointer = [0]
	for i in range(n):
		for j in range(m):
			if dense_matrix[i][j] != 0:
				value.append(dense_matrix[i][j])
				column_index.append(j)
		row_pointer.append(len(value))
	return (value,column_index,row_pointer)


