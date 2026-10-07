def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    n, m = len(matrix), len(matrix[0])

    if mode == 'row':
        L = [0 for i in range(n)]

        for i in range(n):
            for j in range(m):
                L[i] += matrix[i][j]

            L[i] /= m

    elif mode == 'column':
        L = [0 for j in range(m)]

        for j in range(m):
            for i in range(n):
                L[j] += matrix[i][j]

            L[j] /= n

    return L