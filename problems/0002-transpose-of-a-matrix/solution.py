def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    m = len(a)
    n = len(a[0])
    tr = [[0 for i in range(m)]for j in range(n)]
    for i in range(n):
        for j in range(m):
            tr[i][j] = a[j][i]
    return tr