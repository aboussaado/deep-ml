
import numpy as np

def pegasos_kernel_svm(data, labels, kernel='linear',
                       lambda_val=0.01, iterations=100, sigma=1.0):

    def K(x, z):
        if kernel == "linear":
            return np.dot(x, z)
        if kernel == "rbf":
            return np.exp(-np.linalg.norm(x-z)**2 / (2*sigma**2))
        raise ValueError("Unknown kernel")

    n = len(data)
    alpha = np.zeros(n)
    b = 0.0

    for t in range(1, iterations + 1):
        lr = 1 / (lambda_val * t)

        for i in range(n):
            g = sum(alpha[j] * labels[j] * K(data[j], data[i])
                    for j in range(n)) + b

            if labels[i] * g < 1:
                alpha[i] += lr * (1 - lambda_val * alpha[i])
                b += lr * labels[i]
            else:
                alpha[i] -= lr * lambda_val * alpha[i]

    return alpha.tolist(), float(b)
