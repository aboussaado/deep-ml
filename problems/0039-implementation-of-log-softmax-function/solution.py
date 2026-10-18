import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    scores = np.array(scores)
    softmax = np.exp(scores)/np.sum(np.exp(scores))
    return np.log(softmax)