import math

def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    L = [math.exp(scores[i]-m) for i in range(len(scores))]
    tot = sum(L)
    return [x/tot for x in L]