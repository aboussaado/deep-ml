from collections import Counter

def confusion_matrix(data):
	counts = Counter(tuple(pair) for pair in data)
	TP = counts[(1, 1)]
	TN = counts[(0, 0)]
	FP = counts[(0, 1)]
	FN = counts[(1, 0)]
	return [[TP, FN], [FP, TN]]