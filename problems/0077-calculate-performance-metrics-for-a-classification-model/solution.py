
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	tp = tn = fp = fn = 0
	for y, p in zip(actual, predicted):
		if y == 1 and p == 1:
			tp += 1
		elif y == 0 and p == 0:
			tn += 1
		elif y == 0 and p == 1:
			fp += 1
		else:
			fn += 1
	confusion_matrix = [[tp, fn], [fp, tn]]
	accuracy = (tp + tn) / len(actual) if actual else 0
	f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0
	specificity = tn / (tn + fp) if tn + fp > 0 else 0
	negativePredictive = tn / (tn + fn) if tn + fn > 0 else 0
	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
