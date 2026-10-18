import numpy as np

def accuracy_score(y_true, y_pred):
	s = 0
	for i in range(len(y_true)):
		if y_true[i] == y_pred[i]:
			s += 1
	return s/len(y_pred)