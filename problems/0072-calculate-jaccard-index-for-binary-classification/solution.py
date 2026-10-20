
import numpy as np

def jaccard_index(y_true, y_pred):
	s_and = 0
	s_or = 0
	for i in range(len(y_true)):
		if y_true[i] == 1 and y_pred[i] == 1:
			s_and += 1
		if y_true[i] == 1 or y_pred[i] == 1:
			s_or += 1
	result = s_and / s_or
	return round(result, 3)
