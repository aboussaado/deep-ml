import numpy as np
def calculate_brightness(img):
	if not img or not img[0]:
		return - 1
	n = len(img[0])
	for row in img:
		if len(row) != n:
			return -1
		for pixel in row:
			if not (0<=pixel<=255):
				return -1
	return float(np.mean(img))