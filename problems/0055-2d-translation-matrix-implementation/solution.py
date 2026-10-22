import numpy as np
def translate_object(points, tx, ty):
	translated_points = points.copy()
	for i in range(len(translated_points)):
		translated_points[i][0] += tx
	for j in range(len(translated_points)):
		translated_points[j][1] += ty
	return translated_points
