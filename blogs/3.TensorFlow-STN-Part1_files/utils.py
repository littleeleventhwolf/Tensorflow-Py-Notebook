import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image

def img_to_array(data_path, desired_size=None, view=False):
	"""
	Util function for loading RGB image into 4D numpy array.

	Returns array of shape (1, H, W, C)

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	img = Image.open(data_path)
	img = img.convert('RGB')
	if desired_size:
		img = img.resize((desired_size[1], desired_size[0]))
	if view:
		img.show()

	# preprocess
	x = np.asarray(img, dtype='float32')
	x = np.expand_dims(x, axis=0)
	x /= 255.0

	return x

def array_to_img(x):
	"""
	Util function for converting 4D numpy array to RGB image.

	Returns PIL RGB image.

	References
	----------
	- adapted from keras preprocessing/image.py
	"""
	x = np.asarray(x)
	x += max(-np.min(x), 0)
	x_max = np.max(x)
	if x_max != 0:
		x /= x_max
	x *= 255
	return Image.fromarray(x.astype('uint8'), 'RGB')