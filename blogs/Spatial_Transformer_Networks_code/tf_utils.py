# %% Borrowed utils from here: https://github.com/pkmital/tensorflow_tutorials/
import tensorflow as tf
import numpy as np

def conv2d(x, n_filters, k_h=5, k_w=5, stride_h=2, stride_w=2,
	       stddev=0.02, activation=lambda x: x, bias=True, 
	       padding='SAME', name='Conv2D'):
	"""
	2D Convolution with options for kernel size, stride, and init deviation.

	Parameters
	----------
	x : Tensor
	    Input tensor to convolve.
	n_filters : int
	    Number of filters to apply.
	k_h : int, optional
	    Kernel height.
	k_w : int, optional
	    Kernel width.
	stride_h : int, optional
	    Stride in rows.
	stride_w : int, optional
	    Stride in cols.
	stddev : float, optional
	    Initialization's standard deviation.
	activation : arguments, optional
	    Function which applies a nonlinearity.
	padding : str, optional
	    'SAME' or 'VALID'.
	name : str, optional
	    Variable scope to use.

	Returns
	-------
	conv : Tensor
	    Convolved input.
	"""
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], n_filters],
			                initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
		if bias:
			b = tf.get_variable('b', [n_filters],
				                initializer=tf.truncated_normal_initializer(stddev=stddev))
			conv = tf.nn.bias_add(conv, b)
		if activation:
			conv = activation(conv)
		return conv

def linear(x, n_units, scope=None, stddev=0.02, activation=lambda x: x):
	"""
	Fully-connected network.

	Parameters
	----------
	x : Tensor
	    Input tensor to the network.
	n_units : int
	    Number of units to connect to.
	scope : str, optional
	    Variable scope to use.
	stddev : float, optional
	    Initialization's standard deviation.
	activation : arguments, optional
	    Function which applies a nonlinearity.

	Returns
	-------
	Tensor:
	    Fully-connected output.
	"""
	shape = x.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], n_units], tf.float32,
			                     tf.random_normal_initializer(stddev=stddev))
		return activation(tf.matmul(x, matrix))

def weight_variable(shape):
	'''
	Helper function to create a weight variable initialized with a normal 
	distribution.

	Parameters
	----------
	shape : list
	    Size of weight variable.
	'''
	initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
	return tf.Variable(initial)

def bias_variable(shape):
	'''
	Helper function to create a bias variable initialized with a constant value.

	Parameters
	----------
	shape : list
	    Size of bias variable.
	'''
	initial = tf.random_normal(shape, mean=0.0, stddev=0.01)

def dense_to_one_hot(labels, n_classes=2):
	"""Convert class labels from scalars to one-hot vectors."""
	labels = np.array(labels)
	n_labels = labels.shape[0]
	index_offset = np.arange(n_labels) * n_classes
	labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
	labels_one_hot.flat[index_offset + labels.ravel()] = 1
	return labels_one_hot