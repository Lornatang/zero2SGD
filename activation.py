"""Implement some basic operations of Activation function.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

import numpy as np


def linear(x, w, b):
  """ linear activation func
  Paras
  -----------------------------------
  x: input of the linear layer
  w: weight
  b: bias

  Returns
  -----------------------------------
  cal wx + b
  """

  return np.dot(w, x) + b


def linear_backward(x, cache):
  """
  Paras
  -----------------------------------
  x: input of the linear layer
  w: weight
  b: bias

  Returns
  -----------------------------------
  cal wx + b
  """
  A, W, b, z = cache
  dW = np.dot(x, A.T)
  db = np.sum(x, axis=1, keepdims=True)
  dx = np.dot(W.T, x)

  return dx, dW, db


def relu(x):
  """ relu activation function
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return np.maximum(0, x)


def relu_backward(x):
  """ derivation of relu
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return np.multiply(1., np.int64(x > 0))


def sigmoid(x):
  """ implement the activation function sigmoid
  Paras
  -----------------------------------
  x: Output of the linear layer

  Returns
  -----------------------------------
  max of nums"""

  return 1 / (1 + np.exp(-x))


def sigmoid_backward(x):
  """ derivation of sigmoid
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """
  s = sigmoid(x)

  return s * (1 - s)


def tanh(x):
  """ implement the activation function tanh
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """

  return (1 - np.exp(2 * -x)) / (1 + np.exp(2 * -x))


def tanh_backward(x):
  """derivation of tanh
  Paras
  -----------------------------------
  x: output of the linear layer

  Returns
  -----------------------------------
  max of nums
  """
  favl = tanh(x)

  return 1 - favl ** 2


def batch_norm(x, gamma, beta, epsilon=1e-12):
  """
  Paras
  -----------------------------------
  x:       the input of activation (x = np.dot(W,A_pre) + b)
  gamma:   ...
  beta:    ...
  epsilon: is a constant for denominator is 0

  Returns
  -----------------------------------
  z_out, mean, variance
  """
  mu = np.mean(x, axis=1, keepdims=True)
  var = np.var(x, axis=1, keepdims=True)
  sqrt_var = np.sqrt(var + epsilon)
  z_norm = (x - mu) / sqrt_var
  z_out = np.multiply(gamma, z_norm) + beta

  return z_out, mu, var, z_norm, sqrt_var


def batch_norm_backward(dout, cache):
  _, _, _, gamma, sqrt_var, _, Z_norm = cache
  m = dout.shape[1]
  dgamma = np.sum(dout * Z_norm, axis=1, keepdims=True)
  dbeta = np.sum(dout, axis=1, keepdims=True)
  dy = 1. / m * gamma * sqrt_var * (
            m * dout - np.sum(dout, axis=1, keepdims=True) - Z_norm * np.sum(dout * Z_norm, axis=1, keepdims=True))
  return dgamma, dbeta, dy
