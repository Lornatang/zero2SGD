"""Implement some basic operations of Activation function.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

import numpy as np


def relu(x):
  """ linear activation function
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

  return np.int64(x > 0)


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

  fval = sigmoid(x)

  return fval * (1 - fval)


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
