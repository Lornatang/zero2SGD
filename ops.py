"""Implement some basic operations of SGD.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

from activation import *


def random_mini_batches(data, label, mini_batch_size, seed=10):
  """ creates a list of random mini batches from (data, label)
  Paras
  -----------------------------------
  data:            input data, of shape (input size, number of examples)
  label:           true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  mini_batch_size: size of the mini-batches, integer

  Returns
  -----------------------------------
  mini_batches:    list of synchronous (mini_batch_X, mini_batch_Y)
  """
  np.random.seed(seed)
  m = data.shape[1]  # number of training examples
  mini_batches = []

  # Step 1: Shuffle (data, label)
  permutation = list(np.random.permutation(m))
  shuffled_X = data[:, permutation]
  shuffled_Y = label[:, permutation].reshape((1, m))

  # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
  # number of mini batches of size mini_batch_size in your partitioning
  num_complete_mini_batches = m // mini_batch_size
  for k in range(0, num_complete_mini_batches):
    mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
    mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)

  # Handling the end case (last mini-batch < mini_batch_size)
  if m % mini_batch_size != 0:
    mini_batch_X = shuffled_X[:, num_complete_mini_batches * mini_batch_size: m]
    mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * mini_batch_size: m]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)

  return mini_batches


def init_parameters(layer_dims):
  """ initial paras ops
  Paras
  -----------------------------------
  layer_dims: list, the number of units in each layer (dimension)

  Returns
  -----------------------------------
  dictionary: storage parameters w1,w2...wL, b1,...bL
  """
  np.random.seed(10)
  L = len(layer_dims)
  paras = {}
  for l in range(1, L):
    paras["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(
      2 / layer_dims[l - 1])  # he initialization

    paras["b" + str(l)] = np.zeros((layer_dims[l], 1))
  return paras


def forward_propagation(x, parameters):
  """ forward propagation function
  Paras
  ------------------------------------
  x:          input dataset, of shape (input size, number of examples)

  parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2",...,"WL", "bL"
              W -- weight matrix of shape (size of current layer, size of previous layer)
              b -- bias vector of shape (size of current layer,1)

  Returns
  ------------------------------------
  y:          the output of the last Layer(y_predict)
  caches:     list, every element is a tuple:(W,b,z,A_pre)
  """
  L = len(parameters) // 2  # number of layer
  A = x
  caches = [(None, None, None, x)]
  # calculate from 1 to L-1 layer
  for l in range(1, L):
    A_pre = A

    W = parameters["W" + str(l)]
    b = parameters["b" + str(l)]
    z = np.dot(W, A_pre) + b  # cal z = wx + b

    A = relu(z)  # relu activation function

    caches.append((W, b, z, A))

  # calculate Lth layer
  W = parameters["W" + str(L)]
  b = parameters["b" + str(L)]
  z = np.dot(W, A) + b

  y = sigmoid(z)
  caches.append((W, b, z, y))

  return y, caches


def backward_propagation(data, label, caches):
  """ implement the backward propagation presented.
  Paras
  ------------------------------------
  data:   input dataset, of shape (input size, number of examples)
  label:  true "label" vector (containing 0 if cat, 1 if non-cat)
  caches: caches output from forward_propagation(),(W,b,z,pre_A)

  Returns
  ------------------------------------
  gradients -- A dictionary with the gradients with respect to dW,db
  """
  m = label.shape[1]
  L = len(caches) - 1
  # calculate the Lth layer gradients
  prev_AL = caches[L - 1][3]
  dzL = 1. / m * (data - label)
  dWL = np.dot(dzL, prev_AL.T)
  dbL = np.sum(dzL, axis=1, keepdims=True)
  gradients = {"dW" + str(L): dWL, "db" + str(L): dbL}
  # calculate from L-1 to 1 layer gradients
  for l in reversed(range(1, L)):  # L-1,L-3,....,1
    post_W = caches[l + 1][0]  # use later layer para W
    dz = dzL  # use later layer para dz

    dal = np.dot(post_W.T, dz)
    z = caches[l][2]  # use layer z
    dzl = np.multiply(dal, relu_backward(z))
    prev_A = caches[l - 1][3]  # user before layer para A
    dWl = np.dot(dzl, prev_A.T)
    dbl = np.sum(dzl, axis=1, keepdims=True)

    gradients["dW" + str(l)] = dWl
    gradients["db" + str(l)] = dbl
    dzL = dzl  # update para dz
  return gradients


def compute_loss(pred, label):
  """calculate loss function
  Paras
  ------------------------------------
  pred:  pred "label" vector (containing 0 if cat, 1 if non-cat)

  label: true "label" vector (containing 0 if cat, 1 if non-cat)

  Returns
  ------------------------------------
  loss:  the difference between the true and predicted values
  """
  loss = 1. / label.shape[1] * np.nansum(np.multiply(-np.log(pred), label) + np.multiply(-np.log(1 - pred), 1 - label))

  return np.squeeze(loss)


def update_parameters_with_sgd(parameters, grads, learning_rate):
  """ update parameters using SGD
  ```
	VdW = beta * VdW - learning_rate * dW
	Vdb = beta * Vdb - learning_rate * db
	W = W + beta * VdW - learning_rate * dW
	b = b + beta * Vdb - learning_rate * db
	```
  Paras
  ------------------------------------
  parameters:    python dictionary containing your parameters:
                 parameters['W' + str(l)] = Wl
                 parameters['b' + str(l)] = bl
  grads:         python dictionary containing your gradients for each parameters:
                 grads['dW' + str(l)] = dWl
                 grads['db' + str(l)] = dbl
  learning_rate: the learning rate, scalar.

  Returns
  ------------------------------------
  parameters:     python dictionary containing your updated parameters

  """
  L = len(parameters) // 2
  for l in range(L):
    parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
    parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
  return parameters
