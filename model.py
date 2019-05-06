"""Implement some basic operations of Model.
"""

####################################################
# Author:  <Changyu Liu>shiyipaisizuo@gmail.com
# License: MIT
####################################################

from ops import *
from matplotlib import pyplot as plt


def model(data,
          label,
          layer_dims,
          lr,
          iters):
  """ define basic model
  Paras
  -----------------------------------
  data:            input data, of shape (input size, number of examples)
  label:           true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  layer_dims:      list containing the input size and each layer size
  learning_rate:   the learning rate, scalar
  num_iterations:  number of iterative training
  mini_batch_size: size of the mini-batches, integer

  Returns:
  -----------------------------------
  paras：      final paras:(W,b)
  """
  global loss
  losses = []
  # initialize paras
  paras = init_paras(layer_dims)
  for i in range(0, iters):
    # Forward propagation
    pred, caches = forward_propagation(data, paras)
    # Compute cost
    loss = compute_loss(pred, label)
    if i % 1000 == 0:
      print(f"Iter {i} loss {loss:.6f}")
      losses.append(loss)
    # Backward propagation
    grads = backward_propagation(pred, label, caches)
    # update parameters
    paras = update_parameters_with_sgd(paras, grads, lr)

  plt.clf()
  plt.plot(losses)
  plt.xlabel("iterations(thousand)")
  plt.ylabel("loss")
  plt.show()

  return paras


def predict(data, label, paras):
  """predict function
  Paras
  -----------------------------------
  data:            input data, of shape (input size, number of examples)
  label:           true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  parameter:       final parameters:(W,b)

  Returns
  -----------------------------------
  accuracy:        the correct value of the prediction
  """
  batch_size = label.shape[1]
  pred = np.zeros((1, batch_size))
  prob, _ = forward_propagation(data, paras)
  for i in range(prob.shape[1]):
    # Convert probabilities A[0,i] to actual predictions p[0,i]
    if prob[0, i] > 0.5:
      pred[0, i] = 1
    else:
      pred[0, i] = 0
  accuracy = 1 - np.mean(np.abs(pred - label))

  return accuracy


def dnn(X_train,
        y_train,
        X_test,
        y_test,
        layer_dims,
        learning_rate,
        iters):
  """DNN model
   Paras
  -----------------------------------
  X_train:         train data, of shape (input size, number of examples)
  y_train:         train "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  X_test:          test data, of shape (input size, number of examples)
  y_test:          test "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  layer_dims:      list containing the input size and each layer size
  learning_rate:   the learning rate, scalar
  num_iterations:  number of iterative training
  mini_batch_size: size of the mini-batches, integer

  Returns
  -----------------------------------
  accuracy:        the correct value of the prediction
  """
  paras = model(X_train,
                y_train,
                layer_dims,
                learning_rate,
                iters)

  accuracy = predict(X_test, y_test, paras)

  return accuracy
