"""Stochastic optimization methods for MLP
"""

# Author: Changyu Liu <shiyipaisizuo@gmail.com>
# License: MIT

import numpy as np


class BaseOptimizer(object):
  """Base (Stochastic) gradient descent optimizer

      Parameters
      ----------
      params : list, length = len(coefs_) + len(intercepts_)
          The concatenated list containing coefs_ and intercepts_ in MLP model.
          Used for initializing velocities and updating params

      learning_rate_init : float, optional, default 0.1
          The initial learning rate used. It controls the step-size in updating
          the weights

      Attributes
      ----------
      learning_rate : float
          the current learning rate
      """
  def __init__(self, params, learning_rate_init=0.1):
    self.params = [param for param in params]
    self.learning_rate_init = learning_rate_init
    self.learning_rate = float(learning_rate_init)

  def update_params(self, grads):
    """Update parameters with given gradients

    Parameters
    ----------
    grads : list, length = len(params)
      Containing gradients with respect to coefs_ and intercepts_ in MLP
      model. So length should be aligned with params
    """
    updates = self._get_updates(grads)
    for param, update in zip(self.params, updates):
      param += update

  def iteration_ends(self, time_step):
    """Perform update to learning rate and potentially other states at the
            end of an iteration
            """
    pass

  def trigger_stopping(msg, verbose):
    """Decides whether it is time to stop training

    Parameters
    ----------
    msg : str
        Message passed in for verbose output

    verbose : bool
        Print message to stdin if True

    Returns
    -------
    is_stopping : bool
      True if training needs to stop
    """
    if verbose:
      print(msg + " Stopping.")
    return True
