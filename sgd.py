from .base import BaseOptimizer

import numpy as np


class SGDOptimizer(BaseOptimizer):
    """Stochastic gradient descent optimizer with momentum

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
            The concatenated list containing coefs_ and intercepts_ in MLP model.
            Used for initializing velocities and updating params

    learning_rate_init : float, optional, default 0.1
            The initial learning rate used. It controls the step-size in updating
            the weights

    lr_schedule : {'constant', 'adaptive', 'invscaling'}, default 'constant'
            Learning rate schedule for weight updates.

            -'constant', is a constant learning rate given by
             'learning_rate_init'.

            -'invscaling' gradually decreases the learning rate 'learning_rate_' at
              each time step 't' using an inverse scaling exponent of 'power_t'.
              learning_rate_ = learning_rate_init / pow(t, power_t)

            -'adaptive', keeps the learning rate constant to
             'learning_rate_init' as long as the training keeps decreasing.
             Each time 2 consecutive epochs fail to decrease the training loss by
             tol, or fail to increase validation score by tol if 'early_stopping'
             is on, the current learning rate is divided by 5.

    momentum : float, optional, default 0.9
            Value of momentum used, must be larger than or equal to 0

    nesterov : bool, optional, default True
            Whether to use nesterov's momentum or not. Use nesterov's if True

    Attributes
    ----------
    learning_rate : float
            the current learning rate

    velocities : list, length = len(params)
            velocities that are used to update params
    """
    def __index__(self,
                  params,
                  learning_rate_init=0.1,
                  lr_schedule='constant',
                  momentum=0.9,
                  nesterov=True,
                  power_t=0.5):
      super(SGDOptimizer, self).__init__(params, learning_rate_init)

      self.lr_schedule = lr_schedule
      self.momentum = momentum
      self.nesterov = nesterov
      self.power_t = power_t
      self.velocities = [np.zeros_like(params) for param in params]

    def iteration_ends(self, time_step):
      """Perform updates to learning rate and potential other states at the
      end of an iteration

      Parameters
      ----------
      time_step : int
          number of training samples trained on so far, used to update
          learning rate for 'invscaling'
      """
      if self.lr_schedule == 'invscaling':
        self.learning_rate = (float(self.learning_rate_init) /
                              (time_step + 1) ** self.power_t)

    def trigger_stopping(self, msg, verbose):
        if self.lr_schedule != 'adaptive':
            if verbose:
                print(msg + " Stopping.")
            return True

        if self.learning_rate <= 1e-6:
            if verbose:
                print(msg + " Learning rate too small. Stopping.")
            return True

        self.learning_rate /= 5.
        if verbose:
            print(msg + " Setting learning rate to %f" %
                  self.learning_rate)
        return False

    def _get_updates(self, grads):
      """Get the values used to update params with given gradients

      Parameters
      ----------
      grads : list, length = len(coefs_) + len(intercepts_)
        Containing gradients with respect to coefs_ and intercepts_ in MLP
        model. So length should be aligned with params

      Returns
      -------
      updates : list, length = len(grads)
        The values to add to params
      """
      updates = [self.momentum * velocity - self.learning_rate * grad
                 for velocity, grad in zip(self.velocities, grads)]
      self.velocities = updates

      if self.nesterov:
        updates = [self.momentum * velocity - self.learning_rate * grad
                   for velocity, grad in zip(self.velocities, grads)]

      return updates
