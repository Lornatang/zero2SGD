from sklearn.linear_model import LinearRegression
from ops import compute_grad, update_beta, rmse

import pandas as pd
import numpy as np

# input data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
submit = pd.read_csv('data/sample_submit.csv')

# initial setup params
beta = [1, 1]
lr = 0.0001
tol_L = 0.01

# 对x进行归一化
max_x = max(train['id'])
x = train['id'] / max_x
y = train['questions']


# 进行第一次计算
np.random.seed(10)
grad = compute_grad(beta, x, y)
loss = rmse(beta, x, y)
beta = update_beta(beta, lr, grad)
loss_new = rmse(beta, x, y)

# 开始迭代
i = 1
while np.abs(loss_new - loss) > tol_L:
  beta = update_beta(beta, lr, grad)
  grad = compute_grad(beta, x, y)
  if i % 1000 == 0:
    loss = loss_new
    loss_new = rmse(beta, x, y)
    print(f'Round {i} Diff SGD {abs(loss_new - loss)}')
  i += 1


val = LinearRegression()
val.fit(train[['id']], train[['questions']])

print(f'Our     Coef: {beta[1] / max_x}')
print(f'Sklearn Coef: {val.coef_[0][0]}')

print(f'Our     Intercept: {beta[0]}')
print(f'Sklearn Intercept: {val.intercept_[0]}')

our_rmse = rmse(beta, x, y)
sklearn_rmse = rmse([val.intercept_[0], val.coef_[0][0]], train['id'], y)
print(f'Our     RMSE: {our_rmse}')
print(f"Sklearn RMSE: {sklearn_rmse}")
