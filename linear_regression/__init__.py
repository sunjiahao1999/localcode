import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ex1data1.txt', header=None, names=['population', 'profit'])  # read_csv是读取以逗号（或其他）分隔的文件

# print(data.head())

# data.plot(kind='scatter',x='population',y='profit',figsize=(5,4))
# plt.show()

data.insert(0, 'ones', 1)

# print(data.head())

X = np.array(data.iloc[:, :-1].values)
y = np.array(data.iloc[:, -1].values)
theta = np.zeros(2)


# 计算代价函数
def J_cost(X, y, theta):
    inner = np.power(X @ theta - y, 2)
    return np.sum(inner) / (2 * X.shape[0])


init_cost = J_cost(X, y, theta)


# print(cost)
# 梯度下降
def gradient_descent(X, y, theta, alpha=0.01, epoch=1000):
    _theta = theta.copy()
    cost_data = [init_cost]  # 描述学习曲线
    for i in range(epoch):
        _theta = _theta - (alpha / X.shape[0]) * (X.T @ (X @ _theta - y))
        cost_data.append(J_cost(X, y, _theta))
    return _theta, cost_data


final_theta, cost_data = gradient_descent(X, y, theta)
# print(final_theta, cost_data[-1])
x = np.linspace(data.population.min(), data.population.max(), 100)
f = final_theta[0] + final_theta[1] * x

# fig, ax = plt.subplots(figsize=(5, 4))
# ax.plot(x, f, 'r', label='linear_regression')
# ax.scatter(data.population, data.profit, label='training_data')
# ax.legend(loc='best')
# plt.xlabel('population')
# plt.ylabel('profit')
# plt.show()

# fig, ax = plt.subplots(figsize=(5, 4))
# ax.plot(np.arange(1000), cost_data[1:], 'r')
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()


# 使用sklearn库
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X, y)

x = np.array(X[:, 1])
f = model.predict(X)


# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.population, data.profit, label='Traning Data')
# ax.legend(loc=2)
# ax.set_xlabel('Population')
# ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()


# 正规方程
def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta


final_theta2 = normalEqn(X, y)  # 感觉和批量梯度下降的theta的值有点差距
print(final_theta2)
