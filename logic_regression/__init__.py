import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['exam1', 'exam2', 'admitted'])
# print(data.head())

# 用isin索引
positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]


# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(positive['exam1'], positive['exam2'], s=10, c='b', marker='o', label='admitted')
# ax.scatter(negative['exam1'], negative['exam2'], s=10, c='r', marker='x', label='not admitted')
# ax.legend(prop={'size': 6})
# ax.set_xlabel('exam1 score')
# ax.set_ylabel('exam2 score')
# plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# z = np.linspace(-10, 10, 100)
# plt.plot(z, sigmoid(z), 'r')
# plt.show()

def cost(theta, X, y):
    first = np.multiply(-y, np.log(sigmoid(X @ theta)))  # np.multiply是对应点相乘可用*代替
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    return np.sum(first - second) / X.shape[0]


data.insert(0, 'ones', 1)
# print(data.head())
X = np.array(data.iloc[:, :-1].values)
y = np.array(data.iloc[:, -1].values)
theta = np.zeros(X.shape[1])


# print(cost(theta, X, y))


# 仅仅计算梯度,注意推导
def gradient(theta, X, y):
    return X.T @ (sigmoid(X @ theta) - y) / X.shape[0]


# print(gradient(theta, X, y))
# result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))  # 不知道为什么会报出中间过程


# print(result)
# print(cost(result[0], X, y))
# 注意这里的列表生成式
def predict(theta, X):
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_final = np.array([-25.16131853, 0.20623159, 0.20147149])  # 由result[0]得到
predictions = predict(theta_final, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
# print('准确率为 %s %%' % (accuracy * 100))  # z注意格式化表达
###############################################################
path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
# print(data2.head())

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()

degree = 4
x1 = data2['Test 1']
x2 = data2['Test 2']
for i in range(degree):
    for j in range(degree):
        data2['F' + str(i) + str(j)] = np.power(x1, i) * np.power(x2, j)

data2.drop('Test 1', axis=1, inplace=True)
data2.drop('Test 2', axis=1, inplace=True)
# print(data2.head())
X = np.array(data2.iloc[:, 1:].values)
y = np.array(data2.iloc[:, 0].values)
theta = np.zeros(X.shape[1])
lamda = 1


def cost_R(theta, X, y, lamda):
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))
    reg = (lamda / (2 * X.shape[0]) * np.sum(np.power(theta[1:], 2)))  # 不对theta_0做归正则化
    return np.sum(first - second) / X.shape[0] + reg


# print(cost_R(theta, X, y, 1))
def gradient_R(theta, X, y, lamda):
    iter_ = theta.shape[0]
    grad = np.zeros(iter_)
    for j in range(iter_):
        term = (sigmoid(X @ theta) - y) * X[:, j]
        if j == 0:
            grad[j] = np.sum(term) / X.shape[0]
        else:
            grad[j] = np.sum(term) / X.shape[0] + (lamda / X.shape[0]) * theta[j]
    return grad


# print(gradient_R(theta, X, y, lamda))
result2 = opt.fmin_tnc(func=cost_R, x0=theta, fprime=gradient_R, args=(X, y, lamda))
# print(result2)
theta_final = np.array(result2[0])
predictions = predict(theta_final, X)
correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print('准确率为 %s %%' % (accuracy * 100))  # z注意格式化表达
from sklearn import linear_model  # 调用sklearn的线性回归包

model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y)
print(model.score(X,y))
