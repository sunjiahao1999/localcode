import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize

data = loadmat('ex3data1.mat')


# print(data)
# print(data['X'].shape, data['y'].shape)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, lamda):  # 写完cost与grad之后一定要先验证一下有没有写对！！！
    first = (-y) * np.log(sigmoid(X @ theta))
    second = (1 - y) * np.log(1 - sigmoid(X @ theta))  # log在1-sigmoid外面呀！！！！！！！
    reg = (lamda / (2 * X.shape[0])) * np.sum(np.power(theta[1:], 2))
    return np.sum(first - second) / X.shape[0] + reg


def gradient(theta, X, y, lamda):
    error = sigmoid(X @ theta) - y
    grad = (X.T @ error) / X.shape[0] + (lamda / X.shape[0]) * theta
    # 常数项无正则化
    grad[0] = np.sum(np.multiply(error, X[:, 0])) / X.shape[0]
    return grad


def one_vs_all(X, y, num_labels, lamda):
    X = np.insert(X, 0, 1, axis=1)
    all_theta = np.zeros((X.shape[1], num_labels))
    for i in range(1, num_labels + 1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, lamda), method='TNC', jac=gradient)
        theta = np.array(fmin.x)
        all_theta[:, i - 1] = theta
    return all_theta


all_theta = one_vs_all(data['X'], data['y'], 10, 1)
print(all_theta[0, :])


def predict_all(X, all_theta):
    X = np.insert(X, 0, 1, axis=1)
    h = sigmoid(X @ all_theta)
    h_max = np.argmax(h, axis=1) + 1
    return h_max


y_pre = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pre, data['y'])]
accuracy = sum(correct) / len(correct)
print('准确率为%.2f%%' % (accuracy * 100))