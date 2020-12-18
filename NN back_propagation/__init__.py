import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import io
import scipy.optimize as opt
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder


def load_data(path, tranpose=True):
    data = io.loadmat('ex4data1.mat')
    y = np.array(data['y'])
    X = np.array(data['X'])
    if tranpose:  # 将原数据做一次翻转
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    return X, y


# X, y = load_data('ex4data1.mat')
#
#
# def plot_100_image(X):
#     size = int(np.sqrt(X.shape[1]))  # 20
#     sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
#     sample_images = X[sample_idx, :]
#     fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(6, 6))
#     for r in range(10):
#         for c in range(10):
#             ax[r, c].matshow(sample_images[10 * r + c].reshape((size, size)), cmap=matplotlib.cm.binary)
#             plt.xticks(np.array([]))
#             plt.yticks(np.array([]))
#
#
# plot_100_image(X)
# plt.show()
X, y_raw = load_data('ex4data1.mat', tranpose=False)
X = np.insert(X, 0, 1, axis=1)  # (5000,401)
y = OneHotEncoder(sparse=False).fit_transform(y_raw)  # (5000,10)


def serialize(a, b):
    return np.concatenate((a.ravel(), b.ravel()))


def deserialize(seq):
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def feed_forward(theta, X):
    t1, t2 = deserialize(theta)
    a1 = X

    z2 = a1 @ t1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)  # (5000,26)

    z3 = a2 @ t2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


data = io.loadmat('ex4weights.mat')
theta1, theta2 = data['Theta1'], data['Theta2']  # (25,401) (10,26)
theta = serialize(theta1, theta2)  # (10285,)


def cost(theta, X, y):
    m = X.shape[0]
    _, _, _, _, h = feed_forward(theta, X)  # h (5000,10)
    first = (-y) * np.log(h)
    second = (1 - y) * np.log(1 - h)
    return np.sum(first - second) / m


# print(cost(theta, X, y))  # 0.2876
def re_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()
    reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()
    return cost(theta, X, y) + reg_t1 + reg_t2


def gradient(theta, X, y):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    delta1 = np.zeros(t1.shape)  # (25,401)
    delta2 = np.zeros(t2.shape)  # (10,26)
    a1, z2, a2, a3, h = feed_forward(theta, X)  # h(5000,10)

    for i in range(m):
        a1i = a1[i, :]  # (401,)

        a2i = a2[i, :]  # (26,)

        hi = h[i, :]  # (10,)
        yi = y[i, :]  # (10,)

        d3i = hi - yi  # (10，)

        d2i = (t2.T @ d3i) * a2i * (1 - a2i)  # (26,) 目前这里是最难的，不知道首项是怎么来的

        delta2 += d3i[:, None] @ a2i[None, :]
        delta1 += d2i[1:, None] @ a1i[None, :]
    delta1 = delta1 / m
    delta2 = delta2 / m
    return serialize(delta1, delta2)


d1, d2 = deserialize(gradient(theta, X, y))
print(d1.shape, d2.shape)
