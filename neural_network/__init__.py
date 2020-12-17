import numpy as np
import matplotlib.pyplot
from scipy import io
import scipy.optimize as opt
from sklearn.metrics import classification_report

data1 = io.loadmat('ex3data1.mat')
X, y = np.array(data1['X']), np.array(data1['y'])
X = np.insert(X, 0, 1, axis=1)
data2 = io.loadmat('ex3weights.mat')
theta1, theta2 = np.array(data2['Theta1']), np.array(data2['Theta2'])


# print(theta1.shape, theta2.shape)
# print(X.shape, y.shape)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z2 = X @ theta1.T
a2 = sigmoid(z2)
a2 = np.insert(a2, 0, 1, axis=1)
# print(a2.shape)
z3 = a2 @ theta2.T
a3 = sigmoid(z3)
# print(a3.shape)
y_pre = np.argmax(a3, axis=1) + 1
print(classification_report(y,y_pre))
