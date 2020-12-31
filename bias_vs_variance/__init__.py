import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    d = sio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])


X, y, Xval, yval, Xtest, ytest = load_data()  # X(12,) Xt(21,) Xv(21,) y(12,) yt(21,) yv(21,)
# plt.scatter(X, y, figsize=(6, 4))
# plt.xlabel('water_level')
# plt.ylabel('flow')
# plt.show()
X, Xval, Xtest = [np.insert(x[:, None], 0, 1, axis=1) for x in (X, Xval, Xtest)]
theta = np.ones(X.shape[1])  # (2,)


def cost(theta, X, y):
    m = X.shape[0]
    h = X @ theta
    return np.sum((h - y) ** 2) / (2 * m)


def re_cost(theta, X, y, l=1):
    return cost(theta, X, y) + l * np.sum(theta[1:] ** 2) / (2 * X.shape[0])


# cost(theta, X, y)  # 303.95
# re_cost(theta,X,y)
def gradient(theta, X, y):
    m = X.shape[0]
    error = X @ theta - y
    return X.T @ error / m


def re_gradient(theta, X, y, l=1):
    m = X.shape[0]
    theta_ = theta.copy()
    theta_[0] = 0
    return gradient(theta, X, y) + l * theta_ / m


# gradient(theta, X, y)  # [-15.3,598.2]
# re_gradient(theta, X, y)
def linear_regression(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=re_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=re_gradient,
                       options={'disp': 0})
    return res


final_theta = linear_regression(X, y, l=0).get('x')


# print(re_gradient(final_theta, X, y, 0))

# b = final_theta[0]  # intercept
# m = final_theta[1]  # slope
#
# plt.scatter(X[:, 1], y, label="Training data")
# plt.plot(X[:, 1], X[:, 1] * m + b, label="Prediction")
# plt.legend(loc=2)
# plt.show()
# train_cost, cv_cost = [], []
# m = X.shape[0]
# for i in range(1, m + 1):
#     #     print('i={}'.format(i))
#     res = linear_regression(X[:i, :], y[:i], l=0)
#
#     tc = re_cost(res.x, X[:i, :], y[:i], l=0)
#     cv = re_cost(res.x, Xval, yval, l=0)
#     #     print('tc={}, cv={}'.format(tc, cv))
#
#     train_cost.append(tc)
#     cv_cost.append(cv)


# plt.plot(np.arange(1, m + 1), train_cost, label='train')
# plt.plot(np.arange(1, m + 1), cv_cost, label='cv')
# plt.legend(loc='best')
# plt.show()


def prepare_poly_data(*args, power):
    """
    args: keep feeding in X, Xval, or Xtest
        will return in the same order
    """

    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)

        # normalization
        ndarr = normalize_feature(df).values

        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)

    return [prepare(x) for x in args]


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.values if as_ndarray else df


X, y, Xval, yval, Xtest, ytest = load_data()  # X(12,) Xt(21,) Xv(21,) y(12,) yt(21,) yv(21,)


# print(poly_features(X,3))
def normalize_feature(df):
    """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())


X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)


def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m + 1):
        res = linear_regression(X[:i, :], y[:i], l)
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc='best')


# plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)
# plt.show()

l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)

plt.xlabel('lambda')

plt.ylabel('cost')
plt.show()

print(l_candidate[np.argmin(cv_cost)])  # 应该通过交叉验证集去选定超参数
# use test data to compute the cost
# for l in l_candidate:
#     theta = linear_regression(X_poly, y, l).x
#     print('test cost(l={}) = {}'.format(l, cost(theta, Xtest_poly, ytest)))
