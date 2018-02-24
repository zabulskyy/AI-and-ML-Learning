import numpy as np


def compute_cost(X, y, w):
    h = sum((X * w).T)
    error = y - h
    J = sum(error ** 2) / 2 / len(y)
    return J


def gradient_descent(X, y, theta, a, iters):
    m = len(y)
    J_history = np.zeros((iters, 1))

    for i in range(iters):
        h = np.dot(X, theta)
        error = h - y
        coef = a / m
        s = sum(np.array([error]).T * X)
        dt = (s * coef).T
        theta -= dt

        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def normalize(X):
    X_norm = X[:, 1:]
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0)
    sub = np.subtract(X_norm, mu)
    X_norm = np.divide(sub, sigma)
    X[:, 1:] = X_norm
    return X


if 1:
    # y = 0x^2 + 3x + 10
    X = np.array([
        [1., -2., 4.],
        [1., -1., 1.],
        [1., 0., 0.],
        [1., 1., 1.],
        [1., 2., 4.],
        [1., 3., 9.],
    ])

    # X = normalize(X)

    y = np.array([4., 7., 10., 13., 16., 19.])
    theta = np.array([1., 2., 40])
    a, iters = 0.05, 1000
    w, h = gradient_descent(X, y, theta, a, iters)
    print("w = {0}".format(w))
    print(compute_cost(X, y, theta))


# x1 = np.arange(12.0).reshape((6, 2))
# x2 = np.arange(6.0).reshape((6, 1))
# print(x1, x2)
# a = np.subtract(x1, x2)
# print(a)