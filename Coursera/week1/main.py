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
    mu = np.mean(X_norm)
    sigma = np.std(X_norm)
    print(mu, sigma)
    X_norm = (X_norm - mu) / sigma
    X[:, 1:] = X_norm
    print(X)
    return X


if __name__ == "__main__":
    # y = 2x^2 + 3x + 0
    X = np.array([
        [1., -2., 4.],
        [1., -1., 1.],
        [1., 0., 0.],
        [1., 1., 1.],
        [1., 2., 4.],
        [1., 3., 9.],
    ])

    # X = normalize(X)

    y = np.array([2., -2., 0., 5., 14., 27.])
    theta = np.array([1., 2., 0.])
    a, iters = .095, 5000
    w, h = gradient_descent(X, y, theta, a, iters)
    print("w = {0}, err = {1}".format(w, h[-1]))
    print(compute_cost(X, y, theta))

    # y = 3x + 10
