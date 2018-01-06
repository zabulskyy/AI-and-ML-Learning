import numpy as np

np.random.seed(11)

XOR_data = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0],
])


def sigmoid(x):
    return 1 / (1 + np.e ** (-x))


def fit(h, X, y):
    return np.dot(X.T, (h - y).T)


def cost(h, y):
    h = h[0]

    def c(h, y):
        if abs(y - h) == 1:
            return np.infty
        return -np.log(h) if y == 1 else -np.log(1 - h)

    value = np.zeros(h.size)
    for i in range(len(y)):
        value[i] = c(h[i], y[i])
    return sum(value) / len(value)


def hypothesis(w, X):
    def g(z):
        return sigmoid(z)

    return g(np.dot(w, X))


X = np.ones((4, 7))
X[:, 1:3] = XOR_data[:, :2]
X[:, 3:5] = XOR_data[:, :2]
X[:, 5:] = XOR_data[:, :2]
y = XOR_data[:, 2]
w = np.random.rand(7, 1)

generations = 100000
a = 100

for i in range(generations):
    h = hypothesis(w.T, X.T)
    e = cost(h, y)
    w -= a / len(w) * fit(h, X, y)
print(w)
print(h[0])
print(e)
