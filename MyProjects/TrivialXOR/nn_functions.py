import math


def error(*args):
    first = args[0]
    second = args[1]
    n = len(first)
    res = 0
    for i in range(n):
        res += (first[i] - second[i]) ** 2
    return res / n


def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    fx = sigmoid_activation(x)
    return fx * (1 - fx)
