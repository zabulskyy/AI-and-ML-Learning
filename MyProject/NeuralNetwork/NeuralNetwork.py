from random import seed, random
import math

seed("test_XOR_seed")

''' Testing XOR '''


data = [
    {'ip': [0, 0], 'op': 0},
    {'ip': [1, 0], 'op': 1},
    {'ip': [0, 1], 'op': 1},
    {'ip': [1, 1], 'op': 0},
]

''' Trivial XOR'''


def simple_activation(x):
    return 1 if x >= .5 else 0


def error(*args):
    first = args[0]
    second = args[1]
    n = len(first)
    res = 0
    for i in range(n):
        res += (first[i] - second[i]) ** 2
    return res / n


def xor_without_weights(x1, x2):
    h1 = simple_activation(x1 - x2)
    h2 = simple_activation(-x1 + x2)
    return simple_activation(h1 + h2)


def xor_with_weights(x1, x2):
    weights = {
        "x1_h1": 1,
        "x2_h1": -1,
        "x1_h2": -1,
        "x2_h2": 1,
        "b_h1": 0,
        "b_h2": 0,
    }
    h1 = simple_activation(weights["b_h1"] + weights["x1_h1"] * x1 + weights["x2_h1"] * x2)
    h2 = simple_activation(weights["b_h2"] + weights["x1_h2"] * x1 + weights["x2_h2"] * x2)
    return simple_activation(h1 + h2)
