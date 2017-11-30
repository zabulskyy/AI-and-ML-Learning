from random import random
import math


class XORNeuralNetwork:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                "x1_h1": random(),
                "x2_h1": random(),
                "b_h1": random(),

                "x1_h2": random(),
                "x2_h2": random(),
                "b_h2": random(),

                "h1_o1": random(),
                "h2_o1": random(),
                "b_o1": random(),
            }
        else:
            self.weights = weights
        self.data = [
            {'ip': [0, 0], 'op': 0},
            {'ip': [1, 0], 'op': 1},
            {'ip': [0, 1], 'op': 1},
            {'ip': [1, 1], 'op': 0},
        ]

    def compute_result(self, x1, x2):
        h1 = sigmoid_activation(self.weights["b_h1"] + self.weights["x1_h1"] * x1 + self.weights["x2_h1"] * x2)
        h2 = sigmoid_activation(self.weights["b_h2"] + self.weights["x1_h2"] * x1 + self.weights["x2_h2"] * x2)
        o1 = sigmoid_activation(self.weights["b_o1"] + self.weights["h1_o1"] * h1 + self.weights["h2_o1"] * h2)
        return o1

    def train(self):
        deltas = self.fit()
        for i in self.weights:
            self.weights[i] += deltas[i]

    def compute_error(self):
        result = list()
        expected = list()
        for i in self.data:
            x1 = i['ip'][0]
            x2 = i['ip'][1]
            expected.append(i['op'])
            result.append(self.compute_result(x1, x2))
        return error(result, expected)

    def fit(self):
        # define data
        weights = self.weights
        weight_deltas = {
            "x1_h1": 0,
            "x2_h1": 0,
            "b_h1": 0,

            "x1_h2": 0,
            "x2_h2": 0,
            "b_h2": 0,

            "h1_o1": 0,
            "h2_o1": 0,
            "b_o1": 0,
        }

        # define functions
        for i in self.data:
            x1 = i["ip"][0]  # input 1
            x2 = i["ip"][1]  # input 2
            a = i["op"]  # correct answer
            h1_input = \
                weights["x1_h1"] * x1 + \
                weights["x2_h1"] * x2 + \
                weights["b_h1"]
            h1 = sigmoid_activation(h1_input)

            h2_input = \
                weights["x1_h2"] * x1 + \
                weights["x2_h2"] * x2 + \
                weights["b_h2"]
            h2 = sigmoid_activation(h2_input)

            o1_input = \
                weights["h1_o1"] * h1 + \
                weights["h2_o1"] * h2 + \
                weights["b_o1"]

            # prediction of our NN
            o1 = sigmoid_activation(o1_input)

            # start learning
            # calculating deltas
            delta = a - o1
            o1_delta = delta * derivative_sigmoid(o1_input)
            h1_delta = o1_delta * derivative_sigmoid(h1_input)
            h2_delta = o1_delta * derivative_sigmoid(h2_input)

            # trying to fit weights for h1 and h2
            weight_deltas["h1_o1"] += h1 * o1_delta
            weight_deltas["h2_o1"] += h2 * o1_delta
            weight_deltas["b_o1"] += o1_delta

            # trying to fit weights of x1 and x2 with respect to h1 and h2
            weight_deltas["x1_h1"] += x1 * h1_delta
            weight_deltas["x2_h1"] += x2 * h1_delta
            weight_deltas["b_h1"] += h1_delta
            weight_deltas["x1_h2"] += x1 * h2_delta
            weight_deltas["x2_h2"] += x2 * h2_delta
            weight_deltas["b_h2"] += h2_delta

        return weight_deltas


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
