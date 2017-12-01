from random import random
import math
from nn_functions import *


class EONeuralNetwork:
    def __init__(self, weights):
        pass

    def compute_result(self):
        pass

    def train(self):
        deltas = self.fit()
        for i in self.weights:
            self.weights[i] += deltas[i]


class Neuron:
    pass
