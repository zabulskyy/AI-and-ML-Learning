from XORNeuralNetwork import *
from NeuralNetwork import *

# test trivial functions
for i in data:
    x1 = i["ip"][0]
    x2 = i["ip"][1]
    nswr = i["op"]
    assert (xor_without_weights(x1, x2) == nswr), "FU"
    assert (xor_with_weights(x1, x2) == nswr), "FU"

# test not trained nn
expected = []
actual = []


nn = XORNeuralNetwork()
generations = 100000
print("error for not trained NN:\n" + str(nn.compute_error()))

# train
for i in range(generations):
    nn.train()
print("---")
print("error with {0} generations:\n".format(generations), str(nn.compute_error()))
