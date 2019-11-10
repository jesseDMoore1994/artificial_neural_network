from math import exp
from random import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z, derivative=False):
    if derivative:
        return z * (1 - z)
    return 1/(1+np.exp(-z))


class OneHiddenLayerBinaryNeuralNetwork:
    def __init__(self, X, T, H, alpha):
        self.X = X
        self.T = T
        self.num_hidden = H
        self.Y = np.zeros(self.T.shape)
        self.weights01 = np.random.rand(self.X.shape[1], self.num_hidden)
        self.bias01 = np.random.rand()
        self.weights12 = np.random.rand(self.num_hidden, self.T.shape[1])
        self.bias12 = np.random.rand()
        self.learning_rate = alpha

    def feed_forward(self):
        self.net_input1 = np.dot(self.X, self.weights01) + self.bias01
        self.activation1 = sigmoid(self.net_input1)
        self.net_input2 = np.dot(self.activation1, self.weights12) + self.bias12
        self.Y = sigmoid(self.net_input2)
        return self.Y

    def back_propigation(self):
        dj_dz2 = 2*(self.T - self.Y) * sigmoid(self.Y, derivative=True)
        dj_dw12 = np.dot(self.activation1.T, dj_dz2)
        dj_db12 = dj_dz2
        dj_dz1 = np.dot(dj_dz2, self.weights12.T) * sigmoid(self.activation1, derivative=True)
        dj_dw01 = np.dot(self.X.T, dj_dz1)
        dj_db01 = dj_dz1

        self.weights01 = self.weights01 + self.learning_rate * dj_dw01
        self.bias01 = self.bias01 + self.learning_rate * dj_db01
        self.weights12 = self.weights12 + self.learning_rate * dj_dw12
        self.bias12 = self.bias12 + self.learning_rate * dj_db12

BINARY_X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
BINARY_T = np.array(([0], [1], [1], [0]))

NN = OneHiddenLayerBinaryNeuralNetwork(BINARY_X, BINARY_T, 4, 0.1)

last_cost = None
cost_diff = 1
count = 0

output_values = NN.feed_forward()
last_cost = np.mean(np.square(BINARY_T - output_values))
NN.back_propigation()

iteration_no = []
cost_for_iter = []


while np.sqrt(np.square(cost_diff)) > .000001:
    output_values = NN.feed_forward()
    cost = np.mean(np.square(BINARY_T - output_values))
    NN.back_propigation()
    cost_diff = cost - last_cost
    if count % 100 == 0:
        print("training iteration: {}".format(count))
        print("input for network:\n{}".format(BINARY_X))
        print("output for network:\n{}".format(output_values))
        print("target for network:\n{}".format(BINARY_T))
        print("Current Cost: {}".format(cost))
        print("Cost diff from last iteration: {}".format(cost_diff))
        iteration_no.append(count)
        cost_for_iter.append(cost)
    last_cost = cost
    count = count + 1

plt.figure()
plt.plot(iteration_no, cost_for_iter, 'b-', label='Binary Neural Network Cost per iteration')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.legend(loc='upper left')
plt.savefig('binary_neural_network.png')
