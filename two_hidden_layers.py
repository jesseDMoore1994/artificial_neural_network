from math import exp
from random import random
import numpy as np
import matplotlib.pyplot as plt


def bipolar_sigmoid(z, derivative=False):
    if derivative:
        return 0.5 * (1 + z) * (1 - z)
    return -1 + 2/(1+np.exp(-z))


class TwoHiddenLayersBipolarNeuralNetwork:
    def __init__(self, X, T, H, H2, alpha):
        self.X = X
        self.T = T
        self.num_hidden1 = H
        self.num_hidden2 = H2
        self.Y = np.zeros(self.T.shape)
        self.weights01 = np.random.rand(self.X.shape[1], self.num_hidden1)
        self.bias01 = np.random.rand()
        self.weights12 = np.random.rand(self.num_hidden1, self.num_hidden2)
        self.bias12 = np.random.rand()
        self.weights23 = np.random.rand(self.num_hidden2, self.T.shape[1])
        self.bias23 = np.random.rand()
        self.learning_rate = alpha

    def feed_forward(self):
        self.net_input1 = np.dot(self.X, self.weights01) + self.bias01
        self.activation1 = bipolar_sigmoid(self.net_input1)
        self.net_input2 = np.dot(self.activation1, self.weights12) + self.bias12
        self.activation2 = bipolar_sigmoid(self.net_input2)
        self.net_input3 = np.dot(self.activation2, self.weights23) + self.bias23
        self.Y = bipolar_sigmoid(self.net_input3)
        return self.Y

    def back_propigation(self):
        dj_dz3 = 2*(self.T - self.Y) * bipolar_sigmoid(self.Y, derivative=True)
        dj_dw23 = np.dot(self.activation2.T, dj_dz3)
        dj_db23 = dj_dz3
        dj_dz2 = np.dot(dj_dz3, self.weights23.T) * bipolar_sigmoid(self.activation2, derivative=True)
        dj_dw12 = np.dot(self.activation1.T,  dj_dz2)
        dj_db12 = dj_dz2
        dj_dz1 = np.dot(dj_dz2, self.weights12.T) * bipolar_sigmoid(self.activation1, derivative=True)
        dj_dw01 = np.dot(self.X.T,  dj_dz1)
        dj_db01 = dj_dz1

        self.weights01 = self.weights01 + self.learning_rate * dj_dw01
        self.bias01 = self.bias01 + self.learning_rate * dj_db01
        self.weights12 = self.weights12 + self.learning_rate * dj_dw12
        self.bias12 = self.bias12 + self.learning_rate * dj_db12
        self.weights23 = self.weights23 + self.learning_rate * dj_dw23
        self.bias23 = self.bias23 + self.learning_rate * dj_db23

BIPOLAR_X = np.array(([-1, -1], [-1, 1], [1, -1], [1, 1]))
BIPOLAR_T = np.array(([-1], [1], [1], [-1]))

NN = TwoHiddenLayersBipolarNeuralNetwork(BIPOLAR_X, BIPOLAR_T, 3, 2, 0.1)

last_cost = None
cost_diff = 1
count = 0

output_values = NN.feed_forward()
last_cost = np.mean(np.square(BIPOLAR_T - output_values))
NN.back_propigation()

iteration_no = []
cost_for_iter = []


while np.sqrt(np.square(cost_diff)) > .000001:
    output_values = NN.feed_forward()
    cost = np.mean(np.square(BIPOLAR_T - output_values))
    NN.back_propigation()
    cost_diff = cost - last_cost
    if count % 100 == 0:
        print("training iteration: {}".format(count))
        print("input for network:\n{}".format(BIPOLAR_X))
        print("output for network:\n{}".format(output_values))
        print("target for network:\n{}".format(BIPOLAR_T))
        print("Current Cost: {}".format(cost))
        print("Cost diff from last iteration: {}".format(cost_diff))
        iteration_no.append(count)
        cost_for_iter.append(cost)
    last_cost = cost
    count = count + 1

plt.figure()
plt.plot(iteration_no, cost_for_iter, 'b-', label='Bipolar Neural Network Cost per iteration')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.legend(loc='upper left')
plt.savefig('deep_bipolar_neural_network.png')
