from math import exp
from random import random
import numpy as np

#binary inputs for ANN
BINARY_X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]))
BINARY_T = np.array(([0], [1], [1], [0]))


def sigmoid(z, derivative=False):
    if derivative:
        return z * (1 - z)
    return 1/(1+np.exp(-z))


class OneHiddenLayerNeuralNetwork:
    def __init__(self, X, T, H, alpha):
        self.X = X
        self.T = T
        self.num_hidden = H
        self.Y = np.zeros(self.T.shape)
        #weights and bias between layer 0 and 1
        self.weights01 = np.random.rand(self.X.shape[1], self.num_hidden)
        self.bias01 = np.random.rand()
        #weights between layer 1 and 2
        self.weights12 = np.random.rand(self.num_hidden, self.T.shape[1])
        self.bias12 = np.random.rand()
        self.learning_rate = alpha

    def feed_forward(self):
        #start from input and go forward
        self.net_input1 = np.dot(self.X, self.weights01) + self.bias01
        self.activation1 = sigmoid(self.net_input1)
        #use activation from previous layer
        self.net_input2 = np.dot(self.activation1, self.weights12) + self.bias12
        self.Y = sigmoid(self.net_input2)
        return self.Y

    def back_propigation(self):
        #start from the output layer and work backward
        #calculate gradient of loss with respect to net input for output layer (dJ/dZ2)
        #dJ/dZ2 = dJ/dY x dY/dZ2
        #grad of J wrt Y = dJ/dY(||T-Y|||^2) = 2(T-Y)
        #grad of Y wrt Z2 = dY/dZ2(1/(1+np.exp(-z))) = 1+np.exp(-z)) * (1 - 1+np.exp(-z)))
        dJdZ2 = 2*(self.T - self.Y) * sigmoid(self.Y, derivative=True)
        #calculate gradient of loss with respect to weights from 1 to 2
        #dJ/dW12 = dZ2/dW12 x dJ/dZ2
        #grad of J wrt dZ2 solved above
        #grad of Z2 wrt W12 = (A1)T
        dJdW12 = np.dot(self.activation1.T, dJdZ2)
        dJdB12 = dJdZ2
        #calculate gradient of loss with respect to net input for layer 1 (dJ/dZ1)
        #dJ/dZ1 = f'(Z1) x (W12)T x dJ/dZ2
        dJdZ1 = sigmoid(self.activation1, derivative=True) * self.weights12.T * dJdZ2
        #dJ/dW01 = (A1)T x dJdZ1
        dJdW01 = np.dot(self.X.T, dJdZ1)
        dJdB01 = dJdZ1

        #adjust weights and biases
        self.weights01 = self.weights01 + self.learning_rate * dJdW01
        self.bias01 = self.bias01 + self.learning_rate * dJdB01
        self.weights12 = self.weights12 + self.learning_rate * dJdW12
        self.bias12 = self.bias12 + self.learning_rate * dJdB12


NN = OneHiddenLayerNeuralNetwork(BINARY_X, BINARY_T, 4, 0.1)
last_cost = None
cost_diff = 1

def_calculate_cost():
    pass

while cost_diff > .0000001:
    output_values = NN.feed_forward()
    NN.back_propigation()
    print(last_cost)
    if not last_cost:
        last_cost = cost
        continue
    cost_diff = cost - last_cost
    print(cost_diff)
    last_cost = cost
