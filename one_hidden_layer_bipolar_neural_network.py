from math import exp
from random import random
import numpy as np
import matplotlib.pyplot as plt


def bipolar_sigmoid(z, derivative=False):
    if derivative:
        return 0.5 * (1 + z) * (1 - z)
    return -1 + 2/(1+np.exp(-z))


class OneHiddenLayerBipolarNeuralNetwork:
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
        #learning rate
        self.learning_rate = alpha

    def feed_forward(self):
        #start from input and go forward
        self.net_input1 = np.dot(self.X, self.weights01) + self.bias01
        self.activation1 = bipolar_sigmoid(self.net_input1)
        #use activation from previous layer
        self.net_input2 = np.dot(self.activation1, self.weights12) + self.bias12
        self.Y = bipolar_sigmoid(self.net_input2)
        return self.Y

    def back_propigation(self):
        #start from the output layer and work backward
        #calculate gradient of loss with respect to net input for output layer (dJ/dZ2)
        #dJ/dZ2 = dJ/dY x dY/dZ2
        #grad of J wrt Y = dJ/dY(||T-Y|||^2) = 2(T-Y)
        #grad of Y wrt Z2 = dY/dZ2(1/(1+np.exp(-z))) = 1+np.exp(-z)) * (1 - 1+np.exp(-z)))
        dJdZ2 = 2*(self.T - self.Y) * bipolar_sigmoid(self.Y, derivative=True)
        #calculate gradient of loss with respect to weights from 1 to 2
        #dJ/dW12 = dZ2/dW12 x dJ/dZ2
        #grad of J wrt dZ2 solved above
        #grad of Z2 wrt W12 = (A1)T
        dJdW12 = np.dot(self.activation1.T, dJdZ2)
        dJdB12 = dJdZ2
        #calculate gradient of loss with respect to net input for layer 1 (dJ/dZ1)
        #dJ/dZ1 = f'(Z1) x (W12)T x dJ/dZ2
        dJdZ1 = bipolar_sigmoid(self.activation1, derivative=True) * self.weights12.T * dJdZ2
        #dJ/dW01 = (A1)T x dJdZ1
        dJdW01 = np.dot(self.X.T, dJdZ1)
        dJdB01 = dJdZ1

        #adjust weights and biases
        self.weights01 = self.weights01 + self.learning_rate * dJdW01
        self.bias01 = self.bias01 + self.learning_rate * dJdB01
        self.weights12 = self.weights12 + self.learning_rate * dJdW12
        self.bias12 = self.bias12 + self.learning_rate * dJdB12

#binary inputs for ANN
BIPOLAR_X = np.array(([-1, -1], [-1, 1], [1, -1], [1, 1]))
BIPOLAR_T = np.array(([-1], [1], [1], [-1]))

#Define network with four hidden layers, learning rate 0.1
NN = OneHiddenLayerBipolarNeuralNetwork(BIPOLAR_X, BIPOLAR_T, 4, 0.1)

#set up variables for training loop
last_cost = None
cost_diff = 1
count = 0

#run the neural network once get the starting cost
output_values = NN.feed_forward()
last_cost = np.mean(np.square(BIPOLAR_T - output_values))
NN.back_propigation()

#variables to keep track of cost 
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

#Create a graph of the cost over the iterations
plt.figure()
plt.plot(iteration_no, cost_for_iter, 'b-', label='Bipolar Neural Network Cost per iteration')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.legend(loc='upper left')
plt.savefig('bipolar_neural_network.png')
