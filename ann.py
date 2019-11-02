import numpy as np

#binary inputs for ANN
BINARY_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
BINARY_T = [[0], [1], [1], [0]]
#bipolar inputs for ANN
BIPOLAR_X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
BIPOLAR_T = [[-1], [1], [1], [-1]]

def binary_sigmoid(x):
    return 1/(1+np.exp(-x))

def binary_sigmoid_derivative(x):
    return self.binary_sigmoid(x)*(1 - self.binary_sigmoid(x))

def bipolar_sigmoid(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def bipolar_sigmoid_derivative(x):
    return 1 - self.bipolar_sigmoid(x)**2

def mse():
    squared_errors = []
    print('calculating MSE')
    for i, output_neuron in enumerate(self.layers[-1].neurons):
        print('adding squared error = ({} - {})^2'.format(self.targets[0][i], output_neuron.activation_value))
        squared_errors.append((self.targets[0][i] - output_neuron.activation_value)**2)
    print('returning (sum({}) / {})'.format(squared_errors, len(squared_errors)))
    return sum(squared_errors)/len(squared_errors)

#number of nodes in layer
#first layer is input, last layer is output, all other are hidden layers
LAYER_SIZE = np.array([2, 4, 1])
