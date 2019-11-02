import numpy as np
from pprint import PrettyPrinter
pp = PrettyPrinter()

#binary inputs for ANN
BINARY_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
BINARY_T = np.array([[0], [1], [1], [0]])
#bipolar inputs for ANN
BIPOLAR_X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
BIPOLAR_T = np.array([[-1], [1], [1], [-1]])

def binary_sigmoid(x):
    return 1/(1+np.exp(-x))

def binary_sigmoid_derivative(x):
    return self.binary_sigmoid(x)*(1 - self.binary_sigmoid(x))

def bipolar_sigmoid(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def bipolar_sigmoid_derivative(x):
    return 1 - self.bipolar_sigmoid(x)**2

def create_weights_and_biases(layer_sizes):
    WB = {}
    #frst layer is ignored, since it is the input layer
    for layer in range(len(layer_sizes)):
        if layer == 0:
            continue
        W = np.random.randn(layer_sizes[layer], layer_sizes[layer-1])
        B = np.zeros((layer_sizes[layer], 1))
        WB['W{}'.format(layer)] = W
        WB['B{}'.format(layer)] = B
    return WB


def feed_forward(input_vectors, weights_and_biases):
    pass

#number of nodes in layer
#first layer is input, last layer is output, all other are hidden layers
LAYER_SIZES = np.array([2, 4, 1])

WEIGHTS_AND_BIASES = create_weights_and_biases(LAYER_SIZES)
Y = feed_forward(BINARY_X, WEIGHTS_AND_BIASES)
pp.pprint(BINARY_X)
pp.pprint(BINARY_T)
pp.pprint(WEIGHTS_AND_BIASES)
pp.pprint(BINARY_Y)
