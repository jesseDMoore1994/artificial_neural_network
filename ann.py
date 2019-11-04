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

def binary_sigmoid_derivative(a):
    return a*(1 - a)

def bipolar_sigmoid(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def bipolar_sigmoid_derivative(x):
    return 1 - a**2

def create_weights_and_biases(layer_sizes):
    WB = {}
    #frst layer is ignored, since it is the input layer
    for layer in range(len(layer_sizes)):
        if layer == 0:
            continue
        W = np.random.randn(layer_sizes[layer], layer_sizes[layer-1])
        B = np.random.randn(layer_sizes[layer], 1)
        WB['W{}'.format(layer)] = W
        WB['B{}'.format(layer)] = B
    return WB

def feed_forward(input_vector, weights_and_biases, layer_sizes):
    n = input_vector
    wb = weights_and_biases
    cache = {}
    for layer in range(len(layer_sizes)):
        #frst layer is ignored, since it is the input layer
        if layer == 0:
            continue
        for node in range(layer_sizes[layer]):
            #calculate activation
            #ZXY = net input for node Y in layer X
            #AXY = activation for node Y in layer X
            #if layer is 1, use input vector, otherwise use previous activation
            if layer == 1:
                z = np.dot(wb['W1'][node], n) + wb['B1'][node]
                a = binary_sigmoid(z)
            else:
                z = np.dot(
                    #dot multiply the weights for the current node
                    wb['W{}'.format(layer)][node],
                    #with the activations from the previous layers nodes
                    np.array(
                        [ cache['A{}{}'.format(layer-1, y)] for y in range(layer_sizes[layer-1]) ]
                    )
                ) + wb['B{}'.format(layer)][node] #finally add the bias for the current node to get net input
                a = binary_sigmoid(z) #calculate activaton
            cache['Z{}{}'.format(layer, node)] = z
            cache['A{}{}'.format(layer, node)] = a
    #output is activation of final layer
    output = np.array([
            cache['A{}{}'.format(len(layer_sizes)-1, y)]
            for y in range(layer_sizes[-1])
    ])
    return output, cache

def calculate_error(output, target_vector):
    return (1/2*len(output))*(target_vector - output)**2

def back_propigate(output, target, cache, wb, layer_sizes):
    error = calculate_error(output, target)
    print('error')
    pp.pprint(error)
    grad_J_wrt_ZL = np.zeros_like(output)
    print(grad_J_wrt_ZL)
    for node_idx in range(layer_sizes[-1]):
        grad_J_wrt_ZL[node_idx] =


def train(input_vectors, weights_and_biases, target_vectors, layer_sizes):
    #apply each input vector to the neural network
    for in_v, target_v in zip(input_vectors, target_vectors):
        output, cache = feed_forward(in_v, weights_and_biases, layer_sizes)
        print('------------------------------------------')
        print('input')
        pp.pprint(in_v)
        print('target')
        pp.pprint(target_v)
        print('output')
        pp.pprint(output)
        print('weight bias vectors')
        pp.pprint(weights_and_biases)
        print('cache')
        pp.pprint(cache)
        back_propigate(output, target_v, cache, weights_and_biases, layer_sizes)

#number of nodes in layer
#first layer is input, last layer is output, all other are hidden layers
LAYER_SIZES = np.array([2, 4, 1])
WEIGHTS_AND_BIASES = create_weights_and_biases(LAYER_SIZES)
train(BINARY_X, WEIGHTS_AND_BIASES, BINARY_T, LAYER_SIZES)
