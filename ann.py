from math import exp
from random import random
import numpy as np

#binary inputs for ANN
BINARY_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
BINARY_T = [[0], [1], [1], [0]]
#bipolar inputs for ANN
BIPOLAR_X = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
BIPOLAR_T = [[-1], [1], [1], [-1]]


class Neuron:
    def __init__(self):
        self.activation_value = None
        self.net_input = None
        self.dJ_wrt_Z = None
        self.weights_in = []
        self.dJs_wrt_Wi = []
        self.bias_in = None
        self.dJ_wrt_B = None

    def __str__(self):
        return "\n".join([
            "activation_value: {}".format(self.activation_value),
            "net_input: {}".format(self.net_input),
            "weights_in: {}".format(self.weights_in),
            "dJs_wrt_Wi: {}".format(self.dJs_wrt_Wi),
            "bias_in: {}".format(self.bias_in),
            "dJ_wrt_B: {}".format(self.dJ_wrt_B),
        ])


class BinaryNeuron(Neuron):
    def __init__(self):
        super().__init__()

    def binary_sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def binary_sigmoid_derivative(self, x):
        return x*(1 - x)

    def calculate_activation_and_store_input(self, net_input):
        self.net_input = net_input
        self.activation_value = self.binary_sigmoid(net_input)

    def calculate_deriv_activation_wrt_net_input(self):
        return self.binary_sigmoid_derivative(self.activation_value)

class BipolarNeuron(Neuron):
    def __init__(self):
        super().__init__()

    def bipolar_sigmoid(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def bipolar_sigmoid_derivative(self, x):
        return 1 - x**2

    def calculate_activation_and_store_input(self, net_input):
        self.net_input = net_input
        self.activation_value = self.bipolar_sigmoid(net_input)

    def calculate_deriv_activation_wrt_net_input(self):
        return self.bipolar_sigmoid_derivative(self.activation_value)


NEURON_MAP = {'BinaryNeuron': BinaryNeuron, 'BipolarNeuron': BipolarNeuron}


class NetworkLayer:
    def __init__(self, num_units, neuron_class):
        self.neurons = [ neuron_class() for unit in range(num_units) ]

    def __str__(self):
        strs = ["neuron({}):\n{}".format(i, neuron) for i, neuron in enumerate(self.neurons)]
        return '\n'.join(strs)

    def calculate_activation_for_neurons(self, previous_layer):
        for neuron in self.neurons:
            #Calculate net input for each neuron first
            net_input = 0
            for i, previous_neuron in enumerate(previous_layer.neurons):
                net_input = net_input + (neuron.weights_in[i]*previous_neuron.activation_value)
            net_input = net_input + neuron.bias_in
            #Calculate activation and store net input
            neuron.calculate_activation_and_store_input(net_input)


class InputLayer(NetworkLayer):
    def __init__(self, input_vector):
        super().__init__(len(input_vector), Neuron)


class OutputLayer(NetworkLayer):
    def __init__(self, target_vector, neuron_class):
        super().__init__(len(target_vector), neuron_class)


class NeuralNetwork:
    def __init__(self, config):
        #config members
        self.inputs = []
        self.targets = []
        self.output_neuron_type = ""
        self.hidden_layer_data = []
        self.layers = []

        #validate config
        self._validate_and_store_config(config)

        #construct layers
        self._construct_layers()

    def __str__(self):
        strs = ['Printing all the layers of the ANN.']
        for l, layer in enumerate(self.layers):
            strs.append("Layer {}:\n{}".format(l, layer))
        return '\n'.join(strs)

    def _has_valid_input_layer_params(self, config):
        assert config['input_layer']['input_vectors'], "Must define a set of input vectors for training."
        assert len(config['input_layer']['input_vectors']), "Must define at least one input vector"

    def _has_valid_output_layer_params(self, config):
        assert config['output_layer']['target_vectors'], "Must define a set of target vectors for training."
        assert len(config['output_layer']['target_vectors']), "Must define at least one input vector"
        assert config['output_layer']['neuron_type'], "Must define a neuron type for the output layer"

    def _input_and_target_lengths_match(self, config):
        assert (
            len(config['output_layer']['target_vectors']) == len(config['input_layer']['input_vectors'])
        ),"Number of target and input vectors must be equal."

    def _has_valid_hidden_layer_params(self, config):
        if config['hidden_layers']:
            for elem in config['hidden_layers']:
                assert elem['neuron_type'], "Must define a neuron type for each hidden layer."
                assert elem['num_units'], "Must define a number of units for each hidden layer."

    def _has_learning_rate(self, config):
        assert config['learning_rate'], "Must define a learning rate to use in gradient descent."

    def _validate_and_store_config(self, config):
        self._has_valid_input_layer_params(config)
        self._has_valid_output_layer_params(config)
        self._input_and_target_lengths_match(config)
        self._has_valid_hidden_layer_params(config)
        self._has_learning_rate(config)
        self.inputs = config['input_layer']['input_vectors']
        self.targets = config['output_layer']['target_vectors']
        self.output_neuron_type = config['output_layer']['neuron_type']
        self.hidden_layer_data = config['hidden_layers']
        self.learning_rate = config['learning_rate']

    def _construct_layers(self):
        self.layers.append(InputLayer(self.inputs[0]))

        for layer in self.hidden_layer_data:
            self.layers.append(NetworkLayer(layer['num_units'], NEURON_MAP[layer['neuron_type']]))

        self.layers.append(OutputLayer(self.targets[0], NEURON_MAP[layer['neuron_type']]))

        #create weights and biases
        for l, layer in enumerate(self.layers):
            if l == 0:
                #if l == 0 we are on first layer, no weights or biases associated
                continue
            layer.weights = self._create_weights(layer, self.layers[l-1])
            layer.biases = self._create_biases(layer)

    def _create_weights(self, this_layer, previous_layer):
        for neuron in this_layer.neurons:
            weights_on_neuron = []
            for previous_neuron in previous_layer.neurons:
                neuron.weights_in.append(np.random.randn())

    def _create_biases(self, this_layer):
        biases = []
        for neuron in this_layer.neurons:
            neuron.bias_in = 0
        return biases

    def feed_forward(self):
        #apply input vector to layer 0
        print("Applying vector {} to ANN, target is {}.".format(self.inputs[0], self.targets[0]))
        for i, val in enumerate(self.inputs[0]):
            self.layers[0].neurons[i].activation_value = val

        #create weights and biases
        for l, layer in enumerate(self.layers):
            if l == 0:
                #if l == 0 we are on first layer, value is the input
                continue
            layer.calculate_activation_for_neurons(self.layers[l-1])

        print("Output for {} is {}".format(self.inputs[0], [n.activation_value for n in self.layers[-1].neurons ]))


    # returns 1/n * sum((y - t)**2) where n is number of output values/target values
    def mse(self):
        squared_errors = []
        print('calculating MSE')
        for i, output_neuron in enumerate(self.layers[-1].neurons):
            print('adding squared error = ({} - {})^2'.format(self.targets[0][i], output_neuron.activation_value))
            squared_errors.append((self.targets[0][i] - output_neuron.activation_value)**2)
        print('returning (sum({}) / {})'.format(squared_errors, len(squared_errors)))
        return sum(squared_errors)/len(squared_errors)

    def deriv_mse_wrt_Yj(self, Yj, Tj, n):
    #Yj is output of jth neuron in output layer
    #Tj is target value of jth neuron in output layer
    #n is number of values in Y
    # dJ/dYj is d(MSE)/dYj = d/dYj( 1/n * [sum(Yj - Tj)**2 for j=1 to n] )
    # = 2/n * (Yj - Tj)
        return (2/n) * (Yj - Tj)

    def back_propigate(self):
        print('backprop target {}'.format(self.targets[0]))
        #layer counter
        l = len(self.layers) - 1
        #Start with the output layer
        #first we are calculating gradient of loss wrt net input for each node in activation layer
        #grad_J_wrt_Zj[j]: Gradient of J wrt net input for jth node
        grad_J_wrt_Zj = []
        for j, neuron in enumerate(self.layers[l].neurons):
            #Gradient of error wrt net input for each neuron in output
            #calculated by chain rule GRAD(J, Zj) = J wrt Yj x Yj wrt Zj
            # dJ/dYj is derivative of mse wrt Yj or derivative of loss wrt output
            J_wrt_Yj = self.deriv_mse_wrt_Yj(
                neuron.activation_value,
                self.targets[0][j],
                len(self.layers[-1].neurons)
            )
            # dYj/dZj is derivative of sigmoid function wrt net input
            # since this is dependent on the type of neuron being used
            # we delegate it to the neuron. This is derivative of activation
            # function wrt to the net input to the activation function
            neuron.dJ_wrt_Z = J_wrt_Yj * neuron.calculate_deriv_activation_wrt_net_input()

        #next, we calculate the gradients for loss with respect to the weights and bias
        #Gradient of error wrt Wij calculates the influence of the weight between neuron i in layer l-1
        #to the jth node in current layer l. Gradient of error wrt Bj is influence of bias on neuron j
        #in layer l
        for i, neuron in enumerate(self.layers[l].neurons):
            for j, neuron_in_previous_layer in enumerate(self.layers[l-1].neurons):
                #Grad J wrt Wij = Grad_J_wrt_Zj[j] * dZj/dWij
                #dZj/dWij = previous_neuron.activation_value since all other weights are ignored
                dZj_wrt_Wij = neuron_in_previous_layer.activation_value
                dJ_wrt_Wij = neuron.dJ_wrt_Z * dZj_wrt_Wij
                neuron.dJs_wrt_Wi.append(dJ_wrt_Wij)
            #Grad J wrt Bj = Grad_J_wrt_Zj[j] * dZj/DBj
            #dZj/dBj = 1 since the bias has no associated value
            dZj_wrt_Bj = 1
            dJ_wrt_Bj = neuron.dJ_wrt_Z * dZj_wrt_Bj
            neuron.dJ_wrt_B = dJ_wrt_Bj

        #We can then calculate the gradient of cost with respect to net input for the nodes
        #in the previous layer with the gradients we've already calculated, we calculate from
        #the output layer to the input layer.
        l = l - 1
        #while we are not on the input layer, step backwards caclculating the weight and
        #bias adjustments
        while l >= 1:
            for j, neuron in enumerate(self.layers[l].neurons):
                sum_of_dJdZi_and_Wij = 0
                for i, neuron_in_next_layer in enumerate(self.layers[l+1].neurons):
                    sum_of_dJdZi_and_Wij = sum_of_dJdZi_and_Wij + (neuron_in_next_layer.dJ_wrt_Z * neuron_in_next_layer.weights_in[j])
                neuron.dJ_wrt_Z = sum_of_dJdZi_and_Wij * neuron.calculate_deriv_activation_wrt_net_input()
                for k, neuron_in_previous_layer in enumerate(self.layers[l-1].neurons):
                    dZj_wrt_Wij = neuron_in_previous_layer.activation_value
                    dJ_wrt_Wij = neuron.dJ_wrt_Z * dZj_wrt_Wij
                    #print("k: {}, j: {}".format(k, j))
                    #print("dZj_wrt_Wij: {}, neuron.dJ_wrt_Z: {}, dJ_wrt_Wij: {}".format(dZj_wrt_Wij, neuron.dJ_wrt_Z,  dJ_wrt_Wij))
                    neuron.dJs_wrt_Wi.append(dJ_wrt_Wij)
                dZj_wrt_Bj = 1
                dJ_wrt_Bj = neuron.dJ_wrt_Z * dZj_wrt_Bj
                neuron.dJ_wrt_B = dJ_wrt_Bj
            l = l - 1

        #finally, update all weights and biases using gradient descent
        for l, layer in enumerate(self.layers):
            #layer 0 doesn't have any preceeding weights and biases
            if l == 0:
                continue
            for neuron in layer.neurons:
                for i, weight in enumerate(neuron.weights_in):
                    weight = weight - (self.learning_rate * neuron.dJs_wrt_Wi[i])
                neuron.bias_in = neuron.bias_in - (self.learning_rate * neuron.dJ_wrt_B)

    def clean_nodes(self):
        #Clean out nodes
        for l, layer in enumerate(self.layers):
            #layer 0 doesn't have any preceeding weights and biases
            for neuron in layer.neurons:
                neuron.activation_value = None
                neuron.net_input = None
                neuron.dJ_wrt_Z = None
                neuron.dJs_wrt_Wi = []
                neuron.dJ_wrt_B = None

    def train(self):
        #feed forward input vector
        self.feed_forward()

        #calculate loss using MSE
        J = self.mse()

        #Propigate the loss back through the network
        self.back_propigate()

        #clean nodes
        self.clean_nodes()

        #rotate input and target for next training call
        self.inputs = self.inputs[1:] + self.inputs[:1]
        self.targets = self.targets[1:] + self.targets[:1]
        return J


ann = NeuralNetwork({
    'input_layer': {
        'input_vectors': BINARY_X
    },
    'hidden_layers': [{
            'num_units': 4,
            'neuron_type': 'BinaryNeuron'
    }],
    'output_layer': {
        'target_vectors': BINARY_T,
        'neuron_type': 'BinaryNeuron'
    },
    'learning_rate': .1,
})

for i in range(10000):
    print('Run {}:'.format(i))
    J = ann.train()
    print(J)
    J = ann.train()
    print(J)
    J = ann.train()
    print(J)
    J = ann.train()
    print(J)
