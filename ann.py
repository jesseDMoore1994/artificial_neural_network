from math import exp
from random import random

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
        self.weights_in = []
        self.bias_in = None
        self.deriv_cost_wrt_net_input = None

    def __str__(self):
        return "\n".join([
            "activation_value: {}".format(self.activation_value),
            "net_input: {}".format(self.net_input),
            "weights_in: {}".format(self.weights_in),
            "bias_in: {}".format(self.bias_in),
        ])


class BinaryNeuron(Neuron):
    def __init__(self):
        super().__init__()

    def binary_sigmoid(self, x):
        return 1/(1+exp(-x))

    def binary_sigmoid_derivative(self, x):
        return binary_sigmoid(x)*(1 - binary_sigmoid(x))

    def calculate_activation_and_store_input(self, net_input):
        self.net_input = net_input
        self.activation_value = self.binary_sigmoid(net_input)

    def calculate_deriv_activation_wrt_net_input(self):
        self.deriv_activation_wrt_net_input = self.binary_sigmoid_derivative(net_input)

class BipolarNeuron(Neuron):
    def __init__(self):
        super().__init__()

    def bipolar_sigmoid(self, x):
        return (exp(x)-exp(-x))/(exp(x)+exp(-x))

    def bipolar_sigmoid_derivative(self, x):
        return 1 - bipolar_sigmoid(x)**2

    def calculate_activation_and_store_input(self, net_input):
        self.net_input = net_input
        self.activation_value = self.bipolar_sigmoid(net_input)

    def calculate_deriv_activation_wrt_net_input(self):
        self.deriv_activation_wrt_net_input = self.bipolar_sigmoid_derivative(net_input)


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

    def _validate_and_store_config(self, config):
        self._has_valid_input_layer_params(config)
        self._has_valid_output_layer_params(config)
        self._input_and_target_lengths_match(config)
        self._has_valid_hidden_layer_params(config)
        self.inputs = config['input_layer']['input_vectors']
        self.targets = config['output_layer']['target_vectors']
        self.output_neuron_type = config['output_layer']['neuron_type']
        self.hidden_layer_data = config['hidden_layers']

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
                neuron.weights_in.append(random())

    def _create_biases(self, this_layer):
        biases = []
        for neuron in this_layer.neurons:
            neuron.bias_in = random()
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
        #Start with the output layer
        #first we are calculating gradient of loss wrt net input for each node in activation layer
        #grad_J_wrt_Zj[j]: Gradient of J wrt net input for jth node 
        grad_J_wrt_Zj = []
        for j, neuron in enumerate(self.layers[-1].neurons):
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
            neuron.calculate_deriv_activation_wrt_net_input()
            grad_J_wrt_Zj.append(J_wrt_Yj * neuron.deriv_activation_wrt_net_input)

    def train(self):

        #feed forward input vector
        self.feed_forward()

        #calculate loss using MSE
        J = self.mse()

        print(J)

        #rotate input and target for next training call
        self.inputs = self.inputs[1:] + self.inputs[:1]
        self.targets = self.targets[1:] + self.targets[:1]


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
})
print(ann)
ann.train()
print(ann)
