"""
Tools necessary to aid in the creation and teaching of the neural network
Author: Kevin Lee
"""
import numpy as np

class layer:
    """
    LAYER:
    Given certain parameters, design a layer.
    """
    # CONSTRUCTOR: generate the layer structure given the previous layer's no. of outputs(inp), no. of neurons(neu), possibly the previous weights (pre_wei), and possibly the pre-determined biases(est_bia)
    def __init__ (self, random, no_inputs, no_neurons, pre_weights, est_biases):
        self.inputs = no_inputs
        self.neurons = no_neurons
        self.output = 0
        if random:
            # WEIGHTS: Create an array of arrays (according to the number of parameters) in normal distribution:
            # np.random.randn(2, 4)
            # array([[-1.59022344, -0.05409669, -0.40128521,  0.62704859], 
            # [ 0.90419252,  0.83106465, -0.54859216,  1.50170964]])
            self.weights = np.random.randn(self.inputs, self.neurons)


            # BIASES: Unless otherwise pre-established, there should be NO biases being randomly generated:
            # np.zeros((2,4)) <-- NEED BOTH PARENTHESIS, np.zeros() REQUIRES A TUPLE
            # array([[0., 0., 0., 0.],
            # [0., 0., 0., 0.]])
            self.biases = np.zeros((1, self.neurons))
        else:
            self.weights = pre_weights
            self.biases = est_biases

    # SIGMOID: realigning the output to a number between 0(when x is negative) and 1(when x is positive), with y = 0.5 when x = 0
    def sigmoid (self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # RELU: piece-wise function for the output being either the chosen number or 0 (if the number is less than 0), less computationally draining than sigmoid and avoid vanishing gradient problem
    def relu (self, x):
        return np.maximum(x, 0)

    # LEAKY RELU: piece-wise function for the output being either the chosen number or 
    def lrelu(self, x):
        return np.maximum(x, 0.01*x)

    # FORWARD: calculates this particular layer's outputs using the previous layer's inputs
    def forward (self, inputs):
        self.output = self.lrelu(np.dot(inputs, self.weights) + self.biases)
        return self.output

class model:
    """
    MODEL:
    Creates the layers and generates the layer outputs
    """
    # CONSTRUCTOR: 
    def __init__(self, random, pre_weights, est_biases, user_input):
        
        # Initialize layers
        if random:
            self.hidden_layer = layer(True, 4800, user_input, 0, 0)
            self.output_layer = layer(True, user_input, 2, 0, 0)
        else:
            self.hidden_layer = layer(True, 4800, user_input)
            self.output_layer = layer(True, user_input, 2, 0, 0)

    def forward(self, state):
        input_image = state
        hidden_output = self.hidden_layer.forward(input_image)
        output = self.output_layer.forward(hidden_output)
        return output

#
def thought_process (fitness_score, hidden_weights, hidden_biases, output_weights, output_biases, user_input):
    format = [  ('fitness_score', np.int32), 
                ('hidden_weights', np.float64, (4800,user_input)), 
                ('hidden_biases', np.float64, (1,user_input)), 
                ('output_weights', np.float64, (user_input,2)), 
                ('output_biases', np.float64, (1,2))]
    thought_process = np.empty(1, dtype=format)
    thought_process['fitness_score'] = fitness_score
    thought_process['hidden_weights'] = hidden_weights
    thought_process['hidden_biases'] = hidden_biases
    thought_process['output_weights'] = output_weights
    thought_process['output_biases'] = output_biases
    return thought_process