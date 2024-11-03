import numpy as np

class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, rate=0.01):
        self.input_to_hidden_weights = np.random.rand(input_nodes, hidden_nodes) - 0.5
        self.hidden_to_output_weights = np.random.rand(hidden_nodes, output_nodes) - 0.5
        self.hidden_biases = np.random.rand(hidden_nodes) - 0.5
        self.output_biases = np.random.rand(output_nodes) - 0.5
        self.rate = rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return z * (1 - z)

    def forward_pass(self, inputs):
        self.inputs_to_hidden = np.dot(inputs, self.input_to_hidden_weights) + self.hidden_biases
        self.hidden_activation = self.sigmoid(self.inputs_to_hidden) 
        self.hidden_to_output = np.dot(self.hidden_activation, self.hidden_to_output_weights) + self.output_biases
        self.output_activation = self.sigmoid(self.hidden_to_output)
        return self.output_activation

    def backward_pass(self, inputs, targets, outputs):
        output_errors = targets - outputs
        output_gradient = output_errors * self.sigmoid_prime(outputs)
        
        hidden_errors = output_gradient.dot(self.hidden_to_output_weights.T)
        hidden_gradient = hidden_errors * self.sigmoid_prime(self.hidden_activation)

        
        self.hidden_to_output_weights += self.hidden_activation.T.dot(output_gradient) * self.rate
        self.input_to_hidden_weights += inputs.T.dot(hidden_gradient) * self.rate
        self.output_biases += np.sum(output_gradient, axis=0) * self.rate
        self.hidden_biases += np.sum(hidden_gradient, axis=0) * self.rate

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            outputs = self.forward_pass(inputs)
            self.backward_pass(inputs, targets, outputs)