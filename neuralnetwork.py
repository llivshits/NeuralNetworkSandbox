from typing import List
import numpy as np
import math


class InputNeuron:
    def __init__(self, value=0):
        self.value = value
        self.outputs = []

    def forward_prop(self):
        for edge in self.outputs:
            edge.val = self.inputs

    def __repr__(self):
        to_return = "edges: "
        for item in self.inputs:
            to_return += str(item.val) + " " + str(item.weight)
        return to_return


class OutputNeuron:
    def __init__(self, bias):
        self.inputs = []
        self.bias = bias
        self.z = 0
        self.output = 0
        self.delta = 0

    def compute_z(self, total):  # for use in network level softmax
        return sum(item.val * item.weight for item in self.inputs) + self.bias

    def compute_z(self, target):
        self.delta = self.output - target

    def update_weights(self, learning_rate):
        for edge in self.inputs:
            gradient = self.delta * edge.val
            edge.weight -= learning_rate * gradient
        self.bias -= learning_rate * self.delta

    def __repr__(self):
        to_return = "edges: "
        for item in self.inputs:
            to_return += str(item.val) + " " + str(item.weight)
        to_return += ". output: "
        for item in self.outputs:
            to_return += str(item.val)
        return to_return


class HiddenNeuron:
    def __init__(self, bias):
        self.inputs = []
        self.outputs = []
        self.bias = bias
        self.z = 0
        self.delta = 0

    def forward_prop(self):
        total = 0.0
        for item in self.inputs:
            total += item.val * item.weight
        for item in self.outputs:
            item.val = self.relu(total + self.bias)

    def relu(self, value):
        return max(0, value)

    def compute_delta(self, error_from_next_layer):
        relu_derivative = 1.0 if self.z > 0 else 0.0
        self.delta = error_from_next_layer * relu_derivative

    def update_weights(self, learning_rate):
        for edge in self.inputs:
            gradient = self.delta * edge.val
            edge.weight -= learning_rate * gradient
        self.bias -= learning_rate * self.delta

    def __repr__(self):
        to_return = "input edges: "
        for item in self.inputs:
            to_return += str(item.val) + " " + str(item.weight)
        to_return += ". output edges: "
        for item in self.outputs:
            to_return += str(item.val) + " " + str(item.weight)
        return to_return


class Edge:
    def __init__(self, val, weight=None, fan_in=1):
        self.val = val
        if weight is None:  # He initlization, optimizing for ReLU
            limit = np.sqrt(6.0 / (fan_in))
            self.weight = np.random.uniform(-limit, limit)
        else:
            self.weight = weight


class NeuralNetwork:  # For a 3 layer network with one hidden layer
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

    def forward(self, inputs):
        for i, val in enumerate(inputs):  # Sets input values
            self.input_neurons[i].inputs = val
            self.input_neurons[i].forward_prop()

        for h in self.hidden_neurons:  # Passes through hidden layer
            h.forward_prop()

        # Compute z values for output layer
        z_values = []
        for o in self.output_neurons:
            z_values.append(o.compute_z())

        # Applies Softmax for all outputs, stores in neurons
        outputs = self.softmax(z_values)
        for i, o in enumerate(self.output_neurons):
            o.output = outputs[i]

        return outputs

    def backward(self, targets, learning_rate):
        for i, neuron in enumerate(self.output_neurons):
            neuron.compute_delta(targets[i])
            neuron.update_weights(learning_rate)

        for i, neuron in enumerate(self.hidden_neurons):
            error = sum(
                o_neuron.delta * o_neuron.inputs[i].weight
                for o__neuron in self.output_neurons
            )
            neuron.compute_delta(error)
            neuron.update_weights(learning_rate)

    def softmax(self, z_values):
        z_max = max(z_values)
        exp_values = [math.exp(z - z_max) for z in z_values]
        sum_exp = sum(exp_values)
        return [exp_val / sum_exp for exp_val in exp_values]

    def mean_squared_error_loss(self, outputs, targets):
        return 0.5 * ((outputs - targets) ** 2)

    def cross_entropy_loss(self, outputs, targets, eps=1e-12):
        outputs = np.clip(
            outputs, eps, 1 - eps
        )  # makes sure that the values are not 0, because log(0) doesnt exist
        return -np.sum(
            targets * np.log(outputs), axis=1
        )  # returns the negative sum of log of highest probability per output neuron, returns as loss

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for x, target in zip(x_train, y_train):
                outputs = self.forward(x)
                loss = self.cross_entropy_loss(np.array(outputs), np.array(target))
                total_loss += loss
                self.backward(target, learning_rate)

            if epoch % 10 == 0:
                avg_loss = total_loss / len(x_train)
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        print("Training complete!")
