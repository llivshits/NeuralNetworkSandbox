from typing import List
import numpy as np
import math

class InputNeuron():
    def __init__(self, inputs):
        self.inputs = inputs
        self.outputs = [] 
        
    def forward_prop(self):
        for item in self.outputs:
            item.val = self.inputs
    
    def __repr__(self):
        to_return = "edges: "
        for item in self.inputs:
            to_return += str(item.val) + " " + str(item.weight)
        return to_return
        
class OutputNeuron():
    def __init__(self, bias):
        self.inputs = []     
        self.outputs = []
        self.bias = bias
        
    def forward_prop(self):
        total =  0
        for item in self.inputs:
            total += math.exp(item.val * item.weight + self.bias)
        self.outputs = self.softmax(total);               
        return self.outputs
    
    
    def softmax(self, total):
        new_outputs = []
    
        for item in self.inputs:
            new_outputs.append(math.exp(item.val * item.weight + self.bias) / total)

        return new_outputs
    
    def __repr__(self):
        to_return = "edges: "
        for item in self.inputs:
            to_return += str(item.val) + " " + str(item.weight)
        to_return += ". output: "
        for item in self.outputs: 
            to_return += str(item.val)
        return to_return
    
class HiddenNeuron():
    def __init__(self, bias):
        self.inputs = []     
        self.outputs = []
        self.bias = bias
        
    def forward_prop(self):
        total = 0.0
        for item in self.inputs:
            total += (item.val * item.weight) 
        for item in self.outputs: 
           item.val = self.relu(total + self.bias)
           
    def relu(self, value):
        return max(0, value)
    
    def __repr__(self):
        to_return = "input edges: "
        for item in self.inputs:
            to_return += str(item.val) + " " + str(item.weight)
        to_return += ". output edges: "
        for item in self.outputs: 
            to_return += str(item.val) + " " + str(item.weight)
        return to_return
    
class Edge:
    def __init__(self, val, weight = 0.0):
        self.weight = weight
        self.val = val
        
class NeuralNetwork:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        
    def forward(self, inputs):
        for i, val in enumerate(inputs):
            self.input_neurons[i].inputs = val
        for h in self.hidden_neurons:
            h.forward_prop()
        for o in self.output_neurons:
            o.forward_prop()
        return [o.outputs for o in self.output_neurons]
    
    def mean_squared_error_loss(self, outputs, targets):
        return .5 * ((outputs - targets) ** 2)
    
    def cross_entropy_loss(self, outputs, targets, eps = 1e-12):
        outputs = np.clip(outputs, eps, 1- eps) # makes sure that the values are not 0, because log(0) doesnt exist
        return -np.sum(targets * np.log(outputs), axis=1) # returns the negative sum of log of highest probability per output neuron, returns as loss
    