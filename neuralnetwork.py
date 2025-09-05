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
            to_return += item.val + " " + item.weight
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
        return self.softmax();               
        
    def softmax(self, total):
        new_outputs = []
    
        for item in self.inputs:
            new_outputs.append(math.exp((item.val * item.weight + self.bias) / total))

        return new_outputs
    
    def __repr__(self):
        to_return = "edges: "
        for item in self.inputs:
            to_return += item.val + " " + item.weight
        to_return += ". output: "
        for item in self.outputs: 
            to_return += item.val
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
            to_return += item.val + " " + item.weight
        to_return += ". output edges: "
        for item in self.outputs: 
            to_return += item.val + " " + item.weight
        return to_return
    
class Edge:
    def __init__(self, val, weight = 0.0):
        self.weight = weight
        self.val = val