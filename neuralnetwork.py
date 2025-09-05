from typing import List
import numpy as np

class InputNeuron():
    def __init__(self, input):
        self.input = input
        self.output = [] 
        
    def forward_prop(self):
        for item in output:
            item.inVal = (input * item.weight) + item.bias
        
class OutputNeuron():
    def __init__(self, output):
        self.input = []     
        self.output = output
        
    def forward_prop(self):
        sum = 0.0
        for item in input:
            sum += (item.outVal * item.weight)
        output = sum;        
        
    #def softmax(self):
            
    
class HiddenNeuron()  :
    def __init__(self):
        self.input = []     
        self.output = []
        
    def forward_prop(self):
        sum = 0.0
        for item in input:
            sum += (item.outVal * item.weight)
        for item in output: 
           value = (input * item.weight) + item.bias
           item.inVal = relu(value)
           
    def relu(self, value):
        return max(0, max)
    
class Edge:
    def __init__(self, inVal, outVal, bias = 0.0, weight = 0.0):
        self.bias = bias
        self.weight = weight
        self.inVal = inVal
        self.outVal = outVal