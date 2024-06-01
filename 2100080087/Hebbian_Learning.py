"""
Simple Hebbian Learning with PyTorch

This code demonstrates the implementation of a simple Hebbian learning rule using PyTorch.
The Hebbian learning rule is applied to learn associations between input patterns and target labels.
"""

# Importing required libraries
import torch as tc
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define input patterns S and C
S = [
    list("++++"),
    list("+   "),
    list("++++"),
    list("   +"),
    list("++++")
]

C = [
    list("++++"),
    list("+   "),
    list("+   "),
    list("+   "),
    list("++++")
]

# Convert input patterns to PyTorch tensors
S = np.array(S)
S = tc.tensor(np.where(S == "+", 1, -1), dtype=tc.float32)

C = np.array(C)
C = tc.tensor(np.where(C == "+", 1, -1), dtype=tc.float32)

# Define target labels
target1 = tc.tensor(1.0)
target2 = tc.tensor(0.0)

# Define Hebbian Network class
class Hebb_Net(nn.Module):
    def __init__(self, arr_size=(3, 3)):
        super().__init__()
        self.len = np.prod(arr_size)
        self.weights = tc.zeros(self.len)
        self.bias = tc.zeros(1)

    def forward(self, data, target):
        # Update weights and bias using Hebbian learning rule
        self.weights.data += data * target
        self.bias.data += target

        # Return weighted input data
        return (data * self.weights)

# Create an instance of the Hebb_Net class
model = Hebb_Net(arr_size=S.shape)

# Train the model with input patterns S and C and their corresponding target labels
model(S.flatten(), target1)
model(C.flatten(), target2)

# Print the learned weights and bias
print("Learned Weights:", model.weights)
print("Learned Bias:", model.bias)
