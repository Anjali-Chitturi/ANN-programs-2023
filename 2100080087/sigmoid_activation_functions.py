"""
Sigmoidal Activation Functions

This code demonstrates the implementation of binary and bipolar sigmoidal activation functions
and plots their output values over a range of input values.
"""

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the binary sigmoidal activation function
def binary_sigmoidal(x):
    return 1 / (1 + np.exp(-x))

# Define the bipolar sigmoidal activation function
def bipolar_sigmoidal(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

# Generate input values
x = np.arange(-10, 10, 0.1)

# Compute output values using binary sigmoidal function
binary_sigmoidal_values = binary_sigmoidal(x)

# Compute output values using bipolar sigmoidal function
bipolar_sigmoidal_values = bipolar_sigmoidal(x)

# Plot the output values for both activation functions
plt.plot(x, binary_sigmoidal_values, label='Binary sigmoidal')
plt.plot(x, bipolar_sigmoidal_values, label='Bipolar sigmoidal')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.title('Binary and Bipolar Sigmoidal Activation Functions')
plt.grid(True)
plt.show()
