"""
Perceptron Implementation


This code demonstrates the implementation of a perceptron and computes the sigmoid activation function for the weighted sum.
The user can specify the number of perceptrons to create, and the code generates random weights and bias for each perceptron. It then computes the sigmoid value for the weighted sum of each perceptron.
The sigmoid activation function is defined as sigmoid(x) = 1 / (1 + exp(-x)).

"""

import numpy as np

# Define the list to store sigmoid values
l = np.array([])

# Define the sigmoid activation function
def sigmoid(value):
    return 1 / (1 + np.exp(-value))

# Define the function to create a perceptron
def perceptron():
    # Input array
    arr = np.array([list(map(float, input().split()))])
    size = np.shape(arr)
    
    # Initialize random weights and bias
    weight = np.random.rand(1, size[1])
    bias = np.random.rand()
    
    # Compute the weighted sum
    value = np.dot(arr, weight[0]) + bias
    
    # Calculate sigmoid value and store it in the list
    global l
    l = np.array([sigmoid(value)])

if __name__ == '__main__':
    # Get the number of perceptrons from the user
    n = int(input("Enter number of perceptrons: "))
    
    # Create the specified number of perceptrons
    for i in range(n):
        perceptron()
    
    # Print the list containing sigmoid values
    print(l)
