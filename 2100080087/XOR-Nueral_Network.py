"""
XOR Neural Network

This code demonstrates the implementation of a simple XOR neural network using PyTorch.
"""

# Importing required libraries
import torch as tc
import torch.nn as nn

# Define the XOR_N class
class XOR_N(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.lin1 = nn.Linear(2, 2)
        self.sig = nn.Sigmoid()
        self.lin2 = nn.Linear(2, 1)
    
    def forward(self, x):
        # Define forward pass
        x = self.lin1(x)
        x = self.sig(x)
        x = self.lin2(x)
        return x

# Instantiate the XOR_N model
model = XOR_N()

# Define training data and labels
X = tc.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

Y = tc.tensor([0.0, 1.0, 1.0, 0.0]).view(-1, 1)

# Define training parameters
epochs = 1000
mseloss = nn.MSELoss()
optimizer = tc.optim.Adam(model.parameters(), lr=0.03)

# Training loop
for epoch in range(epochs):
    for i in range(4):
        # Forward pass
        optimizer.zero_grad()
        yhat = model(X[i])
        # Compute loss
        loss = mseloss(yhat, Y[i])
        # Backpropagation
        loss.backward()
        optimizer.step()
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Sample {i+1}, Loss: {loss.item()}")

# Test the model with input [0, 1]
output = model(tc.tensor([0.0, 1.0]))
print("Output for [0, 1]: {:.10f}".format(float(output[0].detach().numpy())))
