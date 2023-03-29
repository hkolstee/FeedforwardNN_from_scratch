# Feedforward neural network (implementation: feedforward.py)
### In this project I challenged myself to create a feedforward neural network from scratch.

### Features:
Example initialization call: <br>
```python
from feedforward import FeedForwardNN
nn = FeedForwardNN(input_size = 2, output_size = 1, nr_layers = 3, hidden_units = [10, 10, 10], activation = "relu", output_activation = "sigmoid")
```

We can:
- Define input and output sizes, for different types of classification / regression problems
- Define the number of layers
- Define the number of hidden units (neurons) for each layer seperately
- Set the activation function used in the hidden layers
- Set the activation function used in the hidden to output layer

To train the network (assuming you have your data): <br>
```python
from feedforward import BCELoss

for epoch in range(nr_epochs):
    for i in range(nr_samples):
        output = nn.forward(x[i])
        current_loss = BCELoss(output, y[i])
        nn.backprop(x[i], output, y[i], "BCELoss")
```
Training is done using stochastic gradient descent.

### Activation functions
- tanh
- sigmoid
- relu
- linear
- softmax

### Loss functions
- BCELoss
- TODO: MSELoss + more

# Problems:
The model has no measure against vanishing/exploding gradients, so when using sigmoid it is often observed that the model stops learning due to the gradient vanishing. 
Additionally, the gradients can explode when using relu. Dropout and regularization (or even gradient clipping) should be implemented stil (TODO).