import torch
import torch.nn as nn
#Utility functions
import torch.nn.functional as f

class network_1(nn.Module):
    """
    Simple neural network with 2 hidden fully connected / linear layers. 
    """
    def __init__(self):
        super().__init__()
        ###Initialize Layers

        #layer 1 (784 input nodes, 14 output nodes)
        self.layer_1 = nn.Linear(784, 14)

        #layer 2 (14 in 14 out)
        self.layer_2 = nn.Linear(14, 14)

        #out layer
        self.out_layer = nn.Linear(14, 10)

    def forward(self, x : torch.Tensor):
        """
        forward pass input tensor x through the network

        param: 
            x : (784, 1) shaped tensor representing the handwritten digit

        return:
            out : (10, 1) shaped tensor representing the probability distirbution of what digit the input was

        """
        #Flatten input tensor for the linear layers, start at index 1 as to not flatten the batch layer
        x = torch.flatten(x, start_dim=1)

        #ReLu(Wx + b) - calculate output of first layer
        x = f.relu(self.layer_1(x))

        #Calculate second layer activation values
        x = f.relu(self.layer_2(x))

        #Calculate the output layer activation values
        x = f.relu(self.out_layer(x))

        #convert output activation values to a probability dist
        x = f.softmax(x, dim = 1)

        return x

class network_2(nn.Module):
    """
    Another Fully connected NN, but this one has 1 hidden layer the size of 
    the mean of the input and output sizes(input + output)/2
    """
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 397)
        self.outlayer = nn.Linear(397, 10)

    def forward(self, x : torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.layer1(x))
        x = f.relu(self.outlayer(x))
        x = f.softmax(x, dim=1)

        return x
   