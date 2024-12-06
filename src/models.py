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

class network_3(nn.Module):
    """
    Fully connected NN with 3 hidden layers with nodes in decreasing powers of 2,
    from 128 to 64 to 32
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 128)
        self.outlayer = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        x = f.relu(self.layer3(x))
        x = f.relu(self.outlayer(x))
        x = f.softmax(x, dim=1)

        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##2d Convolutional layer
        #out_channels = number of filters
        #kernel_size = size of convolution matrix (3x3)
        self.conv1 = nn.Conv2d(1, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        #pooling function
        #Stride = number of pixels to move the kernel (subset of pixel matrix we are pooling) over each time (stride = kernel for no overlap)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #output shape after convoluational layer and pooling: (N, out_channles, H/2, W/2)
        self.fc1 = nn.Linear((28/2)**2, 10)
    
    def forward(self, x : torch.Tensor):
        x = self.conv1(x)
        x = self.pool(x)

        return x