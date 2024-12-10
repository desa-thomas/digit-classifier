import importlib.util
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms

pth_path =  os.path.dirname(__file__) + "\\CNN.pth"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        ##2d Convolutional layer
        #out_channels = number of filters
        #kernel_size = size of convolution matrix (3x3)
        out_channels_1 = 32
        out_channels_2 = 64
        self.conv1 = nn.Conv2d(1, out_channels=out_channels_1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=1, padding=1)
        
        #pooling function
        #Stride = number of pixels to move the kernel (subset of pixel matrix we are pooling) over each time (stride = kernel for no overlap)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #output shape after convoluational layer and pooling: (N, out_channles, H/4, W/4) (since I used a padding of 1 in the convolutional layers to retain the shape)
        #flatten to -> (N, out_channels *(H/4)* (W/4))
        self.fc1 = nn.Linear(((28//4)**2) * out_channels_2, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x : torch.Tensor):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
  
        x = torch.flatten(x,start_dim=1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        #Get probability
        x = f.softmax(x, dim = 1)

        return x

model = CNN()
transformations = transforms.Compose([
    transforms.ToTensor(), #convert image to tensor
    transforms.Resize((28, 28)), #Resize the images (they should already be 28x28)
    transforms.Grayscale(), #convert the tensors to grayscale
    transforms.Normalize((0.1307), (0.3081)) #normalize image
])

model.load_state_dict(torch.load(pth_path))

def predict(input):
    processed_img = transformations(input).unsqueeze(0) #add batch dim
    
    model.eval()
    with torch.inference_mode():
        output = model(processed_img)
        _, preds = torch.max(output, dim= 1)

    return preds.item()