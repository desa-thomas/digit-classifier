import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

"""
Script to load MNIST dataset
"""

#Data transformations 
transformations = transforms.Compose([
    transforms.Resize((28, 28)), #Resize the images (they should already be 28x28)
    transforms.ToTensor(), #convert image to tensor
    transforms.Grayscale(), #convert the tensors to grayscale
    transforms.Normalize((0.1307), (0.3081)) #normalize image
])

#batch size
batch_size = 16

#Path to dataset folder
path = os.path.dirname(__file__) + '\\..\\MNIST_dataset'

#Create dataset objects using `torchvision.datasets.ImageFolder` class
train_dataset = datasets.ImageFolder(path + '/train', transform=transformations)
test_dataset  = datasets.ImageFolder(path + '/test', transform=transformations)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)




