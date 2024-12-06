#Imports ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np
import pandas as pd
import os

#Progress bar
from tqdm.auto import tqdm

import models
from datasets import train_loader, test_loader
import argparse

#Set up global variables----------------------------------------------------------

output_path = os.path.dirname(__file__) + '\\..\\outputs'
lr = 1e-3
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}\n')

#cli variables
parser = argparse.ArgumentParser()
#no. epochs
parser.add_argument('-e', '--epochs', type = int, default = 1, help='number of epochs to train our network for')
#which model
parser.add_argument('-m', '--model', type = int, default=3, choices=[1, 2, 3, 4], help="""Choose model""")

args = vars(parser.parse_args())

epochs = args['epochs']
model_no = args['model']

#create model objcect
if(model_no == 1):
    model = models.network_1().to(device)

elif(model_no == 2):
    model = models.network_2().to(device) 

elif(model_no == 3):
    model = models.network_3().to(device) 

elif(model_no == 4):
    model = models.CNN().to(device)

print(model)

#calculate the total number of parameters (numel() calculates the number of values in a tensor)
total_parameters = sum(p.numel() for p in model.parameters())
print(f"{total_parameters} total parameters")
total_trainable_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_parameters} training parameters")

#create optimizer and loss function objects
optimizer = optim.Adam(model.parameters(), lr =lr)
criterion = nn.CrossEntropyLoss()

#-----------------------------------------------------------------------------------------------
def train(model, loader, optimizer, criterion): 
    """
    Training loop for a single epoch
    """
    model.train()
    print('Training')

    #total loss, and correct predicitons for epoch
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(loader), total=len(loader)):
        counter += 1
        images, labels = data
        
        #move image and label tensors to device
        images = images.to(device)
        labels = labels.to(device)

        #set gradients to zero
        optimizer.zero_grad()

        #forward pass
        outputs = model(images)

        #Calcualte the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        ##Calculate the accuracy
        #predictions are the index of the highest probability in each output tensor (along dim one because of the batchs)
        _, preds = torch.max(outputs, dim= 1)

        #sum the number of correct predictions and add it the the running correct value
        train_running_correct += (preds == labels).sum().item()

        #use backpropagation to calculate the gradient of the loss function (stored in the .grad attribute of each paramter)
        loss.backward()

        #Optimize parameters based on calculated loss gradients
        optimizer.step()

    # Average loss and accuracy of the whole epoch
    epoch_loss = train_running_loss / counter 
    epoch_acc = 100. *(train_running_correct) / len(loader.dataset)

    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    print("Validating model")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    #turn on inference mode
    with torch.inference_mode():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            counter += 1

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)

            #calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            #calculate the accuracy (how many correctly labled samples)
            _, preds = torch.max(outputs, dim= 1)
            #sum all correct predictions
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss/counter
    epoch_acc = 100. *valid_running_correct/len(loader.dataset)

    return epoch_loss, epoch_acc


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save model
    """

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, output_path + f'\\model_{model_no}.pth')

#-----TRAINING LOOP-------------------------------------------------

#Loss and accuracy per epoch of model
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

start = time.perf_counter()

for epoch in range(epochs):
    print(f'[INFO]: Epoch {epoch+1} of {epochs}')
    train_epoch_loss, train_epoch_acc = train(model, train_loader, optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, test_loader, criterion)

    train_loss.append(train_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    valid_loss.append(valid_epoch_loss)

    print(f'Training loss: {train_epoch_loss:.3f}, Training accuracy: {train_epoch_acc:3f}')
    print(f'Validation loss: {valid_epoch_loss}, Validation accuracy: {valid_epoch_acc:3f}')

    print('-'*50)
    time.sleep(5)

stop = time.perf_counter()
print(f"TRAIN COMPLETE\nTraining time: {stop-start}")

#Save model and loss and accuracy data
save_model(epochs, model, optimizer, criterion)

table = np.stack([train_loss, train_acc, valid_loss, valid_acc]).T
df = pd.DataFrame(data = table, columns=['Train loss', 'train accuracy', 'valid loss', 'valid accuracy'])

df.to_csv(output_path + f'\\model_{model_no}_stats.csv')