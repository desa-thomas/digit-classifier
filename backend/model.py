import importlib.util
import os
import torch
from torchvision import transforms

pth_path =  os.path.dirname(__file__) + "\\..\\model\\outputs\\CNN.pth"
mod_path =  os.path.dirname(__file__) + "\\..\\model\\src\\models.py"
mod_name = "models"

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

model = models.CNN()
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