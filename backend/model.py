import importlib.util
import os
import torch

pth_path = os.getcwd() + "\\model\\outputs\\CNN.pth"
mod_path = os.getcwd() + "\\model\\src\\models.py"
mod_name = "models"

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

model = models.CNN()

model.load_state_dict(torch.load(pth_path))

model.eval()