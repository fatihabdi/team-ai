
import torch.nn as nn
import torch
import torch.nn.functional as F

def model_initialization(model_path):
    # Initialize model

    model = TheModelClass()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model