# All the models that are used for qualitative and quantitative evaluation in this Project.

import torch
import torch.nn as nn
from torchvision import models

# Supervised and self-supervised models based on ResNet
def res_50():
    model =  models.resnet50(pretrained=True)
    return model.eval()

def dino_res_50():
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50', pretrained = True)  #pretrained encoder
    state_dict = torch.load("../storage/pretrained_weights/dino_resnet50_linearweights.pth")['state_dict'] # load the weights of the pretrained classifier and rename the keys # ---> GIVE YOUR PATH
    state_dict = {k.replace("module.linear.", ""): v for k, v in state_dict.items()}
    mod=nn.Sequential(nn.Flatten(),nn.Linear(2048, 1000)) # building the head of the model
    mod[1].load_state_dict(state_dict) # load the weights dino_resnet50_linearweights to the builded classifier
    model.fc = mod # put the classifier on top of the encoder
    return model.eval()

def moco_v3_res_50():
    model = models.resnet50()  # Moco-v3 architecture based on Resnet in evaluation mode is equal with Resnet architecture
    state_dict = torch.load("../storage/pretrained_weights/linear-1000ep.pth.tar")['state_dict'] # load the weights of the pretrained model and rename the keys # ---> GIVE YOUR PATH
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.eval()
# Supervised and self-supervised models based on Deit
def deit_base():
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_base_patch16_224', pretrained=True)
    return model.eval()

def moco_v3_deit_base():
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_base_patch16_224') # Moco-v3 architecture based on Deit in evaluation mode is equal with Deit architecture
    state_dict = torch.load("../storage/pretrained_weights/linear-vit-b-300ep.pth.tar")['state_dict'] # load the weights of the pretrained model and rename the keys  #-----> GIVE YOUR PATH
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.eval()

# A dictionary with all the pretrained models that are used on this project. Of course more models can be added here.
models_dict = {
  "res_50": res_50(),
  "dino_res_50": dino_res_50(),
  "moco_v3_res_50": moco_v3_res_50(),
  "deit_base": deit_base(),
  "moco_v3_deit_base": moco_v3_deit_base()
}