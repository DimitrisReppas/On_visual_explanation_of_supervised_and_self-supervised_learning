import ipywidgets
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#All functions needed for receiving great variety of raw and rollout saliency maps.

def avg_plus_I(attn): # adding the Identity matrix to take into account the residual connections
    aver_attn = torch.mean(attn, dim=1).cpu()       
    s = aver_attn + 1.0*torch.eye(197)
    s = s / s.sum(dim=-1)
    
    return s

def get_avg_plus_I(backbone, x, layer=0): #supplementary rollout function  
    attn = get_deit_attention(backbone, x, method='raw', layer=layer)
    s = avg_plus_I(attn)

    return s

def preprocess_deit_rollout_attention(multip, dimen): # preprocess of the CLS's attention when choosing Rollout method
    cls_A = multip[0][0][1:] # keep only cls
    cls_min, cls_max = cls_A.min(), cls_A.max()
    cls_A = (cls_A - cls_min)/(cls_max - cls_min) # min max normalization
    reshaped_cls = torch.reshape(cls_A, (14, 14)) # reshape cls
    cls_A = torch.unsqueeze(reshaped_cls, 0)
    cls_A = torch.unsqueeze(cls_A, 0)
    m = nn.Upsample(size=(dimen ,dimen), mode='bilinear', align_corners=True)
    cls_new_size = m(cls_A) # upsample cls to the dimentions of original image
    cls_new_size=cls_new_size.squeeze(0)
    cls_new_size=cls_new_size.squeeze(0)
    cls_new_size = cls_new_size.cpu()
    cls_np = cls_new_size.detach().numpy()

    return cls_np

def preprocess_deit_raw_attention(attn, dim, power=1.0):  # preprocess of the CLS's attention
    if power != 1.0:
       add = 0
       for i in range(0, 12):    #original deit 12 and for tiny 2 for small 6?   
           add = add + attn[:, i, :, :] ** power # When giving --power values < 1.0 one can reveal the next more salient regions of the image. This helps a lot for supervised models. Try --power 0.05 and you will receive meaningful saliency maps 
       aver_attn = add/attn.shape[1]  # calculate the average per chanel
    else:
       aver_attn = torch.mean(attn, dim=1)
    cls_A = aver_attn[0][0][1:]  # keep only cls
    cls_min, cls_max = cls_A.min(), cls_A.max()
    cls_A = (cls_A - cls_min)/(cls_max - cls_min) # min max normalization
    reshaped_cls = torch.reshape(cls_A, (14, 14)) # reshape cls
    cls_A = torch.unsqueeze(reshaped_cls, 0)
    cls_A = torch.unsqueeze(cls_A, 0)
    m = nn.Upsample(size=(dim ,dim), mode='bilinear', align_corners=True)
    cls_new_size = m(cls_A) # upsample cls to the dimentions of original image
    cls_new_size=cls_new_size.squeeze(0)
    cls_new_size=cls_new_size.squeeze(0)
    cls_new_size = cls_new_size.cpu()
    cls_np = cls_new_size.detach().numpy() 
    
    return cls_np

def get_deit_attention(backbone, x, method, layer): # receiving raw attentions from different layers
    model = backbone
    l0 = 0
    if method == 'raw': 
       ps=model.patch_embed(x)  
       ps = torch.cat((model.cls_token.expand(ps.shape[0], -1, -1), ps), dim=1)
       ps = model.pos_drop(ps + model.pos_embed)
       if layer != 0:
          for i in range(layer):
              ps=model.blocks[l0].forward(ps)
              l0 = l0 + 1 
       ps=model.blocks[layer].norm1.forward(ps)
       
       B, N, C = ps.shape 
       num_heads = model.blocks[layer].attn.num_heads
       qkv = model.blocks[layer].attn.qkv(ps).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
       q, k, v = qkv.unbind(0)
       dim = ps.shape[2]
       head_dim = dim // num_heads
       scale = head_dim ** -0.5

       attn = (q @ k.transpose(-2, -1)) * scale
       attn = attn.softmax(dim=-1)
       attn = model.blocks[layer].attn.attn_drop(attn)
       
       return attn

def final_deit_attention(backbone, x, method, layer, power, roll_from, roll_to): # main function for receiving attention-based saliency maps 
    if method == 'raw': # receiving raw attentions from different layers. When giving --power values < 1.0 one can reveal the next more salient regions of the image. This helps a lot for supervised models. Try --power 0.05 and you will receive meaningful saliency maps 
       attn = get_deit_attention(backbone, x, method, layer) # receiving the attentions of the nth layer
       cls_np = preprocess_deit_raw_attention(attn, x.shape[3], power) # preprocess of the CLS's attention
    elif method == 'rollout': # Receiving the attentions resulting from Rollout method. When changing parameters --roll_from, --roll_to you have the chance not to take into account all the layers of the network. One example is taking the Rollout attentions when taking into account the last 3 layers (--roll_from 9, --roll_to 11)
        multip = get_avg_plus_I(backbone, x, layer=roll_from)
        if roll_to > roll_from:
            for i in range(roll_from+1, roll_to+1):
                at_av_plusI = get_avg_plus_I(backbone, x, layer=i) 
                multip = multip @ at_av_plusI  
        cls_np = preprocess_deit_rollout_attention(multip, x.shape[3])  # preprocess of the CLS's attention
    
    return cls_np

    