import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


def save_predict(output, img_name, save_path):
    TAS_palette = [192,192,192, 105,105,105,    160,82,45,  244,164,96, 60,179,113, 34,139,34,
                   154,205,50,  0,128,0,        0,100,0,    0,250,154,  139,69,19,  1,51,73,
                   190,153,153, 0,132,111,      0,0,142,    0,60,100,   135,206,250, 128,0,128,
                   153,153,153, 255,255,0,      220,20,60,  255,182,193, 220,220,220,   0,0,0]
    output_color = Image.fromarray(output.astype(np.uint8)).convert('P')
    output_color.putpalette(TAS_palette)

    output_color.save(os.path.join(save_path, img_name + '_color.png'))
