# model.py
"""
model.py
Builds a classifier using timm pretrained backbones.
"""

import timm
import torch.nn as nn
import torch

def build_model(name="deit_base_patch16_224", num_classes=55, pretrained=True, dropout=0.0):
    """
    name: a timm model name, e.g. 'deit_base_patch16_224', 'vit_base_patch16_224', 'resnet50', 'efficientnet_b0'
    """
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout, drop_path_rate=0.1)
    return model

if __name__ == "__main__":
    m = build_model(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)  # should be [2, 10]
