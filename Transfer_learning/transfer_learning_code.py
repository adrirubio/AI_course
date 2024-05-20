import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os
from glob import glob
import imageio

image_path = '/home/adrian/training/0_951.jpg'

plt.imshow(imageio.imread(image_path))
# remove # to see image
# plt.show()

image_path = '/home/adrian/training/1_293.jpg'

plt.imshow(imageio.imread(image_path))
# remove # to see image
# plt.show()

# Note: normalize mean and std are standardized for ImageNet
transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
