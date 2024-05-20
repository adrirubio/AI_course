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

image_path = '/home/adrian/data/train/nonfood/0_951.jpg'

plt.imshow(imageio.imread(image_path))
# remove # to see image
# plt.show()

image_path = '/home/adrian/data/train/food/1_293.jpg'

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

train_dataset = datasets.ImageFolder(
    '/home/adrian/data/train',
    transform=transform
)
test_dataset = datasets.ImageFolder(
    '/home/adrian/data/test',
    transform=transform
)

batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
)

# Define the pretrained model
vgg = models.vgg16(pretrained=True)

class VGGFeatures(nn.Module):
  def __init__(self, vgg):
    super(VGGFeatures, self).__init__()
    self.vgg = vgg

  def forward(self, X):
    out = self.vgg.features(X)
    out = self.vgg.avgpool(out)
    out = out.view(out.size(0), -1) # flatten
    return out
