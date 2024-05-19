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
