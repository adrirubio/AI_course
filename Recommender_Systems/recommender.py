import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
import requests
import io

url = 'https://lazyprogrammer.me/course_files/spam.csv'
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.content.decode('latin-1')), encoding='latin-1')

print(df)
