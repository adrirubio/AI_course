import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
import os
import zipfile
import urllib.request

# Define the URL to download the MovieLens dataset
url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
filename = "ml-20m.zip"

# Unzip the downloaded file
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall("movielens_data")

data_dir = 'movielens_data/ml-20m/'

# Now read the CSV file
df = pd.read_csv(data_dir + 'ratings.csv')

df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes
