import pandas as pd
import numpy as np
import torch

print("Loading CSV...")
data = pd.read_csv('data/mnist_train.csv')
data = np.array(data)
data = np.random.shuffle(data)

print("Converting to Tensor...")
data = torch.tensor(data)

m, n = data.size()

