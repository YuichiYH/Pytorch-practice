import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class WineDataset(Dataset):
    def __init__(self):
        # data loading

        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:]) 
        self.y = torch.from_numpy(xy[:, [0]]) # n_sample, 1

        self.n_samples = xy.shape[0]


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# dataiter = iter(dataloader)
# data = next(dataiter)
# features, labels = data

# print(features, labels)

# Dummy Training Loop

num_epoch = 2
total_samples = len(dataset)
n_iteration = math.ceil(total_samples/4)
print(total_samples, n_iteration)

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+5) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{n_iteration}, inputs {inputs.shape}')

# Torch premade datasets
torchvision.datasets.MNIST()
# fashion-mnist, cifar, coco