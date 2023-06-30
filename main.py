import pandas as pd
import numpy as np
import torch
def load_csv(path: str) -> torch.Tensor:
    print("Loading CSV...")
    data = pd.read_csv(path)
    data = np.array(data)
    data = np.random.shuffle(data)

    print("Converting to Tensor...")
    data = torch.tensor(data)

    return data


data = load_csv('data/mnist_train.csv')
m, n = data.size()

