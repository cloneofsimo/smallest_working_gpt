import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
import math
import random
from model_simple_MLP import regression_MLP



device = torch.device("cuda:0")

ds = 4096
test_data  = [(a, b) for a, b in [(random.random(), random.random()) for _ in range(ds)]]

train_data = [(a, b, math.sin(a + b + 1/(a + 0.3) + 1/(b + 0.3))) for a, b in [(random.random(), random.random()) for _ in range(ds)]]

class two_variable_dataset(Dataset):
    def __init__(self, data):
        data = torch.tensor(data)
        self.x = data[:, 0:2]
        self.y = data[:, 2]
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = two_variable_dataset(train_data)

test_tensor = torch.tensor(test_data)

batch_size = 1024
weight_decay = 0
lr = 1e-3
epochs = 30

dl = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)

model = regression_MLP()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay= weight_decay)

model.to(device)
model.train()


for epoch in range(epochs):
    pbar = tqdm(dl)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        y = y.unsqueeze_(dim = 1)

        y_pred = model(x)

        loss = ((y_pred - y)*(y_pred - y)).mean(dim = 0)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f'epoch : {epoch + 1}, train loss : {loss.item():.5f}')


#visualize data:

model.eval()
y_pred = model(test_tensor.to(device))

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')

xs, ys, zs = [tnsr.numpy() for tnsr in torch.tensor(train_data).split(1, dim = 1)]

ax.scatter3D(xs, ys, zs, c = 'Red')

xt, yt = [tnsr.numpy() for tnsr in torch.tensor(test_tensor).split(1, dim = 1)]
zt = y_pred.cpu().detach().numpy()

ax.scatter3D(xt, yt, zt, c = 'Blue')

plt.show()