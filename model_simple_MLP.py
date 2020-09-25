import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


class regression_MLP(nn.Module):
    def __init__(self, activation = F.relu):
        super().__init__()
        
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)
        
        
        #equivalent to ->
        '''
        self.mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        '''
        
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        
        return x
        '''
        # or
        x = self.mlp(x)
        #x = torch.sin(x)
        return x
        '''

