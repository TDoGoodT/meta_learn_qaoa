"""
This model is used to predict the parameters of the QAOA circuit.
It was proposed in https://arxiv.org/abs/2208.09888.
"""
import torch
from torch import Tensor, tensor
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
class LeNet(nn.Module):
    
    def __init__(self, out_features):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256,120),  
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,out_features),
            nn.Softmax()

        )
        
    def forward(self,x): 
        a1=self.feature_extractor(x)
        print(a1.shape)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        return a2

class LeNetv2(nn.Module):
    
    def __init__(self, out_features):
        super(LeNetv2, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,out_features),
        )
        
    def forward(self,x: Tensor) -> Tensor:
        a1=self.feature_extractor(x)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        a3 = F.softmax(a2,dim=0)
        return a3
