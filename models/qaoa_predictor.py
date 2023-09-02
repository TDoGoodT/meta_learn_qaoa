"""
This model is used to predict the parameters of the QAOA circuit.
It was proposed in https://arxiv.org/abs/2208.09888.
"""
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
class QAOAPredictor(nn.Module):
    def __init__(self, num_nodes, p):
        super(QAOAPredictor, self).__init__()
        
        # Calculate the input size based on the formula N(N^2-1)
        input_size = num_nodes * (num_nodes**2 - 1)
        hidden_size = 2 * input_size * p
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Define the layers
        self.seq = nn.Sequential()
        self.seq.add_module("fc1", nn.Linear(input_size, hidden_size))
        self.seq.add_module(f"hidden", nn.Linear(hidden_size, hidden_size))
        self.seq.add_module(f"relu", nn.ReLU())
        self.seq.add_module("output", nn.Linear(hidden_size, 2 * p))
        
        
    def forward(self, x: nx.Graph):
        # Convert the graph to a vector
        new_x = tensor(nx.to_numpy_array(x, dtype=np.float32)).flatten()
        # Add padding to the input so it will fit the first layer
        new_x = F.pad(new_x, (0, self.input_size - new_x.shape[0]))
        # Pass the input through the model
        new_x = self.seq(new_x)      
        return new_x
