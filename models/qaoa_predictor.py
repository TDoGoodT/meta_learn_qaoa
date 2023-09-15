import torch
from torch import tensor
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from models.lenet import LeNet, LeNetv2
class QAOAPredictor(nn.Module):
    def __init__(self, p: int) -> None:
        super().__init__()
        self.p = p


def adj_matrix_to_torch(adj_matrix: np.ndarray) -> Data:
    edge_index = tensor(list(nx.from_numpy_array(adj_matrix).edges())).t().contiguous()
    x = tensor([[adj_matrix[v].sum()] for v in range(adj_matrix.shape[0])], dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class QAOAPredictorFC(QAOAPredictor):
    def __init__(self, num_nodes, p):
        super(QAOAPredictorFC, self).__init__(p)
        
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
        
        
    def forward(self, adj_matrix: np.ndarray):
        # Convert the adjacency matrix to a vector
        new_x = tensor(adj_matrix).flatten()
        # Add padding to the input so it will fit the first layer
        new_x = F.pad(new_x, (0, self.input_size - new_x.shape[0]))
        # Pass the input through the model
        new_x = self.seq(new_x)      
        return new_x

class QAOAPredictorGNN(QAOAPredictor):
    def __init__(self, p: int, pretrained_model=None):
        super(QAOAPredictorGNN, self).__init__(p)
        self.conv1 = GCNConv(1, 128)  # Assuming node features are initialized to 1
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 2*p)

        if pretrained_model:
            self.load_state_dict(torch.load(pretrained_model))

    def forward(self, graph: np.ndarray):
        data = adj_matrix_to_torch(graph)
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)  # Global mean pooling
        angles = self.fc(x)
        
        # Apply sigmoid and scale
        angles = 2 * np.pi * torch.sigmoid(angles)
        
        return angles[0]

class QAOAPredictorLSTM(QAOAPredictor):
    def __init__(self, p: int):
        super(QAOAPredictorLSTM, self).__init__(p)
        hidden_size = 50
        output_size = 2 * p
        num_layers = 2
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: np.ndarray):
        # Convert the adjacency matrix to a list of edges
        edges = list(nx.from_numpy_array(x).edges())
        # Pass the edges through the LSTM
        out = tensor(edges).unsqueeze(0).float()
        out, _ = self.lstm(out)
        # Pass the output of the LSTM through the fully connected layers
        out = self.fc(out[:, -1, :])
        return out.squeeze(0)

class QAOAPredictorCNN(QAOAPredictor):
    def __init__(self, p: int):
        super(QAOAPredictorCNN, self).__init__(p)
        self.out_features = 2 * p
        self.p = p
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
            nn.Linear(84,self.out_features),
            nn.Softmax(dim=1)

        )
        
    def forward(self, adj_matrix: np.ndarray):
        """
        Convert the graph to a adjacency matrix
        Pass the matrix through the CNN
        """
        adj_tensor = tensor(adj_matrix).unsqueeze(0).unsqueeze(0)
        padded_adj_tensor = F.pad(adj_tensor, (0, 10 - adj_matrix.shape[0], 0, 10 - adj_matrix.shape[1]))
        a1=self.feature_extractor(padded_adj_tensor)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        return a2.squeeze(0)
        
    