from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np

class QAOADataset(Dataset):
    def __init__(self, graphs: list[nx.Graph], optimal_params: np.ndarray):
        assert len(graphs) == len(optimal_params)
        self.graphs = [nx.adjacency_matrix(graph, dtype=np.float32).todense() for graph in graphs]
        self.optimal_params = optimal_params

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.optimal_params[idx]
    
    
def create_dataloader(qaoa_dataset: QAOADataset, batch_size, shuffle=True):
    return DataLoader(qaoa_dataset, batch_size=batch_size, shuffle=shuffle)