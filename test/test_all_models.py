import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qaoa_predictor import *
import networkx as nx
from conftest import gen_random_graph
def test_fc():
    graph = nx.adjacency_matrix(gen_random_graph(10, 1)).todense().astype(np.float32)
    model = QAOAPredictorFC(num_nodes=10, p=10)
    # sanity check
    angles = model(graph)
    assert angles.shape == (20,)
    
def test_gnn():
    graph = nx.adjacency_matrix(gen_random_graph(10, 1)).todense().astype(np.float32)
    model = QAOAPredictorGNN(p=10)
    # sanity check
    angles = model(graph)
    assert angles.shape == (20,)
    
def test_lstm():
    graph = nx.adjacency_matrix(gen_random_graph(10, 1)).todense().astype(np.float32)
    model = QAOAPredictorLSTM(p=10)
    # sanity check
    angles = model(graph)
    assert angles.shape == (20,)
    
def test_cnn():
    graph = nx.adjacency_matrix(gen_random_graph(10, 1)).todense().astype(np.float32)
    model = QAOAPredictorCNN(p=10)
    # sanity check
    angles = model(graph)
    assert angles.shape == (20,)


