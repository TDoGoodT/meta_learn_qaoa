
from train_funcs import graph_tester
from models.qaoa_predictor import *
from dataset.qaoa_dataset import *
import networkx as nx
import numpy as np
from conftest import gen_random_graph


def test_tester():
    model = QAOAPredictorFC(num_nodes=10, p=10)
    graph = tensor(nx.adjacency_matrix(gen_random_graph(10, 1)).todense().astype(np.float32))
    optimal_param = tensor(np.random.rand(20))
    _, approx = graph_tester(model, graph, optimal_param, model.p)