#%%
from models.qaoa_predictor import QAOAPredictor
import networkx as nx
import qiskit as qk
from qiskit import QuantumCircuit
import pickle, torch
import numpy as np
import networkx as nx
import itertools

import os
from qaoa import QaoaCircuit, max_cut_brute_force


def train_qaoa_perd(graph_size: int, n_layers: int, dataset: tuple[list[nx.Graph], list[QuantumCircuit], list[float]], epochs: int, batch_size: int, lr: float):    
    model = QAOAPredictor(graph_size, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    graphs, _, optimal_params = dataset
    for epoch in range(epochs):
        for i in range(len(graphs)):
            graph = graphs[i]
            optimal_param = optimal_params[i]
            params = model(graph)
            loss = loss_fn(params, torch.tensor(optimal_param, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")
    return model
def get_max_cut_from_state(graph: nx.Graph, state: int) -> int:
    str_state = bin(state)[2:].zfill(graph.number_of_nodes())
    return sum([1 if str_state[i] != str_state[j] else 0 for i, j in graph.edges])
    


def test_qaoa_perd_inst(model: QAOAPredictor, graph: nx.Graph, optimal_params: np.ndarray, qaoa_circuit: QaoaCircuit) -> float:
    params = model(graph)
    max_cut_val, _ = max_cut_brute_force(graph)
    circuit = qaoa_circuit.assign_parameters({qaoa_circuit.beta: optimal_params[:qaoa_circuit.n_layers], qaoa_circuit.gamma: optimal_params[qaoa_circuit.n_layers:]})
    job = qk.execute(circuit, qk.Aer.get_backend("qasm_simulator"), shots=1000)
    result = job.result()
    counts: dict[str, int] = result.get_counts()
    max_count = max(counts, key=lambda x: counts[x])
    qaoa_max_cut_val = -qaoa_circuit.energy_cost(int(max_count, 2))
    approx_ratio = qaoa_max_cut_val / max_cut_val
    return abs(params.detach().numpy() - optimal_params).sum(), approx_ratio
    
    

def test_qaoa_perd(model: QAOAPredictor, dataset: tuple[list[nx.Graph], list[QuantumCircuit], list[float]]) -> float:
    graphs, circuits, optimal_params = dataset
    total_error = 0
    total_approx_ratio = 0
    for i in range(len(graphs)):
        graph = graphs[i]
        optimal_param = optimal_params[i]
        circuit = circuits[i]
        error, approx_ratio = test_qaoa_perd_inst(model, graph, optimal_param, circuit)
        total_error += error
        total_approx_ratio += approx_ratio
    avg_error = total_error / len(graphs)
    avg_approx_ratio = total_approx_ratio / len(graphs)
    print(f"Average error: {avg_error}")
    print(f"Average approximation ratio: {avg_approx_ratio}")
    return avg_error, avg_approx_ratio


def load_dataset(path: str) -> tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_dataset_from_params(dataset_size: int, graph_size: int, n_layers: int) -> tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray]:
    print(f"Loading dataset from data/dataset_G{dataset_size}_N{graph_size}_P{n_layers}.pkl")
    return load_dataset(f"data/dataset_G{dataset_size}_N{graph_size}_P{n_layers}.pkl")
    
def get_dataset_info(file: str) -> tuple[int, int, int]:
    try:
        dataset_size, graph_size, n_layers = file.split("_")[1:]
        dataset_size = dataset_size[1:]
        graph_size = graph_size[1:]
        n_layers = n_layers[1:-4]
        print(f"Dataset {file} has {dataset_size} graphs of size {graph_size} with {n_layers} layers")
        return int(dataset_size), int(graph_size), int(n_layers)
    except:
        print(f"Dataset {file} is not formatted correctly")

def list_datasets():
    res = []
    for file in os.listdir("data"):
        if file.endswith(".pkl"):
            res.append(get_dataset_info(file))
    return res

def split_dataset(dataset):
    graphs, qaoa_circuits, optimal_params = dataset
    split = int(len(graphs) * 0.8)
    train_dataset = (graphs[:split], qaoa_circuits[:split], optimal_params[:split])
    test_dataset = (graphs[split:], qaoa_circuits[split:], optimal_params[split:])
    return train_dataset, test_dataset
def main() -> int:
    for (S, N, P) in list_datasets():
        print(f"Training on dataset {S} with graph size {N} and {P} layers")
        data_set = load_dataset_from_params(S, N, P)
        train_dataset, test_dataset = split_dataset(data_set)
        model = train_qaoa_perd(graph_size=N, n_layers=P, dataset=train_dataset, epochs=25, batch_size=10, lr=0.01)
        test_qaoa_perd(model, test_dataset)
        torch.save(model.state_dict(), f"models/qaoa_predictor_G{S}_N{N}_P{P}.pt")
if __name__ == "__main__":
    exit(main())


# %%
