
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from models.qaoa_predictor import QAOAPredictor
from qaoa import QaoaCircuit
import networkx as nx

def start_with_model(circuit: QaoaCircuit, model: QAOAPredictor, iterations: int, shots: int) -> tuple[float, np.ndarray]:
    """
    Runs the QAOA circuit with the parameters predicted by the model.
    Returns the approximation ratio and parameters of the circuit.
    """
    # Get the predicted parameters
    params = model(circuit.graph).detach().numpy()
    # Run the circuit with the predicted parameters
    return circuit.optimize(params, iterations, shots)


def start_from_random(circuit: QaoaCircuit, iterations: int, shots: int) -> tuple[float, np.ndarray]:
    """
    Runs the QAOA circuit with random parameters.
    Returns the approximation ratio and parameters of the circuit.
    """
    # Generate random parameters
    params = np.random.rand(2 * circuit.n_layers)
    # Run the circuit with the random parameters
    return circuit.optimize(params, iterations, shots)

def start_with_model_v2(circuit: QaoaCircuit, model: QAOAPredictor, iterations: int, shots: int, factor: int) -> tuple[float, np.ndarray]:
    """
    Runs the QAOA circuit with the parameters predicted by the model.
    Returns the approximation ratio and parameters of the circuit.
    """
    # Split the graph into smaller graphs
    N = circuit.graph.number_of_nodes() // factor
    graphs = [circuit.graph.subgraph(list(range(i*factor, (i+1)*factor))) for i in range(N)]
    betas = []
    gammas = []
    for graph in graphs:
        params = model(graph).detach().numpy()
        betas.append(params[:circuit.n_layers])
        gammas.append(params[circuit.n_layers:])
    # Get the predicted parameters
    params = betas + gammas
    # Run the circuit with the predicted parameters
    return circuit.optimize(params, iterations, shots)


def compare_model_to_random(circuits: list[QaoaCircuit], model: QAOAPredictor):
    model_ratios = []
    random_ratios = []
    total_model_ratio = 0
    total_random_ratio = 0
    n_circuits = 10
    iters = list(range(1, 500, 25))
    for i in iters:
        total_model_ratio = 0
        total_random_ratio = 0
        print(f"Running iteration {i}")
        for circuit in circuits[:n_circuits]:
            model_ratio, model_params = start_with_model(circuit, model, i, 1000)
            random_ratio, random_params = start_from_random(circuit, i, 1000)
            total_model_ratio += model_ratio
            total_random_ratio += random_ratio
        model_avg_ratio = total_model_ratio / n_circuits
        random_avg_ratio = total_random_ratio / n_circuits
        model_ratios.append(model_avg_ratio)
        random_ratios.append(random_avg_ratio)
        
    plt.figure()
    plt.plot(iters, [x*100 for x in model_ratios], label="Model")
    plt.plot(iters, [x*100 for x in random_ratios], label="Random")
    plt.xlabel("Iterations")
    plt.ylabel("Approximation Ratio (%)")
    plt.title(f"Average Approximation Ratio P = {circuits[0].n_layers}")
    plt.legend()
    plt.savefig(f"figures/compare_model_to_random_G{len(circuits)}_N{circuits[0].graph.number_of_nodes()}_P{circuits[0].n_layers}.png")
    
    
def compare_model_to_random_v2(circuits: list[QaoaCircuit], model: QAOAPredictor, factor: int):
    model_ratios = []
    random_ratios = []
    total_model_ratio = 0
    total_random_ratio = 0
    n_circuits = 100
    iters = list(range(1, 500, 25))
    for i in iters:
        total_model_ratio = 0
        total_random_ratio = 0
        print(f"Running iteration {i}")
        for circuit in circuits[:n_circuits]:
            model_ratio, model_params = start_with_model_v2(circuit, model, i, 1000, factor)
            random_ratio, random_params = start_from_random(circuit, i, 1000, factor)
            total_model_ratio += model_ratio
            total_random_ratio += random_ratio
        model_avg_ratio = total_model_ratio / n_circuits
        random_avg_ratio = total_random_ratio / n_circuits
        model_ratios.append(model_avg_ratio)
        random_ratios.append(random_avg_ratio)
        
    plt.figure()
    plt.plot(iters, [x*100 for x in model_ratios], label="Model")
    plt.plot(iters, [x*100 for x in random_ratios], label="Random")
    plt.xlabel("Iterations")
    plt.ylabel("Approximation Ratio (%)")
    plt.title(f"Average Approximation Ratio P = {circuits[0].n_layers}")
    plt.legend()
    plt.savefig(f"figures/compare_model_to_random_G{len(circuits)}_N{circuits[0].graph.number_of_nodes()}_P{circuits[0].n_layers}.png")
    
def load_model(size: int, N: int, P: int) -> QAOAPredictor:
    path = f"models/qaoa_predictor_G{size}_N{N}_P{P}.pt"
    model = QAOAPredictor(N, P)
    model.load_state_dict(torch.load(path))
    return model

def load_circuits(size: int, N: int, P: int) -> list[QaoaCircuit]:
    path = f"data/dataset_G{size}_N{N}_P{P}.pkl"
    with open(path, "rb") as f:
        graphs, _, _ = pickle.load(f)
    return [QaoaCircuit(graph, P) for graph in graphs]

def check_with_larger_graphs(model: QAOAPredictor, N: int, P: int, factor: int):
    graphs = [nx.gnp_random_graph(N*factor, 0.5) for _ in range(10)]
    circuits = [QaoaCircuit(graph, P) for graph in graphs]
    compare_model_to_random_v2(circuits, model, factor)
    

for p in [3]:
    model = load_model(500, 12, p)
    circuits = load_circuits(500, 12, p)
    compare_model_to_random(circuits, model)
    # check_with_larger_graphs(model, 9, p, 2)

        