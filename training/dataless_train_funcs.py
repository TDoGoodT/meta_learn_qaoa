from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Union
from rich import box
from rich.console import JustifyMethod
from rich.padding import PaddingDimensions
from rich.style import StyleType
from rich.text import TextType
import networkx as nx
import qiskit as qk
from qiskit import QuantumCircuit
from dataset.qaoa_dataset import QAOADataset
from models.qaoa_predictor import QAOAPredictor
import pickle, torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import os
from qaoa import QaoaCircuit, max_cut_brute_force
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import rich
from rich.table import Column, Table
from rich.progress import track


@dataclass
class TrainEpochSummary:
    model: QAOAPredictor
    loss: float
    approx: float


@dataclass
class InnerEpochSummary:
    model: QAOAPredictor
    loss: float
    approx: float
    highest_approx: float
    lowest_approx: float


@dataclass
class EpochSummary:
    model: QAOAPredictor
    train_loss: float
    test_loss: float
    train_approx: float
    test_approx: float

    def log(self):
        table = EpochTable(epoch_summary=self)
        rich.print(table)


class EpochTable(Table):
    def __init__(self, *args, **kwargs):
        assert "epoch_summary" in kwargs
        epoch_summary = kwargs.pop("epoch_summary")
        assert isinstance(epoch_summary, EpochSummary)
        super(EpochTable, self).__init__(*args, **kwargs)
        self.add_column("Train loss")
        self.add_column("Test loss")
        self.add_column("Train approx")
        self.add_column("Test approx")
        self.add_row(
            str(epoch_summary.train_loss),
            str(epoch_summary.test_loss),
            str(epoch_summary.train_approx),
            str(epoch_summary.test_approx),
        )


@dataclass
class TrainingSummary:
    best_epoch: EpochSummary | None = None
    all_epochs: list[EpochSummary] = field(default_factory=list)

    def update(self, epoch: EpochSummary):
        self.all_epochs.append(epoch)
        if self.best_epoch is None or epoch.test_loss < self.best_epoch.test_loss:
            self.best_epoch = epoch


def custom_loss(approx_ratio):
    # The target is a tensor of ones, with the same shape as approx_ratio
    target = torch.ones_like(approx_ratio)
    # Use mean squared error loss
    loss = F.mse_loss(approx_ratio, target)
    return loss



def create_epoch_summary(
    train_epoch_result: InnerEpochSummary, test_epoch_result: InnerEpochSummary
) -> EpochSummary:
    return EpochSummary(
        model=train_epoch_result.model,
        train_loss=train_epoch_result.loss,
        test_loss=test_epoch_result.loss,
        train_approx=train_epoch_result.approx,
        test_approx=test_epoch_result.approx,
    )


def inner_dataless_epoch(
    model: QAOAPredictor,
    train: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
) -> InnerEpochSummary:
    
    if train:
        model.train()  # Set the model to training mode
    else:
        model.eval()  # Set the model to evaluation mode
    total_loss = torch.tensor(
        [0.0], requires_grad=True
    )  # Initialize batch error if needed
    total_approx = 0
    highest_approx = 0
    lowest_approx = 1
    for batch in track(data_loader, description="Running epoch"):
        graph_batch, optimal_param_batch = batch
        batch_total_approx = 0
        batch_loss = torch.tensor([], requires_grad=True)
        for graph, _ in zip(graph_batch, optimal_param_batch):
            approx = graph_tester(model=model, graph=graph, n_layers=model.p)
            if approx > highest_approx:
                highest_approx = approx
            if approx < lowest_approx:
                lowest_approx = approx
            total_approx += approx
            loss = custom_loss(approx)
            batch_loss = torch.cat([batch_loss, loss])
        if train:
            assert optimizer is not None
            optimizer.zero_grad()  # Zero the gradients
            batch_loss.backward()  # Backward pass on the total loss for the batch
            optimizer.step()  # Update model parameters

        total_loss = total_loss + batch_loss
        total_approx += batch_total_approx
    assert data_loader.batch_size is not None
    avg_loss = torch.sum(total_loss) / (num_batches * data_loader.batch_size)
    avg_approx = total_approx / (num_batches * data_loader.batch_size)
    epoch_summary = InnerEpochSummary(
        model=model,
        loss=avg_loss.item(),
        approx=avg_approx,
        highest_approx=highest_approx,
        lowest_approx=lowest_approx,
    )
    return epoch_summary




def run_dataless_epoch(
    model: QAOAPredictor,
    optimizer: torch.optim.Optimizer,
) -> EpochSummary:
    train_epoch_result = inner_dataless_epoch(
        model=model, train=True, optimizer=optimizer
    )
    test_epoch_result = inner_dataless_epoch(
        model=model, train=False
    )
    return create_epoch_summary(train_epoch_result, test_epoch_result)


def dataless_train(
    model: QAOAPredictor,
    epochs: int,
    lr: float,
    split_ratio: float = 0.8,
    verbose: bool = False,
) -> TrainingSummary:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_summary = TrainingSummary()
    for epoch in range(epochs):
        epoch_summary = run_dataless_epoch(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            split_ratio=split_ratio,
        )
        if verbose:
            epoch_summary.log()
        train_summary.update(epoch_summary)
    return train_summary



def get_qaoa_max_cut_val(params: torch.Tensor, graph: nx.Graph, n_layers: int) -> float:
    qaoa_circuit = QaoaCircuit(graph, n_layers)
    mapping = {
        qaoa_circuit.beta: params[: qaoa_circuit.n_layers],
        qaoa_circuit.gamma: params[qaoa_circuit.n_layers :],
    }
    circuit = qaoa_circuit.assign_parameters(mapping)
    job = qk.execute(circuit, qk.Aer.get_backend("qasm_simulator"), shots=1000)
    result = job.result()
    counts: dict[str, int] = result.get_counts()
    max_count = max(counts, key=lambda x: counts[x])
    qaoa_max_cut_val = -qaoa_circuit.energy_cost(int(max_count, 2))
    return qaoa_max_cut_val


def graph_tester(model, graph: torch.Tensor, n_layers: int) -> float:
    nx_graph = nx.from_numpy_array(graph.numpy())
    max_cut_val, _ = max_cut_brute_force(nx_graph)
    qaoa_max_cut_val = get_qaoa_max_cut_val(
        params=model(graph), graph=nx_graph, n_layers=n_layers
    )
    approx_ratio = qaoa_max_cut_val / max_cut_val
    return approx_ratio


def test_qaoa_perd(
    model, dataset: tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray]
) -> tuple[float, float]:
    graphs, circuits, optimal_params = dataset
    total_error = 0
    total_approx_ratio = 0
    for i in range(len(graphs)):
        graph = graphs[i]
        optimal_param = optimal_params[i]
        circuit = circuits[i]
        error, approx_ratio = graph_tester(
            model, graph, optimal_param, circuit, circuit.n_layers
        )
        total_error += error
        total_approx_ratio += approx_ratio
    avg_error = total_error / len(graphs)
    avg_approx_ratio = total_approx_ratio / len(graphs)
    print(f"Average error: {avg_error}")
    print(f"Average approximation ratio: {avg_approx_ratio}")
    return avg_error, avg_approx_ratio


def load_dataset(path: str) -> tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray]:
    with open(path, "rb") as f:
        x, y, z = pickle.load(f)
        return list(x), list(y), z


def load_dataset_from_params(
    dataset_size: int, n_layers: int, min_graph_size: int, max_graph_size: int
) -> tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray]:
    print(
        f"Loading dataset from data/dataset_G{dataset_size}_P{n_layers}_N{min_graph_size}-{max_graph_size}.pkl"
    )
    return load_dataset(
        f"data/dataset_G{dataset_size}_P{n_layers}_N{min_graph_size}-{max_graph_size}.pkl"
    )


def load_dataset_by_name(name: str, dataset_size: int, n_layers: int) -> QAOADataset:
    if dataset_size is None:
        all_datasets = list_datasets()
        for dataset in all_datasets:
            if dataset[0] == name and dataset[2] == n_layers:
                dataset_size = dataset[1]
                break
    print(f"Loading dataset from data/dataset_{name}_G{dataset_size}_P{n_layers}.pkl")
    graphs, _, params = load_dataset(
        f"data/dataset_{name}_P{n_layers}_N{dataset_size}.pkl"
    )
    non_empty_indices = [
        i for i, graph in enumerate(graphs) if graph.number_of_edges() > 0
    ]
    graphs = [graphs[i] for i in non_empty_indices]
    params = params[non_empty_indices]
    return QAOADataset(graphs, params)


def get_dataset_info(file: str) -> tuple[str, int, int]:
    try:
        name, n_layers, num_of_graphs = file.split("_")[1:]
        n_layers = n_layers[1:]
        num_of_graphs = num_of_graphs[1:].split(".")[0]
        return name, int(num_of_graphs), int(n_layers)
    except:
        raise Exception(f"Dataset {file} is not formatted correctly")


def list_datasets(verbose: bool = False) -> list[tuple[str, int, int]]:
    """
    List all the datasets in the data folder

    Returns:
        A list of tuples of the form (name, num_of_graphs, n_layers)

    """
    datasets = sorted(
        [
            get_dataset_info(file)
            for file in os.listdir("data")
            if file.endswith(".pkl")
        ],
        key=lambda x: -x[2],
    )
    if verbose:
        for name, num_of_graphs, n_layers in datasets:
            print(f"Dataset {name} has {num_of_graphs} graphs and {n_layers} layers")
    return datasets


def filter_graphs(
    data_set: tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray], graph_size: int
) -> tuple[list[nx.Graph], list[QuantumCircuit], np.ndarray]:
    graphs, circuits, optimal_params = data_set
    new_graphs = [graph for graph in graphs if graph.number_of_nodes() == graph_size]
    circuits = [
        circuits[i]
        for i in range(len(circuits))
        if graphs[i].number_of_nodes() == graph_size
    ]
    optimal_params = [
        optimal_params[i]
        for i in range(len(optimal_params))
        if graphs[i].number_of_nodes() == graph_size
    ]
    return new_graphs, circuits, np.array(optimal_params)