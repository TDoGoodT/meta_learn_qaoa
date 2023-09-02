import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from networkx import Graph
from qiskit.algorithms.optimizers import COBYLA


def max_cut_brute_force(G) -> tuple[int, tuple[set, set]]:
    """
    Find the Max-Cut of graph G using brute force.
    
    G: A networkx graph.
    
    Returns:
    - Max cut value
    - A tuple with two sets representing the partition.
    """
    nodes = G.nodes()
    n = G.number_of_nodes()
    max_cut_value = 0
    best_partition = None

    # Iterate over all possible partitions (combinations of nodes)
    for i in range(1, n // 2 + 1):  # Only need to go up to half since the rest are symmetric
        for partition in itertools.combinations(nodes, i):
            set_A = set(partition)
            set_B = set(nodes) - set_A
            
            cut_value = 0
            for u, v in G.edges():
                if (u in set_A and v in set_B) or (u in set_B and v in set_A):
                    cut_value += G[u][v].get('weight', 1)

            if cut_value > max_cut_value:
                max_cut_value = cut_value
                best_partition = (set_A, set_B)

    return max_cut_value, best_partition


class QaoaCircuit(QuantumCircuit):
    def __init__(self, graph: Graph, p: int, **kwargs):
        regs = graph.number_of_nodes()
        super(QaoaCircuit, self).__init__(regs, **kwargs)
        self.graph = graph
        self.solution = max_cut_brute_force(graph)[0]
        self.n_layers = p
        self.gamma = ParameterVector("gamma", p)
        self.beta = ParameterVector("beta", p)
        self.build(p)

    def build(self, n_layers: int):
        for i in range(n_layers):
            for edge in self.graph.edges:
                self.rzz(2 * self.gamma[i], edge[0], edge[1])
            for node in self.graph.nodes:
                self.rx(2 * self.beta[i], node)
        self.measure_all()
        return self

    def energy_cost(self, state: int):
        str_state = bin(state)[2:].zfill(self.graph.number_of_nodes())
        return -sum([1 if str_state[i] != str_state[j] else 0 for i, j in self.graph.edges])
    
    def run(self, params: np.ndarray, shots: int) -> dict[str, int]:
        """
        Runs the circuit with the given parameters.
        Returns the counts of the result.
        """
        from qiskit import Aer, execute
        params_dict = {self.beta: params[:self.n_layers], self.gamma: params[self.n_layers:]}
        qc = self.assign_parameters(params_dict)
        job = execute(qc, Aer.get_backend("qasm_simulator"), shots=shots)
        result = job.result()
        counts: dict[str, int] = result.get_counts()
        return counts
    
    def optimize(self, params: np.ndarray, iterations: int, shots: int) -> tuple[float, np.ndarray]:
        """
        Runs the circuit with the given parameters.
        Returns the approximation ratio and the parameters.
        """
        optimizer = COBYLA(maxiter=iterations)
        def objective_function(params) -> float:
            """Objective function for QAOA."""
            counts = self.run(params, shots)
            max_count = max(counts, key=lambda x: counts[x])
            return self.energy_cost(int(max_count, 2))
        op = optimizer.minimize(objective_function, params)
        return (-objective_function(op.x) / self.solution), op.x
