import networkx as nx

def gen_random_graph(n_nodes: int, p_edge: float) -> nx.Graph:
    """Generate a random graph using Networkx."""
    return nx.gnp_random_graph(n_nodes, p=p_edge)