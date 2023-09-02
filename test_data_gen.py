from gen_data import *
import unittest

class TestDataGen(unittest.TestCase):

    def test_generate_graphs(self):
        graphs = generate_graphs(10, 10, 10)
        for graph in graphs:
            assert graph is not None
            assert len(graph.nodes) == 10, f"Graph has {len(graph.nodes)} nodes"


    def test_generate_qaoa(self):
        graphs = generate_graphs(10, 10, 10)
        for graph in graphs:
            qaoa = qaoa_from_graph(graph)
            assert qaoa is not None
            
