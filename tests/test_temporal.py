import numpy as np
import sys, os
import networkx as nx
import pickle as pkl
import warnings
from sklearn.cluster import DBSCAN

import temporalmapper as tm

data_folder = 'data/'

def compute_temporal_mapper(kwargs={}):
    """ Integration test from loading data to producing a graph. """
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    clusterer = DBSCAN()
    TM = tm.TemporalMapper(
        timestamps,
        data,
        clusterer,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        TM.build()
    return TM

def test_fit_mapper(kwargs={}):
    """ Integration test from loading data to fitting a graph. """
    data_time = np.load(data_folder+"genus1_test.npy")
    data_unsort = data_time[:,1].T
    timestamps_unsort = data_time[:,0].T
    sorted_indices = np.argsort(timestamps_unsort)
    data = data_unsort[sorted_indices]
    timestamps = timestamps_unsort[sorted_indices]
    clusterer = DBSCAN()
    TM = tm.TemporalMapper(
        clusterer = clusterer,
        **kwargs,
    )
    X = np.vstack([data, timestamps]).T
    TM.fit(X)
    assert hasattr(TM, "graph")

def test_random_utilities():
    TM = compute_temporal_mapper(kwargs={
        'n_slices':10
    })
    TM.get_vertex_data('0:0')
    TM.assign_topics()
    TM.vertex_subgraph('0:1')

def test_compute_temporal_mapper():
    parameters = [
        {'n_slices':8, 'slice_method':'time'},
        {'n_slices':8, 'slice_method':'data'},
        {'n_slices':8, 'kernel':tm.kernels.square, 'density_based':False}, # vanilla mapper
        {'n_slices':3, 'overlap':0.1, 'n_neighbors':10}
    ]
    for i in range(len(parameters)):
        assert hasattr(compute_temporal_mapper(kwargs=parameters[i]), "graph")


class _StubMapper:
    """The parts of TemporalMapper that topic_contract reads."""
    def __init__(self, graph, weights, vertex_data):
        self.graph, self.weights, self._vertex_data = graph, weights, vertex_data

    def get_vertex_data(self, node):
        return self._vertex_data[node]


def _two_parent_mapper(n_a, n_b):
    # v (slice 1) has two parents in slice 0 sharing n_a and n_b of its points
    G = nx.DiGraph()
    G.add_node('0:0', slice_no=0, topic=0)
    G.add_node('0:1', slice_no=0, topic=1)
    G.add_node('1:0', slice_no=1, topic=2)
    G.add_edge('0:0', '1:0', drift=0.0)
    G.add_edge('0:1', '1:0', drift=0.0)
    n = n_a + n_b
    weights = np.ones((2, n))
    vertex_data = {
        '0:0': np.arange(n_a),
        '0:1': np.arange(n_a, n),
        '1:0': np.arange(n),
    }
    return _StubMapper(G, weights, vertex_data)


def test_topic_contract_merges_homogeneous_node():
    from temporalmapper.topics import topic_contract
    m = _two_parent_mapper(n_a=9, n_b=1)  # impurity 0.18 <= 1/(2*2)
    topic_contract(m, '1:0')
    assert m.graph.nodes['1:0']['topic'] == m.graph.nodes['0:0']['topic']


def test_topic_contract_keeps_mixed_node():
    from temporalmapper.topics import topic_contract
    m = _two_parent_mapper(n_a=5, n_b=5)  # impurity 0.5 > 1/(2*2)
    topic_contract(m, '1:0')
    assert m.graph.nodes['1:0']['topic'] == 2
