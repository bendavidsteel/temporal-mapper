import networkx as nx
import numpy as np

def topic_contract(mapper, v, dir='forward'):
    G = mapper.graph.to_undirected()
    # a list, not the neighbour iterator: it is read more than once below
    neighbours = list(G.neighbors(v))
    d = len(neighbours)
    if d == 0:
        return None
    if (mapper.graph.in_degree(v) == 1):
        single_edge_contract(mapper.graph, v)
    if (d==1) and (mapper.graph.out_degree(v) == 1):
        # this is v a source
        return None
    idx = G.nodes[v]['slice_no']
    w = mapper.weights[idx,:]
    sizes = np.array(
        [np.sum(w[mapper.get_vertex_data(u)]) for u in neighbours]
    )
    total = np.sum(sizes)
    if total == 0:
        return None
    sizes = sizes / total
    impurity = 1 - np.sum(sizes**2)
    if impurity <= 1/(2*d):
        # highly homogeneous, merge node into most similar
        best = neighbours[int(np.argmax(sizes))]
        # write to mapper.graph: G is an undirected copy, so changes to it are discarded
        topic = nx.get_node_attributes(mapper.graph, 'topic')
        topic[v] = topic[best]
        nx.set_node_attributes(mapper.graph, topic, 'topic')

def single_edge_contract(G, v):
    # special case to check for large semantic drift.
    drift = nx.get_edge_attributes(G, 'drift')
    drifts = [drift[e] for e in G.edges()]
    threshold = np.median(drifts)
    # this line is stupid but I can't figure out
    # how else to get just one edge out of an OutEdgeDataView
    e = [e for e in G.in_edges(v)][0] 
    if drift[e] < threshold:
        topic = nx.get_node_attributes(G, 'topic')
        topic[v] = topic[e[0]]
        nx.set_node_attributes(G, topic, 'topic')
