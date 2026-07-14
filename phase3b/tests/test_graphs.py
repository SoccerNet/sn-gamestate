import numpy as np
from phase3b.graphs import knn_edges, LABEL2ID

def test_knn_edges_shape():
    P = np.array([[0,0],[1,0],[2,0],[3,0]], float)
    e = knn_edges(P, k=2)
    assert e.shape[0] == 2 and e.shape[1] == 4 * 2   # n*k directed edges
    assert e.max() < 4 and e.min() >= 0

def test_label_map():
    assert LABEL2ID == {"low": 0, "mid": 1, "high": 2}