import torch, numpy as np
from phase3b.train import split_items

def test_split_by_seq():
    items = [{"seq": "078", "label": 0}, {"seq": "021", "label": 1}]
    tr, va = split_items(items, val_seqs=[78])
    assert len(tr) == 1 and tr[0]["seq"] == "021"
    assert len(va) == 1 and va[0]["seq"] == "078"