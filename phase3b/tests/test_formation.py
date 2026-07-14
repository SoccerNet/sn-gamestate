import numpy as np
from phase3b.gt_labels import _match_formation

def test_match_formation():
    assert _match_formation((4, 4, 2)) == "4-4-2"
    assert _match_formation((3, 5, 2)) == "3-5-2"
    assert _match_formation((4, 3, 3)) == "4-3-3"
    assert _match_formation((4, 4, 1)) == "4-4-2"   # nearest by L1