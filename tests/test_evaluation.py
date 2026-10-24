import numpy as np
import pytest
from src.evaluation import evaluate

def test_perfect_and_reverse_map_order_distinction():
    truth=[[1,2,3,4,5]]
    good=evaluate(np.array([[1,2,3,4,5]]),truth)
    reverse=evaluate(np.array([[5,4,3,2,1]]),truth)
    assert good['map5']==pytest.approx(1)
    assert reverse['map5']==pytest.approx(1) # Standard MAP rewards prediction rank of hits, not ground-truth permutation.
    assert reverse['ordered_agreement5']==pytest.approx(.2)
    assert reverse['graded_ndcg5']<1

def test_hand_calculated_average_precision():
    value=evaluate(np.array([[8,1,9,2,10]]),[[1,2,3,4,5]])
    assert value['map5']==pytest.approx((1/2+2/4)/5)
    assert value['recall5']==pytest.approx(.4)
    assert value['precision5']==pytest.approx(.4)

def test_sparse_labels_denominator_and_empty():
    assert evaluate(np.array([[1,2,3,4,5]]),[[1,2]])['map5']==pytest.approx(1)
    with pytest.raises(ValueError): evaluate(np.array([[1,2,3,4,5]]),[[]])
