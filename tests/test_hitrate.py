import numpy as np
import pytest
from src.evaluation import evaluate

def test_hitrate_and_duplicate_prediction_penalty():
    x=evaluate(np.array([[1,1,9,8,7],[9,8,7,6,5]]),[[1,2],[1,2]])
    assert x['hitrate5']==.5
    assert x['precision5']==pytest.approx(.1)
    assert x['map5']==pytest.approx(.25)
