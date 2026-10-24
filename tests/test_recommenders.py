import numpy as np
from scipy import sparse
from src.recommenders import association_rules,row_scale,rank_signal

def test_directional_confidence_support_and_self_rule_exclusion():
    matrix=sparse.csr_matrix([[1,1,0],[1,1,0],[1,0,1],[1,0,0]],dtype=np.float32)
    rules=association_rules(matrix,support=2,confidence=.03).toarray()
    assert np.all(np.diag(rules)==0)
    assert rules[0,1]>0 and rules[1,0]>0
    assert np.all(rules[2]==0) and np.all(rules[:,2]==0)
    strict=association_rules(matrix,support=2,confidence=.75).toarray()
    assert strict[0,1]==0 and strict[1,0]>0

def test_normalization_and_empty_user_fusion():
    x=np.array([[2,1,0],[0,0,0]],dtype=np.float32)
    assert np.allclose(row_scale(x),[[1,.5,0],[0,0,0]])
    assert np.isfinite(rank_signal(x)).all()
    assert np.all(rank_signal(x)[1]==0)
