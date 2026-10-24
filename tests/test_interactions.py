import numpy as np
import pytest
from src.interactions import merge_intervals,intersect_intervals

def coverage(intervals,broadcasts):
    a=np.array(sorted(intervals),dtype=np.int64).reshape(-1,3)
    u,s,e=merge_intervals(a[:,0],a[:,1],a[:,2])
    b=np.array(broadcasts,dtype=np.int64)
    uu,bb,ss=intersect_intervals(u,s,e,b[:,0],b[:,1],np.arange(len(b),dtype=np.int64))
    d={}
    for user,bid,seconds in zip(uu,bb,ss): d[(user,bid)]=d.get((user,bid),0)+seconds
    return d

def test_union_partial_overlap_duplicate_and_multiple_broadcasts():
    d=coverage([(1,10,50),(1,30,80),(1,30,80),(1,80,180),(2,0,79)],[(0,100),(100,200)])
    assert d=={(1,0):90,(1,1):80,(2,0):79}
    assert [p for p,x in d.items() if x*5>=100*4]==[(1,0),(1,1)]

def test_disjoint_sessions_sum_to_eighty_percent():
    assert coverage([(0,0,40),(0,60,100)],[(0,100)])=={(0,0):80}

def test_midnight_invalid_and_exact_boundary():
    assert coverage([(0,86380,86480),(0,86490,86490)],[(86300,86400),(86400,86500)])=={(0,0):20,(0,1):80}
    assert coverage([(0,100,200)],[(0,100),(200,300)])=={}

def test_nonmonotone_broadcast_ends():
    assert coverage([(0,150,170)],[(0,300),(100,120),(160,200)])=={(0,0):20,(0,2):10}

def test_randomized_against_bruteforce_union():
    rng=np.random.default_rng(17)
    for _ in range(60):
        intervals=[(int(rng.integers(0,3)),int(s),int(s+rng.integers(1,40))) for s in rng.integers(0,200,size=30)]
        broadcasts=sorted([(int(s),int(s+rng.integers(1,90))) for s in rng.integers(0,200,size=15)])
        actual=coverage(intervals,broadcasts)
        expected={}
        for user in range(3):
            points=set()
            for u,s,e in intervals:
                if u==user: points.update(range(s,e))
            for j,(s,e) in enumerate(broadcasts):
                n=len(points.intersection(range(s,e)))
                if n: expected[(user,j)]=n
        assert actual==expected

def test_timezone_normalization_and_cutoff_clipping():
    from datetime import datetime,timezone
    a=int(datetime.fromisoformat('2020-03-29T01:00:00+00:00').timestamp())
    b=int(datetime.fromisoformat('2020-03-29T04:00:00+03:00').timestamp())
    assert a==b
    from src.preprocessing import stamp
    assert stamp('2020-03-29T01:00:00+00:00')==stamp('2020-03-29T04:00:00+03:00')
    assert coverage([(0,a,a+100)],[(b,b+100)])=={(0,0):100}
    assert coverage([(0,0,min(150,75))],[(0,100)])=={(0,0):75}
