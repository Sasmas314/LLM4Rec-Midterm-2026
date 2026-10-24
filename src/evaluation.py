"""Standard binary MAP@5, set metrics and explicit ordered diagnostics."""
import numpy as np

METRICS=['map5','recall5','precision5','ndcg5','ordered_agreement5','exact_ordered5','graded_ndcg5','hitrate5']

def per_user_metrics(predictions,truth):
    rows=[]
    discounts=1/np.log2(np.arange(2,7))
    for pred,actual in zip(predictions,truth):
        actual=list(actual); aset=set(actual)
        hit=np.array([int(p in aset and p not in pred[:j]) for j,p in enumerate(pred)],dtype=float)
        denom=min(5,len(actual))
        if not denom: raise ValueError('Empty ground truth is not evaluable')
        ap=float(np.sum(np.cumsum(hit)/np.arange(1,6)*hit)/denom)
        ndcg=float(np.sum(hit*discounts)/discounts[:denom].sum())
        agreement=sum(int(j<len(actual) and pred[j]==actual[j]) for j in range(5))/5
        gains={a:5-j for j,a in enumerate(actual[:5])}
        graded=sum(gains.get(int(p),0)*discounts[j] for j,p in enumerate(pred))/sum((5-j)*discounts[j] for j in range(denom))
        rows.append([ap,hit.sum()/len(actual),hit.sum()/5,ndcg,agreement,float(list(pred)==actual),graded,float(hit.sum()>0)])
    return np.asarray(rows,dtype=float)

def evaluate(predictions,truth):
    return dict(zip(METRICS,per_user_metrics(predictions,truth).mean(axis=0)))

def paired_bootstrap(first,second,seed=42,repeats=2000):
    delta=first-second
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(repeats)])
    return {'mean_map5_difference':float(delta.mean()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975)),'users':len(delta),'bootstrap_replicates':repeats}
