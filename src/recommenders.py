"""Frequency, metadata cosine, association rules, implicit ALS and hybrids."""
from dataclasses import dataclass
import re
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from implicit.cpu.als import AlternatingLeastSquares

@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    half_life: int=0
    history_days: int=0
    availability_power: float=0
    support: int=2
    confidence: float=.03
    factors: int=32
    regularization: float=.1
    content_weight: float=0
    association_weight: float=0
    als_weight: float=0
    channel_weight: float=0
    metadata_weight: float=2
    fusion: str='score'

def specifications():
    specs=[ModelSpec('global_popularity','popularity'),ModelSpec('historical_frequency_unfiltered','unfiltered'),ModelSpec('frequency','frequency')]
    specs += [ModelSpec(f'recency_{d}d','frequency',half_life=d) for d in [7,14,28,56]]
    specs += [ModelSpec('last28_frequency','frequency',history_days=28)]
    specs += [ModelSpec(f'availability_p{p}','availability',half_life=28,availability_power=p) for p in [.25,.5,1.0]]
    specs += [ModelSpec(f'raw_availability_p{p}','availability',availability_power=p) for p in [.25,.5,1.0]]
    specs += [ModelSpec('channel_time','channel',half_life=28)]
    specs += [ModelSpec('content','content',half_life=28),ModelSpec('content_metadata4','content',half_life=28,metadata_weight=4)]
    specs += [ModelSpec(f'association_s{s}_c{c}','association',half_life=28,support=s,confidence=c) for s,c in [(2,.03),(5,.03),(5,.10)]]
    specs += [ModelSpec(f'als_{f}_r{r}','als',half_life=28,factors=f,regularization=r) for f,r in [(32,.1),(64,.5)]]
    specs += [ModelSpec(f'hybrid_c{c}_a{a}','hybrid',half_life=28,availability_power=.25,content_weight=c,association_weight=a) for c,a in [(.05,.05),(.1,.1),(.2,.2),(0,.15),(.15,0)]]
    specs += [ModelSpec('hybrid_als','hybrid',half_life=28,availability_power=.25,content_weight=.05,association_weight=.05,als_weight=.1),
              ModelSpec('hybrid_raw','hybrid',availability_power=.25,content_weight=.1,association_weight=.1),
              ModelSpec('hybrid_channel','hybrid',half_life=28,availability_power=.25,content_weight=.05,association_weight=.05,channel_weight=.1),
              ModelSpec('rank_fusion','hybrid',half_life=28,content_weight=.1,association_weight=.1,fusion='rank')]
    return specs

def interaction_matrix(events,n_users,items,cutoff,half_life=0,history_days=0):
    if history_days: events=events.filter(pl.col('start_time')>=cutoff-history_days*86400)
    if not events.height: return sparse.csr_matrix((n_users,len(items)),dtype=np.float32)
    cols=np.searchsorted(items,events['tv_show_id'].to_numpy())
    values=np.ones(events.height,dtype=np.float32)
    if half_life: values=np.exp2(-(cutoff-events['end_time'].to_numpy())/(half_life*86400)).astype(np.float32)
    return sparse.coo_matrix((values,(events['user_idx'].to_numpy(),cols)),shape=(n_users,len(items))).tocsr()

def metadata_features(schedule,items,metadata_weight=2):
    fields=['tv_show_title','tv_show_category','tv_show_genre_1','tv_show_genre_2','tv_show_genre_3','actors','director','year_of_production']
    catalog=schedule.group_by('tv_show_id').agg([pl.col(c).drop_nulls().unique().sort().head(12).str.join(' ').alias(c) for c in fields])
    rows={r['tv_show_id']:r for r in catalog.iter_rows(named=True)}
    docs=[]; meta=[]; people=[]
    for item in items:
        r=rows.get(int(item),{})
        docs.append(r.get('tv_show_title','') or '')
        tokens=[]
        for c in fields[1:5]:
            tokens += [c+'_'+re.sub(r'\W+','_',w.lower()) for w in (r.get(c,'') or '').split(',') if w.strip()]
        meta.append(' '.join(tokens))
        people.append(' '.join((r.get(c,'') or '') for c in fields[5:]))
    def vectorize(texts):
        if not any(x.strip() for x in texts): return sparse.csr_matrix((len(items),1),dtype=np.float32)
        return TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1,max_features=24000,sublinear_tf=True,dtype=np.float32).fit_transform(texts)
    return normalize(sparse.hstack([vectorize(docs),metadata_weight*vectorize(meta),.5*vectorize(people)],format='csr'))

def association_rules(matrix,support=2,confidence=.03,topk=120):
    binary=matrix.copy(); binary.data[:]=1
    users=np.asarray(binary.sum(axis=0)).ravel()
    co=(binary.T@binary).tocoo()
    conf=co.data/np.maximum(users[co.row],1)
    keep=(co.row!=co.col)&(co.data>=support)&(conf>=confidence)
    # Shrunk confidence with square-root inverse target prevalence (partial lift).
    vals=conf[keep]*(co.data[keep]/(co.data[keep]+5))/np.sqrt(np.maximum(users[co.col[keep]],1)/matrix.shape[0])
    rules=sparse.csr_matrix((vals,(co.row[keep],co.col[keep])),shape=(matrix.shape[1],matrix.shape[1]))
    for i in range(rules.shape[0]):
        start,end=rules.indptr[i:i+2]
        if end-start>topk:
            drop=np.argpartition(rules.data[start:end],end-start-topk)[:end-start-topk]
            rules.data[start+drop]=0
    rules.eliminate_zeros()
    return rules

def row_scale(scores):
    scores=np.maximum(scores,0).astype(np.float32,copy=False)
    return scores/np.maximum(scores.max(axis=1,keepdims=True),1e-12)

def rank_signal(scores):
    ranks=np.argsort(np.argsort(-scores,axis=1,kind='stable'),axis=1,kind='stable')
    return ((scores>0)/(ranks+5)).astype(np.float32)

class ScoreEngine:
    def __init__(self,events,historical_schedule,future_schedule,items,n_users,cutoff,predict_users):
        assert not events.height or events['end_time'].max()<=cutoff
        assert future_schedule['start_time'].min()>=cutoff
        self.events=events; self.hist=historical_schedule; self.future=future_schedule
        self.items=items; self.n_users=n_users; self.cutoff=cutoff; self.users=np.asarray(predict_users,dtype=int)
        self.candidates=np.searchsorted(items,np.sort(future_schedule['tv_show_id'].unique().to_numpy()))
        self.cache={}; self.matrices={}
        self.opportunities=np.bincount(np.searchsorted(items,future_schedule['tv_show_id'].to_numpy()),minlength=len(items)).astype(np.float32)
        self.historical_opportunities=np.bincount(np.searchsorted(items,historical_schedule['tv_show_id'].to_numpy()),minlength=len(items)).astype(np.float32)
        self.popularity=np.asarray(self.matrix(0).sum(axis=0)).ravel()

    def matrix(self,half_life=0,history_days=0):
        key=(half_life,history_days)
        if key not in self.matrices: self.matrices[key]=interaction_matrix(self.events,self.n_users,self.items,self.cutoff,half_life,history_days)
        return self.matrices[key]

    def content(self,weight=2):
        key=('content',weight)
        if key not in self.cache:
            features=metadata_features(pl.concat([self.hist,self.future]),self.items,weight)
            profile=normalize(self.matrix(28)[self.users]@features)
            self.cache[key]=(profile@features[self.candidates].T).toarray()
        return self.cache[key]

    def association(self,support=2,confidence=.03):
        key=('association',support,confidence)
        if key not in self.cache:
            matrix=self.matrix(28); rules=association_rules(matrix,support,confidence)
            profile=matrix[self.users].copy(); profile.data=np.log1p(profile.data)
            self.cache[key]=(profile@rules[:,self.candidates]).toarray()
        return self.cache[key]

    def als(self,factors=32,regularization=.1):
        key=('als',factors,regularization)
        if key not in self.cache:
            matrix=self.matrix(28).copy(); matrix.data=10*np.log1p(matrix.data)
            model=AlternatingLeastSquares(factors=factors,regularization=regularization,iterations=25,num_threads=2,random_state=42)
            model.fit(matrix,show_progress=False)
            scores=model.user_factors[self.users]@model.item_factors[self.candidates].T
            scores[:,np.asarray(matrix.sum(axis=0)).ravel()[self.candidates]==0]=0
            self.cache[key]=np.maximum(scores,0)
        return self.cache[key]

    def channel_time(self):
        if 'channel' in self.cache: return self.cache['channel']
        channels=np.sort(pl.concat([self.hist['channel_id'],self.future['channel_id']]).unique().to_numpy())
        slots=len(channels)*6
        def slot(frame): return np.searchsorted(channels,frame['channel_id'].to_numpy())*6+(frame['start_time'].to_numpy()//14400)%6
        e=self.events
        values=np.exp2(-(self.cutoff-e['end_time'].to_numpy())/(28*86400)).astype(np.float32)
        x=sparse.coo_matrix((values,(e['user_idx'].to_numpy(),slot(e))),shape=(self.n_users,slots)).tocsr()[self.users]
        exposure=np.bincount(slot(self.hist),minlength=slots)
        x=x.multiply(1/np.maximum(exposure+10,1))
        f=self.future
        future=sparse.coo_matrix((np.ones(f.height),(slot(f),np.searchsorted(self.items,f['tv_show_id'].to_numpy()))),shape=(slots,len(self.items))).tocsr()
        self.cache['channel']=(x@future[:,self.candidates]).toarray().astype(np.float32)
        return self.cache['channel']

    def score(self,spec):
        if spec.family=='popularity': return np.broadcast_to(self.popularity[self.candidates],(len(self.users),len(self.candidates))).copy()
        if spec.family=='content': return self.content(spec.metadata_weight).copy()
        if spec.family=='association': return self.association(spec.support,spec.confidence).copy()
        if spec.family=='als': return self.als(spec.factors,spec.regularization).copy()
        if spec.family=='channel': return self.channel_time().copy()
        scores=self.matrix(spec.half_life,spec.history_days)[self.users][:,self.candidates].toarray()
        if spec.availability_power:
            scores*=((self.opportunities[self.candidates]+2)/(self.historical_opportunities[self.candidates]+2))**spec.availability_power
        if spec.family=='hybrid':
            transform=rank_signal if spec.fusion=='rank' else row_scale
            scores=transform(scores)
            for weight,component in [(spec.content_weight,self.content),(spec.association_weight,self.association),(spec.als_weight,self.als),(spec.channel_weight,self.channel_time)]:
                if weight: scores+=weight*transform(component())
        return scores

    def recommend(self,spec):
        # Unfiltered baseline deliberately measures the cost of ignoring availability.
        if spec.family=='unfiltered':
            scores=self.matrix()[self.users].toarray(); candidates=np.arange(len(self.items))
        else: scores=self.score(spec); candidates=self.candidates
        # Stable lexicographic tie break: main score, training popularity, ascending ID.
        pop=self.popularity[candidates]
        order=np.lexsort((self.items[candidates],-pop))
        sorted_scores=scores[:,order]
        # Stable score sorting preserves popularity/ID tie order.
        pred=np.empty((len(self.users),5),dtype=np.int64)
        for row in range(len(self.users)):
            selected=order[np.argsort(-sorted_scores[row],kind='stable')[:5]]
            pred[row]=self.items[candidates[selected]]
        return pred
