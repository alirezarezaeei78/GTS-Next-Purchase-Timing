"""Select numerical input count, context and grouping without test outcomes.

Every candidate numerical feature is optional, including the former base set.
The reported winner is the best tested configuration, not a global optimum.
"""
from dataclasses import dataclass,asdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

@dataclass
class SelectionConfig:
    seed:int=42
    max_train_rows:int=20000
    max_validation_rows:int=12000
    probe_iterations:int=80
    max_groups:int=8

class AdaptiveInputs:
    def __init__(self,config=None):
        self.config=config or SelectionConfig()

    def fit(self,x_train,y_train,x_validation,y_validation,feature_names,
            profiles_train,profiles_validation):
        c=self.config;rng=np.random.default_rng(c.seed)
        assert x_train.ndim==3 and x_validation.shape[1:]==x_train.shape[1:]
        assert len(feature_names)==x_train.shape[-1]
        # Rank original named variables at observed input rows, using train only.
        last=np.asarray(x_train[:,-1,:],float);y=np.asarray(y_train,float).reshape(-1)
        centered=last-last.mean(0);dy=y-y.mean()
        denom=np.sqrt((centered**2).sum(0)*(dy**2).sum())
        score=np.divide(abs(centered.T@dy),denom,out=np.zeros(last.shape[1]),where=denom>1e-12)
        active=np.flatnonzero(np.std(last,axis=0)>1e-9)
        self.ranking_=active[np.argsort(-score[active],kind='stable')]
        self.feature_names_=list(feature_names)
        self.training_scores_={name:float(v) for name,v in zip(feature_names,score)}
        ti=np.sort(rng.choice(len(y),min(c.max_train_rows,len(y)),replace=False))
        vi=np.sort(rng.choice(len(y_validation),min(c.max_validation_rows,len(y_validation)),replace=False))
        yt=y[ti];yv=np.asarray(y_validation,float).reshape(-1)[vi]
        xt=np.asarray(x_train[ti]);xv=np.asarray(x_validation[vi])
        self.profile_scaler_=StandardScaler().fit(profiles_train)
        pt=self.profile_scaler_.transform(profiles_train)
        pv=self.profile_scaler_.transform(profiles_validation)
        self.clusterers_={};group_inputs={0:(None,None)}
        with threadpool_limits(limits=4):
            for k in range(2,min(c.max_groups,len(pt)-1)+1):
                km=KMeans(k,n_init=10,random_state=c.seed).fit(pt)
                self.clusterers_[k]=km
                group_inputs[k]=(np.eye(k,dtype=np.float32)[km.predict(pt[ti])],
                                 np.eye(k,dtype=np.float32)[km.predict(pv[vi])])
        cache={};self.trials_=[]
        def evaluate(count,length,groups):
            key=(int(count),int(length),int(groups))
            if key in cache:return cache[key]
            selected=self.ranking_[:count]
            a=xt[:,-length:,selected].reshape(len(xt),-1)
            b=xv[:,-length:,selected].reshape(len(xv),-1)
            if groups:
                gt,gv=group_inputs[groups]
                a=np.column_stack([a,gt]);b=np.column_stack([b,gv])
            model=HistGradientBoostingRegressor(loss='quantile',quantile=.65,
                max_iter=c.probe_iterations,max_leaf_nodes=15,min_samples_leaf=25,
                learning_rate=.07,l2_regularization=.1,early_stopping=False,random_state=c.seed)
            with threadpool_limits(limits=4):
                model.fit(a,np.log1p(np.minimum(yt,np.quantile(yt,.99))))
                pred=np.maximum(0,np.expm1(model.predict(b)))
            error=float(np.abs(yv-pred).mean())
            row={'features':count,'context':length,'groups':groups,'validation_MAE':error}
            self.trials_.append(row);cache[key]=row
            print('Input search',len(cache),key,'validation MAE',round(error,5),flush=True)
            return row
        def best():
            return min(cache.values(),key=lambda r:(r['validation_MAE'],r['features'],r['groups'],r['context']))
        # Exhaustive count scans, followed by coordinate searches for context and
        # group count; two rounds allow their interaction to change the subset.
        count=len(self.ranking_);length=x_train.shape[1];groups=0
        for _ in range(2):
            for n in range(1,len(self.ranking_)+1):evaluate(n,length,groups)
            win=best();count=win['features'];length=win['context'];groups=win['groups']
            for k in group_inputs:evaluate(count,length,k)
            win=best();count=win['features'];length=win['context'];groups=win['groups']
            for length_ in range(1,x_train.shape[1]+1):evaluate(count,length_,groups)
            win=best();count=win['features'];length=win['context'];groups=win['groups']
        self.best_=best();self.indices_=self.ranking_[:self.best_['features']]
        return self

    def transform(self,x,profiles):
        out=np.asarray(x[:,-self.best_['context']:,self.indices_],np.float32)
        k=self.best_['groups']
        if k:
            labels=self.clusterers_[k].predict(self.profile_scaler_.transform(profiles))
            indicators=np.repeat(np.eye(k,dtype=np.float32)[labels][:,None,:],out.shape[1],axis=1)
            out=np.concatenate([out,indicators],axis=2)
        return out

    def report(self):
        return {'selection':self.best_,'selected_features':[self.feature_names_[i] for i in self.indices_],
            'training_rank_scores':self.training_scores_,'trials':self.trials_,
            'budget':asdict(self.config),'objective':'Validation MAE of a boosting probe; test labels are not an input',
            'scope':'All nonconstant numerical candidates optional; coordinate search; no claim of global optimality'}
