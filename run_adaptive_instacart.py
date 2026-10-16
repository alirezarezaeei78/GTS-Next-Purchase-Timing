"""A fresh, matched fixed/adaptive GTS experiment on full Instacart histories.

This CUDA implementation is a new experiment, not a recreation of archived
Keras predictions. Both fixed and adaptive arms use the same code and budgets.
"""
from pathlib import Path
import argparse,copy,gc,json,os,sys,time
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--benchmark-root',type=Path,default=ROOT/'work')
parser.add_argument('--experiment-root',type=Path,default=ROOT/'work/adaptive/instacart')
parser.add_argument('--cuda-runtime',type=Path,help='Optional directory of an existing compatible PyTorch installation')
parser.add_argument('--stage',choices=['select','train','all'],default='all')
parser.add_argument('--seeds',type=int,nargs='+',default=[42,7,1400,2026,31337])
parser.add_argument('--arms',nargs='+',default=['fixed','adaptive'])
args=parser.parse_args()
if args.cuda_runtime and args.cuda_runtime.exists():sys.path.insert(0,str(args.cuda_runtime))
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from adaptive_inputs import AdaptiveInputs

def dump(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def log(*s):print(time.strftime('%H:%M:%S'),*s,flush=True)
def metrics(y,p):
    e=np.abs(np.asarray(p)-np.asarray(y))
    return {'MAE':float(e.mean()),'RMSE':float(np.sqrt((e**2).mean())),
            **{f'within_{k}_pct':float(100*(e<k).mean()) for k in [1,2,3]}}

base=args.benchmark_root/'prepared/instacart_pairs';out=args.experiment_root;out.mkdir(parents=True,exist_ok=True)
split=np.load(base/'split.npy');tr=split==0;va=split==1;te=split==2
y=np.load(base/'y.npy')[:,0].astype(np.float64)
group=np.load(base/'group.npy');profiles=np.load(base/'profiles.npy');profile_rows=profiles[group]
raw=np.load(out/'candidate_windows.npy',mmap_mode='r');names=json.loads((out/'candidate_names.json').read_text())
if args.stage in ['select','all'] and not (out/'selection.joblib').exists():
    log('Selecting optional inputs without reading test targets')
    controller=AdaptiveInputs().fit(raw[tr],y[tr],raw[va],y[va],names,profile_rows[tr],profile_rows[va])
    joblib.dump(controller,out/'selection.joblib');dump(out/'selection_report.json',controller.report())
    log('Chosen configuration',controller.best_)
if args.stage=='select':sys.exit(0)

import torch
from torch import nn
torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark=False
hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
log('Experiment runtime',torch.__version__,device,hardware)
dump(out/'runtime.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':hardware,
    'numpy':np.__version__,'batch_size':1024,'epoch_cap':30,'patience':6,'seed_list':args.seeds,
    'implementation':'Fresh matched PyTorch fixed/adaptive study, not bit-identical Keras reproduction'})

class Predictor(nn.Module):
    def __init__(self,numeric,graph,cell):
        super().__init__();self.graph=graph
        self.project=nn.Linear(numeric+(16 if graph else 0),32)
        self.rnn=(nn.LSTM if cell=='lstm' else nn.GRU)(32,64,batch_first=True)
        self.dropout=nn.Dropout(.2);self.hidden=nn.Linear(64,64);self.output=nn.Linear(64,1)
        if graph:
            self.q=nn.Linear(64,128);self.k=nn.Linear(64,128);self.v=nn.Linear(64,128)
            self.attention_output=nn.Linear(128,64);self.norm=nn.LayerNorm(64,eps=1e-6)
    def forward(self,x,g):
        if self.graph:x=torch.cat([x,g],dim=-1)
        x=self.dropout(torch.relu(self.project(x)))
        # Explicit input dropout; no claim of matching Keras recurrent masks.
        x=self.dropout(x);h,_=self.rnn(x)
        if self.graph:
            n,l,_=h.shape
            q=self.q(h).reshape(n,l,4,32).transpose(1,2)
            k=self.k(h).reshape(n,l,4,32).transpose(1,2)
            v=self.v(h).reshape(n,l,4,32).transpose(1,2)
            a=torch.softmax(q@k.transpose(-2,-1)/(32**.5),dim=-1)
            z=(self.dropout(a)@v).transpose(1,2).reshape(n,l,128)
            h=self.norm(h+self.attention_output(z)).mean(1)
        else:h=h[:,-1]
        return self.output(torch.relu(self.hidden(self.dropout(h)))).squeeze(-1)

def fit_predict(numeric,graph,targets,seed,label,folder):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Predictor(numeric.shape[-1],label.startswith('encoder'),'gru' if label=='gru' else 'lstm').to(device)
    x=torch.as_tensor(numeric,device=device);g=torch.as_tensor(graph,device=device)
    target=torch.as_tensor(targets.astype(np.float32),device=device)
    training=torch.as_tensor(np.flatnonzero(tr),device=device);valid=torch.as_tensor(np.flatnonzero(va),device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    best=float('inf');best_state=None;stale=0;history=[];started=time.time()
    def objective(pred,truth):
        err=truth-pred
        loss=torch.maximum(.65*err,-.35*err).mean()
        # Dense and input recurrent matrices receive the stated L2 penalty.
        return loss+1e-5*sum((p**2).sum() for name,p in model.named_parameters()
            if name in ['project.weight','rnn.weight_ih_l0','hidden.weight'])
    for epoch in range(30):
        model.train();order=training[torch.randperm(len(training),device=device)]
        total=0.
        for batch in order.split(1024):
            optimizer.zero_grad(set_to_none=True);pred=model(x[batch],g[batch])
            loss=objective(pred,target[batch]);loss.backward();optimizer.step();total+=float(loss.detach())*len(batch)
        model.eval();vl=0.
        with torch.no_grad():
            for batch in valid.split(8192):vl+=float(objective(model(x[batch],g[batch]),target[batch]))*len(batch)
        vl/=len(valid);history.append({'epoch':epoch+1,'training_loss':total/len(order),'validation_loss':vl})
        if vl<best-1e-8:best=vl;stale=0;best_state=copy.deepcopy(model.state_dict())
        else:stale+=1
        if epoch==0 or (epoch+1)%5==0:log(folder.name,label,'epoch',epoch+1,'val',round(vl,6),'seconds',round(time.time()-started,1))
        if stale>=6:break
    model.load_state_dict(best_state);model.eval();parts=[]
    with torch.no_grad():
        for mask in [va,te]:
            ix=torch.as_tensor(np.flatnonzero(mask),device=device)
            parts.append(np.concatenate([model(x[b],g[b]).cpu().numpy() for b in ix.split(8192)]).astype(float))
    torch.save(best_state,folder/f'{label}.pt');dump(folder/f'{label}_history.json',history)
    del model,x,g,target,training,valid,optimizer,best_state;torch.cuda.empty_cache();gc.collect()
    return parts

import stacking
controller=joblib.load(out/'selection.joblib');old=np.load(base/'x_num.npy',mmap_mode='r')
graph_raw=np.load(base/'x_graph.npy',mmap_mode='r');customers=np.load(base/'customer.npy')
for arm in args.arms:
    for seed in args.seeds:
        folder=out/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metrics.json').exists():log('Already done',arm,seed);continue
        if arm=='adaptive':num=controller.transform(raw,profile_rows)
        else:
            num=np.array(old,dtype=np.float32)
            with threadpool_limits(limits=4):
                lab=KMeans(4,n_init=10,random_state=seed).fit_predict(StandardScaler().fit_transform(profiles))
            num[:,:,16:20]=np.eye(4,dtype=np.float32)[lab[group]][:,None,:]
        length=num.shape[1];graph=np.array(graph_raw[:,-length:,:],np.float32)
        scaler=StandardScaler().fit(num[tr].reshape(-1,num.shape[-1]))
        scaled=scaler.transform(num.reshape(-1,num.shape[-1])).reshape(num.shape).astype(np.float32)
        cap=float(np.quantile(y[tr],.99));direct=np.log1p(np.minimum(y,cap))
        # Anchor and naive mean remain available from observed gaps even if the
        # selector excludes a gap variable from the learned numerical inputs.
        gap=np.asarray(raw[:,-length:,names.index('Days_Since_Last_Purchase')])
        anchor=np.median(gap,axis=1);mean=gap.mean(1)
        target={'direct':direct,'residual':direct-np.log1p(anchor)}
        val_pool={'mean_gap':mean[va]};test_pool={'mean_gap':mean[te]}
        with threadpool_limits(limits=4):
            gb=HistGradientBoostingRegressor(loss='quantile',quantile=.65,max_iter=300,max_leaf_nodes=31,
                min_samples_leaf=25,learning_rate=.05,l2_regularization=.1,early_stopping=False,random_state=seed)
            flat=num.reshape(len(num),-1);gb.fit(flat[tr],direct[tr])
            val_pool['gbm']=np.maximum(0,np.expm1(gb.predict(flat[va])))
            test_pool['gbm']=np.maximum(0,np.expm1(gb.predict(flat[te])))
        joblib.dump(gb,folder/'boosting.joblib');deep={}
        for label,parameter in [('lstm','direct'),('gru','direct'),('encoder_direct','direct'),('encoder_residual','residual')]:
            sy=StandardScaler().fit(target[parameter][tr,None]);z=sy.transform(target[parameter][:,None])[:,0]
            pv,pt=fit_predict(scaled,graph,z,seed,label,folder)
            pv=sy.inverse_transform(pv[:,None])[:,0];pt=sy.inverse_transform(pt[:,None])[:,0]
            if parameter=='residual':pv+=np.log1p(anchor[va]);pt+=np.log1p(anchor[te])
            pv=np.maximum(0,np.expm1(pv));pt=np.maximum(0,np.expm1(pt))
            if label.startswith('encoder'):deep[parameter]=(pv,pt)
            else:val_pool[label]=pv;test_pool[label]=pt
        deep['average']=tuple((deep['direct'][i]+deep['residual'][i])/2 for i in [0,1])
        choice=min(deep,key=lambda k:np.abs(y[va]-deep[k][0]).mean())
        val_pool['encoder'],test_pool['encoder']=deep[choice]
        order=['encoder','gbm','lstm','gru','mean_gap'];v=np.column_stack([val_pool[k] for k in order]);t=np.column_stack([test_pool[k] for k in order])
        with threadpool_limits(limits=4):pred,kind,scores=stacking.select_meta_learner(v,y[va],t,groups=customers[va])
        pred=np.maximum(0,pred)
        np.savez_compressed(folder/'predictions.npz',test=pred,validation_members=v,test_members=t)
        dump(folder/'metrics.json',{'arm':arm,'seed':seed,'numeric_shape':list(num.shape),'encoder_choice':choice,
            'combiner':kind,'inner_scores':scores,'test':metrics(y[te],pred),'seconds_note':'See training histories and logs'})
        log('FINISHED',arm,seed,metrics(y[te],pred))
        del num,scaled,graph,flat,gb;gc.collect()
log('All requested arms complete')
