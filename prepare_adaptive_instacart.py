"""Recover the complete 40-variable candidate pool on the archived origins.

The experiment uses the same full cohort and fixed five-event maximum context;
the selector is free to use any smaller context. No transaction is simulated.
"""
from pathlib import Path
import argparse,gc,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def candidate_matrix(df):
    df=df.sort_values(['cid','event_day','order_number'],kind='stable').reset_index(drop=True)
    ids,start,size=np.unique(df.cid,return_index=True,return_counts=True)
    rank=np.repeat(np.arange(len(ids)),size);left0=np.repeat(start,size)
    ix=np.arange(len(df));pos=ix-left0;days=df.event_day.to_numpy()
    gaps=np.zeros(len(df));gaps[1:]=np.maximum(0,np.diff(days));gaps[pos==0]=0
    spend=df.sale_price_total.to_numpy()
    prefix=lambda a:np.r_[0.,np.cumsum(a,dtype=np.float64)]
    pg,pg2,ps,ps2=map(prefix,[gaps,gaps**2,spend,spend**2])
    f={'Days_Since_Last_Purchase':gaps,'Transaction_Count':pos+1};dynamic=[]
    for lag in [2,3,4,5]:
        v=np.zeros(len(df));valid=pos>=lag-1;v[valid]=gaps[ix[valid]-lag+1]
        name=f'Gap{lag}Last';f[name]=v;dynamic.append(name)
    for w in [3,5,10,20]:
        lo=np.maximum(left0,ix-w+1);n=ix-lo+1
        for label,p1,p2 in [('Gap',pg,pg2),('Spend',ps,ps2)]:
            total=p1[ix+1]-p1[lo];sq=p2[ix+1]-p2[lo]
            f[f'Avg{label}_w{w}']=total/n
            sd=np.sqrt(np.maximum(0,sq-total**2/n)/np.maximum(1,n-1));sd[n==1]=0
            f[f'{label}Std_w{w}']=sd
        dynamic += [f'AvgGap_w{w}',f'GapStd_w{w}',f'AvgSpend_w{w}',f'SpendStd_w{w}']
    n=pos+1;total=pg[ix+1]-pg[left0];sq=pg2[ix+1]-pg2[left0]
    f['AvgGap']=total/n;f['GapStd']=np.sqrt(np.maximum(0,sq-total**2/n)/np.maximum(1,n-1));f['GapStd'][n==1]=0
    f['CumSpend']=ps[ix+1]-ps[left0];f['AvgSpend']=f['CumSpend']/n;f['LastAmount']=spend
    f['RelativeLastAmount']=np.divide(spend,f['AvgSpend'],out=np.zeros(len(df)),where=f['AvgSpend']!=0)
    f['CustomerTenure']=days-days[left0]
    stride=int(days.max()-days.min()+100);encoded=rank.astype(np.int64)*stride+days-days.min()
    for w in [7,30]:
        lo=np.maximum(left0,np.searchsorted(encoded,encoded-w,side='left'))
        f[f'Count_{w}d']=ix-lo;f[f'Spend_{w}d']=ps[ix]-ps[lo]
        dynamic += [f'Count_{w}d',f'Spend_{w}d']
    f['sin_dow']=np.sin(2*np.pi*df.dow.to_numpy()/7);f['cos_dow']=np.cos(2*np.pi*df.dow.to_numpy()/7)
    f['sin_month']=np.zeros(len(df));f['cos_month']=np.zeros(len(df))
    for k in ['line_items','unique_products_in_event','quantity']:f[k]=df[k].to_numpy()
    base=['Days_Since_Last_Purchase','Transaction_Count','AvgGap','GapStd','CumSpend','AvgSpend','LastAmount',
          'RelativeLastAmount','CustomerTenure','line_items','unique_products_in_event','quantity',
          'sin_dow','cos_dow','sin_month','cos_month']
    names=base+dynamic
    return df,np.column_stack([f[k] for k in names]).astype(np.float32),names,gaps

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--benchmark-root',type=Path,default=ROOT/'work')
    p.add_argument('--output',type=Path,default=ROOT/'work/adaptive/instacart')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    orders=pd.read_csv(a.data_root/'orders.csv',dtype={'order_id':'int32','user_id':'int32','order_number':'int16',
        'order_dow':'int8','order_hour_of_day':'int8','days_since_prior_order':'float32'})
    orders.sort_values(['user_id','order_number'],inplace=True)
    orders['relative_day']=orders.days_since_prior_order.fillna(0).groupby(orders.user_id).cumsum().astype(np.int32)
    maxid=int(orders.order_id.max())+1;lut={}
    for col in ['user_id','order_number','order_dow','relative_day']:
        v=np.zeros(maxid,np.int32);v[orders.order_id]=orders[col].to_numpy();lut[col]=v
    del orders;gc.collect()
    op=pd.concat([pd.read_csv(a.data_root/f,usecols=['order_id','product_id'],dtype={'order_id':'int32','product_id':'int32'})
        for f in ['order_products__prior.csv','order_products__train.csv']],ignore_index=True)
    oid=op.order_id.to_numpy();pid=op.product_id.to_numpy();pair=lut['user_id'][oid].astype(np.int64)*60000+pid
    ids,counts=np.unique(pair,return_counts=True);eligible=ids[counts>=21];keep=np.isin(pair,eligible)
    bn=np.bincount(oid,minlength=maxid).astype(np.float32);o=oid[keep]
    df=pd.DataFrame({'cid':pair[keep],'user_id':lut['user_id'][o],'order_id':o,'order_number':lut['order_number'][o],
        'event_day':lut['relative_day'][o],'dow':lut['order_dow'][o],'sale_price_total':bn[o],
        'line_items':bn[o],'unique_products_in_event':bn[o],'quantity':bn[o]})
    del op,oid,pid,pair,ids,counts,eligible,keep,o,bn,lut;gc.collect()
    print('Constructing all candidates for',len(df),'observed pair events',flush=True)
    df,x,names,gaps=candidate_matrix(df)
    archived=a.benchmark_root/'prepared/instacart_pairs';keys=pd.read_parquet(archived/'keys.parquet')
    lookup=pd.Series(np.arange(len(df)),index=pd.MultiIndex.from_frame(df[['cid','order_id']]))
    loc=lookup.loc[pd.MultiIndex.from_arrays([keys.series_id,keys.origin_order_id])].to_numpy()
    win=loc[:,None]+np.arange(-4,1);assert np.all(df.cid.to_numpy()[win]==keys.series_id.to_numpy()[:,None])
    target=np.load(archived/'y.npy');assert np.array_equal(gaps[loc+1],target[:,0])
    numeric=x[win];old=np.load(archived/'x_num.npy',mmap_mode='r')
    old_names=json.loads((archived/'manifest.json').read_text())['feature_names']
    for j,name in enumerate(old_names):
        if not name.startswith('Cluster_'):
            assert np.allclose(numeric[:,:,names.index(name)],old[:,:,j],atol=.001,rtol=1e-5),name
    np.save(a.output/'candidate_windows.npy',numeric)
    (a.output/'candidate_names.json').write_text(json.dumps(names,indent=2))
    (a.output/'preparation_audit.json').write_text(json.dumps({'n_windows':len(numeric),'shape':list(numeric.shape),
        'same_origins_and_targets':True,'all_archived_numerical_candidates_matched':True,'cohort':'All 103855 eligible pairs'},indent=2))
    print('Saved',numeric.shape,'all archived shared features match',flush=True)

if __name__=='__main__':main()
