"""Compact original vector schematics with editable diagrams.net XML."""
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'svg.fonttype':'none','pdf.fonttype':42})
blue='#176B92';gold='#B66A22';gray='#74828B';ink='#243641';light='#EDF5F8';warm='#FBF3E8'
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=280)
    plt.close(fig)
nodes=[];edges=[]
fig,ax=plt.subplots(figsize=(10.6,4.0));ax.set(xlim=(0,14.6),ylim=(0,5.3));ax.axis('off')
def box(key,x,y,w,h,title,fill=light,color=blue,fs=14):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.12',facecolor=fill,edgecolor=color,lw=1.2))
    ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,color=ink,linespacing=1.35)
    nodes.append((key,x,y,w,h,title,fill,color))
def arrow(src,dst,a,b,via=None,dashed=False):
    style=dict(arrowstyle='-|>',mutation_scale=12,color=gray,lw=1.3,linestyle='--' if dashed else '-')
    if via:
        ax.plot([a[0]]+[q[0] for q in via],[a[1]]+[q[1] for q in via],color=gray,lw=1.3,ls=style['linestyle'])
        ax.add_patch(FancyArrowPatch(via[-1],b,**style))
    else:ax.add_patch(FancyArrowPatch(a,b,**style))
    edges.append((src,dst,a,b,via,dashed))
box('history',.15,3.0,1.8,1.1,'Purchase\nhistory')
box('inputs',2.6,3.0,2.0,1.1,'Select\ninputs')
box('graph',2.6,4.5,2.0,.65,'Basket graph',fs=12)
# Only the graph-temporal member receives graph vectors.
box('encoder',5.4,3.8,3.0,.75,'Graph-temporal',fs=13.5)
box('boosting',5.4,2.8,3.0,.65,'Boosting',fs=13.5)
box('recurrent',5.4,1.85,3.0,.65,'LSTM / GRU',fs=13.5)
box('rule',5.4,.9,3.0,.65,'History rule',fs=13.5)
box('combine',9.25,2.8,2.1,1.25,'Combine\npredictions',fill=warm,color=gold)
box('forecast',12.15,2.8,2.1,1.25,'Next gap\nin days',fill='#EAF2EE',color='#3F705C')
arrow('history','inputs',(1.98,3.55),(2.55,3.55))
arrow('history','graph',(1.05,4.14),(2.55,4.825),via=[(1.05,4.825)])
ax.plot([4.97,4.97],[1.22,4.17],color=gray,lw=1.3)
arrow('inputs','encoder',(4.64,3.55),(5.35,4.17),via=[(4.97,3.55),(4.97,4.17)])
for key,y in [('boosting',3.125),('recurrent',2.175),('rule',1.225)]:arrow('inputs',key,(4.97,y),(5.35,y))
arrow('graph','encoder',(4.64,4.825),(6.9,4.59),via=[(6.9,4.825)])
ax.plot([8.8,8.8],[1.22,4.17],color=gray,lw=1.3)
for key,y in [('encoder',4.175),('boosting',3.125),('recurrent',2.175),('rule',1.225)]:
    ax.plot([8.44,8.8],[y,y],color=gray,lw=1.3)
    edges.append((key,'combine',(8.44,y),(9.2,3.425),[(8.8,y),(8.8,3.425)],False))
arrow(None,'combine',(8.8,3.425),(9.2,3.425))
arrow('combine','forecast',(11.39,3.425),(12.10,3.425))
ax.text(5.4,.25,'COMPLEMENTARY PREDICTORS',fontsize=10,color=gray,weight='bold')
ax.text(9.25,4.55,'VALIDATE',fontsize=10,color=gold,weight='bold')
ax.text(12.15,4.55,'PREDICT',fontsize=10,color=gray,weight='bold')
save(fig,'gts_overview')
mxfile=ET.Element('mxfile',host='app.diagrams.net');diagram=ET.SubElement(mxfile,'diagram',name='GTS overview',id='gts-overview')
model=ET.SubElement(diagram,'mxGraphModel',grid='1',page='1',pageWidth='1500',pageHeight='600')
root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
for key,x,y,w,h,title,fill,color in nodes:
    cell=ET.SubElement(root,'mxCell',id=key,value=title.replace('\n','<br>'),vertex='1',parent='1',style=f'rounded=1;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize=18;fillColor={fill};strokeColor={color};fontColor={ink};')
    ET.SubElement(cell,'mxGeometry',x=str(100*x),y=str(100*(5.3-y-h)),width=str(w*100),height=str(h*100),attrib={'as':'geometry'})
for i,(src,dst,a,b,via,dashed) in enumerate(edges):
    attr={'id':f'e{i}','edge':'1','parent':'1','style':'edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=block;strokeColor='+gray+';'}
    if src:attr['source']=src
    if dst:attr['target']=dst
    cell=ET.SubElement(root,'mxCell',**attr);geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
    if not src:ET.SubElement(geo,'mxPoint',x=str(a[0]*100),y=str((5.3-a[1])*100),attrib={'as':'sourcePoint'})
    if via:
        arr=ET.SubElement(geo,'Array',attrib={'as':'points'})
        for x,y in via:ET.SubElement(arr,'mxPoint',x=str(x*100),y=str((5.3-y)*100))
ET.ElementTree(mxfile).write(OUT/'gts_overview.drawio',encoding='utf-8',xml_declaration=True)

fig,ax=plt.subplots(figsize=(10.6,3.0));ax.set(xlim=(0,15.3),ylim=(0,3.6));ax.axis('off')
def block(x,y,w,h,title,fill=light,ec=blue,fs=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.1',facecolor=fill,edgecolor=ec,lw=1.1));ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,color=ink,linespacing=1.25)
def arr(a,b,via=None):
    if via:
        ax.plot([a[0]]+[q[0] for q in via],[a[1]]+[q[1] for q in via],color=gray,lw=1.2);a=via[-1]
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=gray,lw=1.2))
block(.1,2.2,2.05,.65,'Selected\ninputs',fs=11.5)
block(.1,.6,2.05,.65,'Graph\nvectors',fs=11.5)
block(2.8,1.28,1.9,1.1,'Project\n32 units')
block(5.35,1.28,1.9,1.1,'LSTM\n64 units')
block(7.9,1.28,2.2,1.1,'Attention\n4 heads')
block(10.75,1.28,2.0,1.1,'Pool +\nhead',fill=warm,ec=gold)
block(13.4,1.28,1.6,1.1,'Gap',fill='#EAF2EE',ec='#3F705C')
arr((2.18,2.525),(2.76,2.0));arr((2.18,.925),(2.76,1.6))
for a,b in [(4.74,5.31),(7.29,7.86),(10.14,10.71),(12.79,13.36)]:arr((a,1.83),(b,1.83))
arr((6.3,2.41),(9.,2.42),via=[(6.3,3.05),(9.,3.05)])
ax.text(7.6,3.2,'Residual + normalization',fontsize=11,ha='center',color=gray)
ax.text(.1,.02,'Sequence length and numerical input count are selected before final fitting.',fontsize=11,color=gray)
save(fig,'encoder_detail')

fig,axes=plt.subplots(3,1,figsize=(10.7,5.5),layout='constrained')
examples=[('A  Customer-day timing','Earlier observed gaps → five-event context → validation gap → test gap',1),
          ('B  Instacart same-item timing','Product A purchased in orders 1, 3 and 4: its gap includes skipped orders',2),
          ('C  Five-step transaction timing','One forecast origin → five future transaction gaps; no test feedback',3)]
for ax,(title,subtitle,kind) in zip(axes,examples):
    ax.set(xlim=(0,12),ylim=(0,1.5));ax.axis('off');ax.text(.03,1.24,title,fontsize=13,weight='bold',color=ink);ax.text(.03,.98,subtitle,fontsize=10.5,color=ink)
    if kind==1:
        ax.plot([.3,11.5],[.45,.45],color=ink)
        for x,label,c in [(1,'train',blue),(2.2,'train',blue),(3.4,'train',blue),(4.6,'train',blue),(5.8,'train',blue),(7.4,'validation',gold),(9.6,'test','#42684C')]:
            ax.scatter(x,.45,s=55,color=c,zorder=3);ax.text(x,.08,label,ha='center',fontsize=10)
    elif kind==2:
        for x,label,products in [(1,'order 1','A, B'),(4,'order 2','B'),(7,'order 3','A, C'),(10,'order 4','A, B')]:
            ax.scatter(x,.5,s=60,color=blue if 'A' in products else gray,zorder=3);ax.text(x,.09,f'{label}: {products}',ha='center',fontsize=10)
        ax.plot([1,4,7,10],[.5]*4,color=ink);ax.text(4,.66,'5 + 7 = 12 days for A',ha='center',fontsize=10,color=gold);ax.text(8.5,.66,'6 days for A',ha='center',fontsize=10,color=gold)
    else:
        ax.axvline(4.2,ymin=.1,ymax=.5,color=gold,ls='--');ax.text(2,.15,'Observed history',ha='center',fontsize=10.5)
        for j,x in enumerate([5.2,6.5,7.8,9.1,10.4]):
            ax.scatter(x,.48,s=55,color='#42684C');ax.text(x,.10,f'gap {j+1}',ha='center',fontsize=10)
        ax.plot([.3,11.3],[.48,.48],color=ink,lw=1);ax.text(4.2,.67,'origin',ha='center',fontsize=10,color=gold)
save(fig,'target_protocols')
print('Saved overview (PDF/SVG/PNG/drawio), encoder detail and target schematics.')
