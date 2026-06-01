"""
Liu et al. (2021) 데이터 CRU/RA 분석 v2
NaN/이상치 제거 + 통계 정리
"""

import numpy as np
import os, glob
from scipy.spatial import KDTree
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
import warnings; warnings.filterwarnings('ignore')

BASE_DIR   = os.path.expanduser('~/Downloads/sponge-like rigid/shear')
MAX_CYCLES = 14
BOOTSTRAP_B = 300

sc = lambda X: (X-X.mean(0))/(X.std(0)+1e-10)

def cv_r2(X, y, g, k=5):
    assert len(X)==len(y)==len(g)
    if X.ndim==1: X=X.reshape(-1,1)
    cids=np.unique(g); kf=KFold(n_splits=min(k,len(cids)-1),shuffle=True,random_state=42)
    Xs=sc(X); pred=np.zeros(len(y))
    for tr,te in kf.split(cids):
        tm=np.isin(g,cids[tr]); vm=np.isin(g,cids[te])
        if not vm.any(): continue
        m=LinearRegression().fit(Xs[tm],y[tm]); pred[vm]=m.predict(Xs[vm])
    ss=np.sum((y-y.mean())**2)
    return 1-np.sum((y-pred)**2)/ss if ss>0 else np.nan

def bc_exact(adj):
    N=len(adj); b=np.zeros(N)
    al=[np.where(adj[i]>0)[0] for i in range(N)]
    for s in range(N):
        stk=[]; pred_=[[] for _ in range(N)]
        sig=np.zeros(N); sig[s]=1.0
        dist=np.full(N,-1); dist[s]=0; q=deque([s])
        while q:
            v=q.popleft(); stk.append(v)
            for w in al[v]:
                if dist[w]<0: q.append(w); dist[w]=dist[v]+1
                if dist[w]==dist[v]+1: sig[w]+=sig[v]; pred_[w].append(v)
        delta=np.zeros(N)
        while stk:
            w=stk.pop()
            for v in pred_[w]:
                if sig[w]>0: delta[v]+=sig[v]/sig[w]*(1+delta[w])
            if w!=s: b[w]+=delta[w]
    dn=(N-1)*(N-2); return b/dn if dn>0 else b

def load_run(run_dir, max_cycles=MAX_CYCLES):
    fwd=os.path.join(run_dir,'forward')
    if not os.path.isdir(fwd): return None
    pdfiles=sorted(glob.glob(os.path.join(fwd,'*ParticleData.dlm')))
    if not pdfiles: return None
    if max_cycles: pdfiles=pdfiles[:max_cycles]
    Os,Rs,Ys,gs=[],[],[],[]; ci=0
    for pf in pdfiles:
        prefix=pf.replace('ParticleData.dlm','')
        bf=prefix+'BinaryAdjacencyMatrix.dlm'
        nf=prefix+'NormWeightedAdjacencyMatrix.dlm'
        if not all(os.path.exists(f) for f in [bf,nf]): continue
        try:
            c=np.loadtxt(pf,delimiter=',')
            ab=np.loadtxt(bf,delimiter=',')
            an=np.loadtxt(nf,delimiter=',')
        except: continue
        N=len(c)
        x_,y_,r_=c[:,1],c[:,2],c[:,3]; L=x_.max()-x_.min()
        if L==0: continue
        tree=KDTree(c[:,1:3])
        ld=np.array([len(tree.query_ball_point(c[i,1:3],r=r_[i]*3))-1 for i in range(N)])
        nn=tree.query(c[:,1:3],k=2)[0][:,1]
        O_i=np.column_stack([x_/L,y_/L,r_/r_.mean(),ld,nn])
        ab_sym=((ab+ab.T)>0).astype(float)
        an_sym=(an+an.T)/2
        al=[np.where(ab_sym[i]>0)[0] for i in range(N)]
        deg=ab_sym.sum(1)
        clust=np.zeros(N)
        for i in range(N):
            ni=al[i]
            if len(ni)>=2:
                tri=sum(1 for j in ni for k_ in ni if k_>j and ab_sym[j,k_]>0)
                clust[i]=2*tri/(len(ni)*(len(ni)-1))
        a2=(ab_sym@ab_sym>0).astype(float); np.fill_diagonal(a2,0)
        two=(a2-ab_sym).clip(0).sum(1)
        b_=bc_exact(ab_sym)
        ndv=np.array([deg[al[i]].std() if len(al[i])>0 else 0 for i in range(N)])
        R_i=np.column_stack([deg,clust,two,b_,ndv])
        Y_i=an_sym.sum(1)
        Os.append(O_i); Rs.append(R_i); Ys.append(Y_i)
        gs.extend([ci]*N); ci+=1
    if not Os: return None
    return np.vstack(Os),np.vstack(Rs),np.concatenate(Ys),np.array(gs),ci,N

if __name__=='__main__':
    print("="*60)
    print("Liu et al. (2021) CRU/RA 분석 v2")
    print("="*60)

    run_dirs=sorted([d for d in glob.glob(os.path.join(BASE_DIR,'*')) if os.path.isdir(d)])
    print(f"Run 폴더 {len(run_dirs)}개\n")

    raw=[]
    for rd in run_dirs:
        rname=os.path.basename(rd)
        res=load_run(rd)
        if res is None: continue
        O,R,Y,g,nc,N=res
        if nc<3: continue
        r2O=cv_r2(O,Y,g); r2R=cv_r2(R,Y,g)
        r2deg=cv_r2(O,R[:,0],g); CRU=1-r2deg; RA=r2R-r2O
        raw.append({'run':rname,'r2O':r2O,'r2R':r2R,'CRU':CRU,'RA':RA})

    # NaN 및 이상치 제거 (CRU>1 또는 NaN)
    valid=[r for r in raw if np.isfinite(r['CRU']) and np.isfinite(r['RA'])
           and 0<=r['CRU']<=1.0 and r['r2O']>-0.5]
    excl =[r for r in raw if r not in valid]

    print(f"유효 run: {len(valid)}개  |  제외: {len(excl)}개 {[r['run'] for r in excl]}")
    print(f"{'Run':>4} {'r2O':>6} {'r2R':>6} {'CRU':>6} {'RA':>7}")
    print("-"*38)
    for r in valid:
        print(f"  {r['run']:>3}  {r['r2O']:6.3f}  {r['r2R']:6.3f}  {r['CRU']:6.3f}  {r['RA']:+7.3f}")

    r2Os=np.array([r['r2O'] for r in valid])
    r2Rs=np.array([r['r2R'] for r in valid])
    CRUs=np.array([r['CRU'] for r in valid])
    RAs =np.array([r['RA']  for r in valid])

    print(f"\n{'='*60}")
    print(f"집계 (N={len(valid)} runs)")
    print(f"  r2O:  {r2Os.mean():.3f} ± {r2Os.std():.3f}  [{r2Os.min():.3f}, {r2Os.max():.3f}]")
    print(f"  r2R:  {r2Rs.mean():.3f} ± {r2Rs.std():.3f}  [{r2Rs.min():.3f}, {r2Rs.max():.3f}]")
    print(f"  CRU:  {CRUs.mean():.3f} ± {CRUs.std():.3f}  [{CRUs.min():.3f}, {CRUs.max():.3f}]")
    print(f"  RA:   {RAs.mean():+.3f} ± {RAs.std():.3f}  [{RAs.min():+.3f}, {RAs.max():+.3f}]")

    rho,ps=spearmanr(CRUs,RAs); r,pp=pearsonr(CRUs,RAs)
    ra_pos=(RAs>0).sum()
    print(f"\nCRU-RA 관계:")
    print(f"  Pearson r  = {r:+.3f}  (p={pp:.4f})")
    print(f"  Spearman ρ = {rho:+.3f}  (p={ps:.4f})")
    print(f"  RA > 0:  {ra_pos}/{len(valid)} runs")

    print(f"\n{'='*60}")
    print("논문(Kollmer 4개) vs Liu(shear):")
    print(f"  CRU:  Kollmer 0.56–0.99  |  Liu {CRUs.min():.2f}–{CRUs.max():.2f}")
    print(f"  RA>0: Kollmer 4/4        |  Liu {ra_pos}/{len(valid)}")
    print(f"  r2O:  Kollmer 0.03–0.49  |  Liu {r2Os.min():.3f}–{r2Os.max():.3f}")
    print(f"  r2R:  Kollmer 0.61–0.73  |  Liu {r2Rs.min():.3f}–{r2Rs.max():.3f}")

    if ra_pos==len(valid):
        print(f"\n  ✓ 전체 {len(valid)}/{len(valid)} run에서 R > O")
        print("  → 다른 프로토콜(shear)에서도 논문 방향 완전 재현")
    elif ra_pos>len(valid)*0.8:
        print(f"\n  ✓ {ra_pos}/{len(valid)} run에서 R > O — 대체로 일관")
    else:
        print(f"\n  ✗ {ra_pos}/{len(valid)} — 방향 불일치")

    np.save('liu2021_results.npy', valid, allow_pickle=True)
    print("\n결과 저장: liu2021_results.npy")
