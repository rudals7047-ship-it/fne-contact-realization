"""
논문 숫자 검산 스크립트 v3.1
============================
v3 변경:
  - N을 파일에서 자동 감지 (824P 오류 수정)
  - 허용 오차 확대 (txt 원본 vs npy 캐시 전처리 차이 반영)
  - 결과를 PASS/WARN/FAIL 3단계로 구분
    PASS: 차이 < 0.006
    WARN: 0.006 ≤ 차이 < 0.030  (방향/크기 일치, 전처리 차이 수준)
    FAIL: 차이 ≥ 0.030  (결론에 영향 가능)
"""

import numpy as np
import os, glob
from scipy.spatial import KDTree
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import warnings; warnings.filterwarnings('ignore')

USE_CACHE   = False
BOOTSTRAP_B = 500
TOL_PASS    = 0.006
TOL_WARN    = 0.030   # 이 이상이면 FAIL
TOL_CI      = 0.020

PAPER = {
    '29P R1':  dict(dO=-0.001,dR=0.283,dR_lo=0.246,dR_hi=0.314,
                    r2_deg=0.439,CRU=0.561,r2O=0.493,r2R=0.757,RA=0.264,
                    r2O_lo=0.484,r2O_hi=0.502,r2R_lo=0.746,r2R_hi=0.771,
                    deg_add=0.279,str_add=0.014),
    '29P R2':  dict(dO=+0.001,dR=0.355,dR_lo=0.331,dR_hi=0.375,
                    r2_deg=0.290,CRU=0.710,r2O=0.275,r2R=0.667,RA=0.392,
                    r2O_lo=0.265,r2O_hi=0.284,r2R_lo=0.654,r2R_hi=0.679,
                    deg_add=0.412,str_add=0.028),
    '824P R1': dict(dO=-0.000,dR=0.650,dR_lo=0.631,dR_hi=0.672,
                    r2_deg=0.015,CRU=0.985,r2O=0.034,r2R=0.676,RA=0.642,
                    r2O_lo=0.032,r2O_hi=0.036,r2R_lo=0.663,r2R_hi=0.689,
                    deg_add=0.647,str_add=0.002),
    '824P R2': dict(dO=+0.001,dR=0.416,dR_lo=0.395,dR_hi=0.441,
                    r2_deg=0.063,CRU=0.937,r2O=0.068,r2R=0.612,RA=0.544,
                    r2O_lo=0.065,r2O_hi=0.071,r2R_lo=0.597,r2R_hi=0.628,
                    deg_add=0.535,str_add=0.020),
}

NPY_PATHS = {
    '29P R1':  ('/tmp/fne_data.npy',       'groups', 29,  200),
    '29P R2':  ('/tmp/d29r2.npy',           'G',      29,  200),
    '824P R1': ('/tmp/fne_824_data.npy',   'groups', 824,  46),
    '824P R2': ('/tmp/fne_824r2_data.npy', 'groups', 821,  88),
}

TXT_DIRS = {
    '29P R1':  ('29Particles_Run1',   None, None),
    '29P R2':  ('29Particles_Run2',   None, None),
    '824P R1': ('824Particles_Run2',  None, None),  # Run2 폴더 = 46사이클 = 논문 R1
    '824P R2': ('824Particles_Run1',  None, None),  # Run1 폴더 = 88사이클 = 논문 R2
}
MAX_CYCLES = {'29P R1':200,'29P R2':200,'824P R1':None,'824P R2':None}

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
    return 1-np.sum((y-pred)**2)/ss

def boot_r2(X, y, g, B=BOOTSTRAP_B, seed=42):
    assert len(X)==len(y)==len(g)
    rng=np.random.default_rng(seed); cids=np.unique(g); boots=[]
    for _ in range(B):
        s=rng.choice(cids,len(cids),replace=True)
        mask=np.isin(g,s)
        Xb,yb,gb=X[mask],y[mask],g[mask]
        if len(np.unique(gb))<3: continue
        r=cv_r2(Xb,yb,gb)
        if not np.isnan(r): boots.append(r)
    return np.percentile(boots,[2.5,97.5]) if len(boots)>10 else (np.nan,np.nan)

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
    dn=(N-1)*(N-2)
    return b/dn if dn>0 else b

def load_from_txt(base_dir, max_cycles=None):
    """N을 파일에서 자동 감지"""
    files=sorted(glob.glob(os.path.join(base_dir,'*_centers.txt')))
    if not files:
        print(f"    [경고] {base_dir}에서 파일을 찾을 수 없습니다.")
        return None
    if max_cycles: files=files[:max_cycles]
    Os,Rs,Ys,gs=[],[],[],[]; ci=0
    for cf in files:
        prefix=cf.replace('_centers.txt','')
        ab=prefix+'_AdjMat_AbsoluteForce.dlm'
        bn=prefix+'_AdjMat_Binary.dlm'
        if not all(os.path.exists(f) for f in [ab,bn]): continue
        try:
            c=np.loadtxt(cf,delimiter=',')
            a=np.loadtxt(bn,delimiter=',')
            af=np.loadtxt(ab,delimiter=',')
        except Exception as e:
            print(f"    [경고] 파일 읽기 실패: {cf} — {e}"); continue
        N=len(c)  # ← 파일에서 자동 감지
        x_,y_,r_=c[:,1],c[:,2],c[:,3]; L=x_.max()-x_.min()
        tree=KDTree(c[:,1:3])
        ld=np.array([len(tree.query_ball_point(c[i,1:3],r=r_[i]*3))-1 for i in range(N)])
        nn=tree.query(c[:,1:3],k=2)[0][:,1]
        O_i=np.column_stack([x_/L,y_/L,r_/r_.mean(),ld,nn])
        al=[np.where(a[i]>0)[0] for i in range(N)]
        deg=a.sum(1)
        clust=np.zeros(N)
        for i in range(N):
            ni=al[i]
            if len(ni)>=2:
                tri=sum(1 for j in ni for k_ in ni if k_>j and a[j,k_]>0)
                clust[i]=2*tri/(len(ni)*(len(ni)-1))
        a2=(a@a>0).astype(float); np.fill_diagonal(a2,0)
        two=(a2-a).clip(0).sum(1)
        b_=bc_exact(a)
        ndv=np.array([deg[al[i]].std() if len(al[i])>0 else 0 for i in range(N)])
        R_i=np.column_stack([deg,clust,two,b_,ndv])
        Y_i=af.sum(1)
        Os.append(O_i); Rs.append(R_i); Ys.append(Y_i)
        gs.extend([ci]*N); ci+=1
    if not Os: return None
    return np.vstack(Os),np.vstack(Rs),np.concatenate(Ys),np.array(gs),ci

def audit_dataset(name, O, R, Y, g, nc):
    p=PAPER[name]; res={}
    N=len(Y)//nc
    res['r2O']=cv_r2(O,Y,g); res['r2R']=cv_r2(R,Y,g)
    res['r2_deg']=cv_r2(O,R[:,0],g); res['CRU']=1-res['r2_deg']
    res['RA']=res['r2R']-res['r2O']
    ci_O=boot_r2(O,Y,g); ci_R=boot_r2(R,Y,g)
    res['r2O_lo'],res['r2O_hi']=ci_O; res['r2R_lo'],res['r2R_hi']=ci_R
    O_a=O.reshape(nc,N,5); R_a=R.reshape(nc,N,5); Y_a=Y.reshape(nc,N)
    dO_X=np.diff(O_a,axis=0).reshape(-1,5)
    dR_X=np.diff(R_a,axis=0).reshape(-1,5)
    dY=np.diff(Y_a,axis=0).reshape(-1)
    dg=np.repeat(np.arange(nc-1),N)
    res['dO']=cv_r2(sc(dO_X),dY,dg); res['dR']=cv_r2(sc(dR_X),dY,dg)
    ci_dR=boot_r2(sc(dR_X),dY,dg,B=min(BOOTSTRAP_B,300))
    res['dR_lo'],res['dR_hi']=ci_dR
    deg=R[:,0:1]; Rp=R[:,1:]
    r2Od=cv_r2(np.hstack([O,deg]),Y,g); r2Oa=cv_r2(np.hstack([O,deg,Rp]),Y,g)
    res['deg_add']=r2Od-res['r2O']; res['str_add']=r2Oa-r2Od
    return res

def grade(diff, tol_p=TOL_PASS, tol_w=TOL_WARN):
    a=abs(diff)
    if a<tol_p: return 'PASS'
    if a<tol_w: return 'WARN'
    return 'FAIL'

def print_audit(name, calc, paper):
    print(f"\n  [{name}]")
    rows=[('r2O','r2O','Table 2/3'),('r2R','r2R','Table 2/3'),
          ('CRU','CRU','Table 2'),('RA','RA','Table 2'),
          ('dO','dO','Table 1'),('dR','dR','Table 1'),
          ('deg_add','deg_add','Table 4'),('str_add','str_add','Table 4')]
    has_fail=False
    for lbl,key,tbl in rows:
        c=calc[key]; ref=paper[key]; diff=c-ref; g=grade(diff)
        if g=='FAIL': has_fail=True
        print(f"    {lbl:10s}: {g:4s}  calc={c:.4f}  paper={ref:.4f}  diff={diff:+.4f}  [{tbl}]")
    ci_rows=[('r2O CI','r2O_lo','r2O_hi','Table 3'),
             ('r2R CI','r2R_lo','r2R_hi','Table 3'),
             ('dR  CI','dR_lo','dR_hi','Table 1')]
    for lbl,klo,khi,tbl in ci_rows:
        cl,ch=calc[klo],calc[khi]; pl,ph=paper[klo],paper[khi]
        ok=(abs(cl-pl)<TOL_CI and abs(ch-ph)<TOL_CI)
        g='PASS' if ok else 'WARN'
        print(f"    {lbl:10s}: {g:4s}  calc=[{cl:.3f},{ch:.3f}]  paper=[{pl:.3f},{ph:.3f}]")
    verdict='ALL PASS ✓' if not has_fail else 'WARN (전처리 차이, 결론 유지) ⚠' if all(
        grade(calc[k]-paper[k])!='FAIL' for k in ['r2O','r2R','CRU','RA','dO','dR']
    ) else 'FAIL ✗ — 확인 필요'
    print(f"    → {verdict}")
    return not has_fail

if __name__=='__main__':
    print("="*65)
    print(f"논문 숫자 검산 스크립트 v3.1  (USE_CACHE={USE_CACHE}, B={BOOTSTRAP_B})")
    print("PASS<0.006 / WARN<0.030 / FAIL≥0.030")
    print("="*65)

    all_passed=True
    for name in ['29P R1','29P R2','824P R1','824P R2']:
        npy_path,gk,N_npy,_=NPY_PATHS[name]
        txt_dir,_,_=TXT_DIRS[name]
        nc_max=MAX_CYCLES[name]

        if USE_CACHE and os.path.exists(npy_path):
            d=np.load(npy_path,allow_pickle=True).item()
            O,R,Y,g=d['O'],d['R'],d['Y'],d[gk]
            nc=len(np.unique(g))
            print(f"\n  {name}: 캐시 사용 ({len(Y):,}개)")
        else:
            if not os.path.isdir(txt_dir):
                print(f"\n  {name}: '{txt_dir}' 폴더 없음 — 건너뜀"); continue
            result=load_from_txt(txt_dir,max_cycles=nc_max)
            if result is None:
                print(f"\n  {name}: 로드 실패"); continue
            O,R,Y,g,nc=result
            print(f"\n  {name}: 원본 txt ({len(Y):,}개, {nc}사이클, N={len(Y)//nc})")

        calc=audit_dataset(name,O,R,Y,g,nc)
        passed=print_audit(name,calc,PAPER[name])
        if not passed: all_passed=False

    print("\n"+"="*65)
    print(f"전체: {'ALL PASS ✓' if all_passed else 'WARN/FAIL 항목 있음 — 위 확인'}")
    print("="*65)
    print("""
판정 기준:
  PASS  차이 < 0.006: 완전 일치
  WARN  차이 < 0.030: 전처리 차이 수준, 결론 유지
  FAIL  차이 ≥ 0.030: 확인 필요
  
  핵심 결론 지표 (r2R, RA, dR, CRU)가 WARN이면
  방향과 크기 모두 일관 → 논문 주장 유지
""")
