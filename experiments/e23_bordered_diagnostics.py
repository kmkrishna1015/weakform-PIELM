"""Section 4.3 support: diagnostics of the bordered one-shot strong pencil (spurious-eigenvalue check, fixed-point residual, 50-digit solve of the same matrices) and the Brent bracket failure of the iterated strong form at 10 modes. Place in experiments/ and run from there."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
import sys; sys.path.insert(0,'.')
from _common import *
from scipy.linalg import eig, svdvals
from pielm.pointmass import PointMassBeam, rA, Lb, EI
CASES = {
 "H1": lambda: PointMassBeam("C","F",None,0,0,m_tip=1.0*rA*Lb),
 "H2": lambda: PointMassBeam("S","S",0.3*Lb,1.0*rA*Lb),
 "H3": lambda: PointMassBeam("C","F",0.6*Lb,2.0*rA*Lb),
 "H4": lambda: PointMassBeam("C","F",0.4*Lb,1.0*rA*Lb,k=100*EI/Lb**3)}
def build(b,n=23,N=5000):
    Bs,G,S4,tot,Nb=b._strong_matrices(n,N)
    def lam_rows(lam):
        rows=[]
        if b.x0 is not None:
            r=np.zeros(tot); r[:Nb]=-EI*Bs[0].d([Bs[0].L],3)[0]; r[Nb:]=EI*Bs[1].d([0.0],3)[0]
            rows.append(r-(b.m*lam-b.k)*b._phi0(Bs,tot,Nb,"x0"))
        if b.mt and b.bcR=="F":
            r=np.zeros(tot); B=Bs[-1]; r[(len(Bs)-1)*Nb:]=EI*B.d([B.L],3)[0]+b.mt*lam*B.d([B.L],0)[0]; rows.append(r)
        return np.vstack(rows)
    fixed=np.vstack(b._ess(Bs,Nb,tot,True)); Zf,_=null_space_svd(fixed)
    c0=lam_rows(0.0)@Zf; c1=(lam_rows(0.0)-lam_rows(1.0))@Zf
    A=Zf.T@S4@Zf; M=Zf.T@G@Zf; nc=c0.shape[0]
    Kb=np.block([[A,c0.T],[c0,np.zeros((nc,nc))]]); Mb=np.block([[M,c1.T],[c1,np.zeros((nc,nc))]])
    return Bs,G,S4,tot,Nb,Kb,Mb,A,M,c0,c1
for key,mk in CASES.items():
    b=mk(); ex=b.exact()
    Bs,G,S4,tot,Nb,Kb,Mb,A,M,c0,c1=build(b)
    lam=eig(Kb,Mb,right=False)
    fin=np.isfinite(lam); re=lam[fin]
    ok=np.abs(re.imag)<=1e-8*np.abs(re.real)+1e-300
    real_pos=np.sort(re.real[ok&(re.real>1e-3)])
    om=np.sqrt(real_pos[:5])
    # what is in the real positive list below / around the 5 lowest: show first 8 and count
    print(key,'size',Kb.shape,'n_inf/nonfinite',int((~fin).sum()),'n_real_pos',len(real_pos),'complex finite',int((~ok).sum()))
    print('  first 8 omega/ex[:5]:',np.round(np.sqrt(real_pos[:8])[:5]/ex[:5]-1,8).tolist(), ' extra:',np.sqrt(real_pos[5:8]).round(1).tolist())
    # consistency: residual of fixed-point equation h(lam)=mu_j(lam)-lam at bordered lam
    Bs2,G2,S42,tot2,Nb2=Bs,G,S4,tot,Nb
    hs=[]
    for j in range(5):
        l=real_pos[j]
        mu=b._strong_pass(Bs,G,S4,tot,Nb,l)
        # find the branch eigenvalue closest to l
        hs.append(float(np.min(np.abs(mu-l))/l))
    print('  fixed-point residual min|mu-l|/l  (should be ~0 if bordered==iteration):',['%.1e'%h for h in hs])
    # scales and conditioning
    print('  ||A||=%.2e ||M||=%.2e ||C0||=%.2e ||C1||=%.2e  cond(M)=%.1e'%(np.linalg.norm(A,2),np.linalg.norm(M,2),np.linalg.norm(c0,2),np.linalg.norm(c1,2),np.linalg.cond(M)))
    # asymmetry of A
    print('  asym(A)=%.1e'%(np.linalg.norm(A-A.T)/np.linalg.norm(A)))
    # iterated reference
    r=b.strong(23,N=5000,mode='brent'); print('  iterated rel err first5 max: %.1e'%(np.max(np.abs(r.omega/ex[:5]-1))))

print("=== 50-digit solve of the SAME double-precision bordered matrices ===")
import mpmath as mp
mp.mp.dps=50
for key in ("H1","H2","H4"):
    b=CASES[key](); ex=b.exact()
    Bs,G,S4,tot,Nb,Kb,Mb,A,M,c0,c1=build(b)
    Kmp=mp.matrix(Kb.tolist()); Mmp=mp.matrix(Mb.tolist())
    T=mp.inverse(Mmp)*Kmp
    E=mp.eig(T,left=False,right=False)
    lam=[complex(e) for e in E]
    lam=np.array(lam)
    re=lam[(np.abs(lam.imag)<=1e-8*np.abs(lam.real)+1e-300)&(lam.real>1e-3)].real
    re=np.sort(re); om=np.sqrt(re[:5])
    print(key,'mp-solve rel err of first 5 omega:',['%.1e'%x for x in (om/ex[:5]-1)])

# ---- iterated strong form: where does Brent fail? ----
import traceback
b=PointMassBeam("S","S",0.3*Lb,1.0*rA*Lb)
Bs,G,S4,tot,Nb=b._strong_matrices(23,5000)
lam=b._strong_pass(Bs,G,S4,tot,Nb,0.0)
print('real eigenvalues returned by one constrained pass at lam*=0:',len(lam),'(space dim ~ %d)'%(tot-6))
for nm in (5,8,9,10):
    b.nmodes=nm
    try:
        r=b.strong(23,N=5000,mode='brent'); print(nm,'ok solves',r.info['solves'])
    except Exception as e:
        print(nm,'FAILED:',type(e).__name__,str(e)[:120])
