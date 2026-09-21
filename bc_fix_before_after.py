import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
GAMMA = 5.0/3.0; GRAV = 9.8e-3
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
dx = xc[1]-xc[0]; dz = zc[1]-zc[0]; dA = dx*dz
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]
E_DEP, F_KIN = 0.30, 0.50

def nframes(D, fld): return os.path.getsize(f"{D}/proc0/slice_{fld}.xz")//FRAME
def read_frame(D, fld, fr):
    g = np.empty((NG,NG)); t=0.0
    for p in range(NPX*NPZ):
        ipx, ipz = p % NPX, p // NPX
        with open(f"{D}/proc{p}/slice_{fld}.xz","rb") as f:
            f.seek(fr*FRAME+12)
            sl = np.fromfile(f, np.float64, NL*NL).reshape(NL,NL)
            f.seek(fr*FRAME+12+NL*NL*8+8)
            t = np.fromfile(f, np.float64, 1)[0]
        g[ipz*NL:(ipz+1)*NL, ipx*NL:(ipx+1)*NL] = sl
    for c in SEAM_COLS:
        g[:, c] = 0.5*(g[:, c-1] + g[:, c+1])
    return t, g

def compute(D):
    nf = nframes(D, "lnrho")
    _, lnr0 = read_frame(D, "lnrho",0); _, pp0 = read_frame(D, "pp",0)
    rho0 = np.exp(lnr0)
    Eint_t0 = (pp0/(GAMMA-1)).sum()*dA
    Egrv_t0 = (rho0*GRAV*zc[:,None]).sum()*dA
    Eint_pre = Eint_t0 - (1-F_KIN)*E_DEP
    ts=[]; Esum=[]; MASS=[]
    for fr in range(nf):
        t, lnr = read_frame(D, "lnrho", fr)
        _, ux = read_frame(D, "uu1", fr); _, uz = read_frame(D, "uu3", fr); _, pp = read_frame(D, "pp", fr)
        rho = np.exp(lnr); spd2 = ux*ux+uz*uz
        ke = 0.5*(rho*spd2).sum()*dA
        eint = (pp/(GAMMA-1)).sum()*dA - Eint_pre
        egrv = (rho*GRAV*zc[:,None]).sum()*dA - Egrv_t0
        Esum.append(ke+eint+egrv); MASS.append(rho.sum()*dA); ts.append(t)
    return np.array(ts), np.array(Esum), np.array(MASS)

t_old, E_old, M_old = compute(f"{BASE}/leaky_bc_original_run/data")
t_new, E_new, M_new = compute(f"{BASE}/data")

plt.rcParams.update({"font.size":13,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False, "axes.spines.right":False})
fig, ax = plt.subplots(1,2, figsize=(13,5.5), constrained_layout=True)

ax[0].axhline(100, color="grey", ls="--", lw=1.3, label="100% (perfectly conserved)")
ax[0].plot(t_old, 100*E_old/E_old[0], color="C3", lw=2.4, label="before fix (bcz uz = 's', open)")
ax[0].plot(t_new, 100*E_new/E_new[0], color="C0", lw=2.4, label="after fix (bcz uz = 'a', closed)")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("energy remaining [%]")
ax[0].set_title(f"Energy budget\nbefore: {100*(1-E_old[-1]/E_old[0]):.0f}% lost   after: {100*(E_new[-1]/E_new[0]-1):+.1f}% drift")
ax[0].legend(fontsize=9, loc="lower left")

ax[1].axhline(100, color="grey", ls="--", lw=1.3, label="100% (perfectly conserved)")
ax[1].plot(t_old, 100*M_old/M_old[0], color="C3", lw=2.4, label="before fix")
ax[1].plot(t_new, 100*M_new/M_new[0], color="C0", lw=2.4, label="after fix")
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("mass remaining [%]")
ax[1].set_title(f"Mass budget\nbefore: {100*(1-M_old[-1]/M_old[0]):.2f}% lost   after: {100*(1-M_new[-1]/M_new[0]):+.3f}% lost")
ax[1].legend(fontsize=9, loc="lower left")

fig.suptitle("bcz fix: closing uz (was 's', now 'a') removes the advective leak", fontsize=13.5)
out = f"{BASE}/analysis/figures/myblast14_bc_fix_before_after.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print(f"OLD: E {E_old[0]:.4f}->{E_old[-1]:.4f} ({100*(1-E_old[-1]/E_old[0]):.1f}% lost)   M {M_old[0]:.5f}->{M_old[-1]:.5f} ({100*(1-M_old[-1]/M_old[0]):.3f}% lost)")
print(f"NEW: E {E_new[0]:.4f}->{E_new[-1]:.4f} ({100*(E_new[-1]/E_new[0]-1):+.1f}% change)   M {M_new[0]:.5f}->{M_new[-1]:.5f} ({100*(1-M_new[-1]/M_new[0]):+.4f}% lost)")
