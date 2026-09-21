import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
ZS = 1.0
GAMMA = 5.0/3.0; GRAV = 9.8e-3
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
dx = xc[1]-xc[0]; dz = zc[1]-zc[0]; dA = dx*dz
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]
E_DEP, F_KIN = 0.30, 0.50
K = 2

def nframes(fld): return os.path.getsize(f"{D}/proc0/slice_{fld}.xz")//FRAME
def read_frame(fld, fr):
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

nf = nframes("lnrho")
_, lnr0 = read_frame("lnrho",0); _, pp0 = read_frame("pp",0)
rho0 = np.exp(lnr0)
Eint_t0 = (pp0/(GAMMA-1)).sum()*dA
Egrv_t0 = (rho0*GRAV*zc[:,None]).sum()*dA
Eint_pre = Eint_t0 - (1-F_KIN)*E_DEP
pamb = np.median(pp0[zc < ZS])

ts=[]; Esum=[]; rng=[]; dpk=[]
for fr in range(nf):
    t, lnr = read_frame("lnrho", fr)
    _, ux = read_frame("uu1", fr); _, uz = read_frame("uu3", fr); _, pp = read_frame("pp", fr)
    rho = np.exp(lnr); spd2 = ux*ux+uz*uz
    ke = 0.5*(rho*spd2).sum()*dA
    eint = (pp/(GAMMA-1)).sum()*dA - Eint_pre
    egrv = (rho*GRAV*zc[:,None]).sum()*dA - Egrv_t0
    Esum.append(ke+eint+egrv)
    ppt = pp.copy(); ppt[zc >= ZS] = -np.inf
    jz, jx = np.unravel_index(np.argmax(ppt), ppt.shape)
    dpk.append(pp[jz,jx]-pamb); rng.append(np.hypot(xc[jx], zc[jz]-ZS))
    ts.append(t)

ts=np.array(ts); Esum=np.array(Esum); rng=np.array(rng); dpk=np.array(dpk)
frac_E = 100*Esum/Esum[0]

m = (rng>0.01)&(dpk>0)
tm, logr, logdp = ts[m], np.log(rng[m]), np.log(dpk[m])
local_slope = -np.gradient(logdp, logr)     # -dlnP/dlnr ; Sedov(2D) = 2

plt.rcParams.update({"font.size":13,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False, "axes.spines.right":False})
fig, ax = plt.subplots(2,1, figsize=(8.5,8), sharex=True, constrained_layout=True)

ax[0].axhline(100, color="grey", ls="--", lw=1.3, label="100% (perfectly conserved)")
ax[0].plot(ts, frac_E, color="C3", lw=2.5)
ax[0].set_ylabel("energy remaining\nin the box [%]")
ax[0].set_ylim(45,110)
ax[0].legend(fontsize=9, loc="lower left")

ax[1].axhspan(1.8, 2.2, color="green", alpha=0.15, label="matches 2-D Sedov ($\\pm$10%)")
ax[1].axhline(2.0, color="k", ls="--", lw=1.3)
ax[1].plot(tm, local_slope, color="C0", lw=2.2)
ax[1].set_ylim(0.5,3.0)
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("blast decay exponent\n$-d\\ln\\Delta p/d\\ln r$")
ax[1].legend(fontsize=9, loc="upper right")

fig.suptitle("As energy leaks out of the box, the blast falls off the Sedov curve", fontsize=14)
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_sedov_energy_link.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
