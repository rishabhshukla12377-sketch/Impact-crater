import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
ZS = 1.0
NU, NU_SHOCK = 1e-2, 3.0
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
SEAM = [k*NL - 1 for k in range(1, NPX+1)]   # all 8 per-proc corner rows/cols, incl. true domain edge

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
    # the (row,col) intersections of proc corners carry a stale slice-I/O value (~=t); blank + interpolate
    mask = np.zeros_like(g, dtype=bool)
    for r in SEAM:
        for c in SEAM:
            mask[max(r-1,0):r+2, max(c-1,0):c+2] = True
    gg = g.copy()
    for r in SEAM: gg[r,:] = 0.5*(g[r-1,:]+np.roll(g,-1,axis=0)[r,:] if r+1<NG else g[r-1,:])
    for c in SEAM: gg[:,c] = 0.5*(g[:,c-1]+np.roll(g,-1,axis=1)[:,c] if c+1<NG else g[:,c-1])
    return t, gg

nf = nframes("lnrho")
t0, pp0 = read_frame("pp", 0)
pamb = np.median(pp0[zc < ZS])

ts, rng = [], []
for fr in range(nf):
    t, pp = read_frame("pp", fr)
    ppt = pp.copy(); ppt[zc >= ZS] = -np.inf
    jz, jx = np.unravel_index(np.argmax(ppt), ppt.shape)
    rng.append(np.hypot(xc[jx], zc[jz]-ZS)); ts.append(t)
ts, rng = np.array(ts), np.array(rng)
m = (ts>0)&(rng>0.01)
Re = rng[m]*(rng[m]/ts[m])/NU

# domain-max shock indicator vs time (excluding the known corrupted corner cells)
good = np.ones((NG,NG), dtype=bool)
for r in SEAM:
    good[max(r-2,0):r+3, :] = False
for c in SEAM:
    good[:, max(c-2,0):c+3] = False
shock_max_t = []
for fr in range(nf):
    _, shock = read_frame("shock", fr)
    shock_max_t.append(shock[good].max())
shock_max_t = np.array(shock_max_t)
nu_eff_max = NU + NU_SHOCK*shock_max_t

plt.rcParams.update({"font.size":13,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False, "axes.spines.right":False})
fig, ax = plt.subplots(1,2, figsize=(13,5.3), constrained_layout=True)

ax[0].axhline(1, color="grey", ls="--", lw=1.2, label="Re = 1\n(viscous $\\approx$ inertial)")
ax[0].plot(ts[m], Re, color="C3", lw=2.4)
ax[0].set_yscale("log")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("blast Reynolds number\n$Re = R\\cdot(R/t)/\\nu$")
ax[0].set_title("Inertia beats viscosity by 2-4 orders\nof magnitude on the shock scale")
ax[0].legend(fontsize=9, loc="lower right")

ax[1].plot(ts, nu_eff_max, color="C0", lw=2.3, label="worst-case local viscosity\n$\\nu+\\nu_{shock}\\times\\max($shock$)$")
ax[1].axhline(NU, color="grey", ls=":", lw=1.4, label=f"background $\\nu$ = {NU}")
ax[1].set_yscale("log")
ax[1].set_ylim(NU*0.5, NU*20)
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("effective viscosity [code units]")
ax[1].set_title("Even at its most active point,\nartificial shock viscosity barely exceeds $\\nu$")
ax[1].legend(fontsize=9, loc="upper right")

fig.suptitle("Why Sedov still applies in a \"viscous\" run: viscosity (physical + numerical) stays\nnegligible compared to inertia everywhere, all the time ($Re\\gg1$)", fontsize=13)
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_why_sedov.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print("Re range:", Re.min(), Re.max())
print("shock_max_t range:", shock_max_t.min(), shock_max_t.max())
print("nu_eff_max range:", nu_eff_max.min(), nu_eff_max.max(), " vs background", NU)
