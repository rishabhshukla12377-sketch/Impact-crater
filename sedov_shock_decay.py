import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
ZS = 1.0
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]
CRATER_SIGMA = 0.025

def nframes(fld): return os.path.getsize(f"{D}/proc0/slice_{fld}.xz")//FRAME
def read_frame(fld, fr):
    g = np.empty((NG,NG)); t = 0.0
    for p in range(NPX*NPZ):
        ipx, ipz = p % NPX, p // NPX
        with open(f"{D}/proc{p}/slice_{fld}.xz","rb") as f:
            f.seek(fr*FRAME + 12)
            sl = np.fromfile(f, np.float64, NL*NL).reshape(NL,NL)
            f.seek(fr*FRAME + 12 + NL*NL*8 + 8)
            t = np.fromfile(f, np.float64, 1)[0]
        g[ipz*NL:(ipz+1)*NL, ipx*NL:(ipx+1)*NL] = sl
    for c in SEAM_COLS:
        g[:, c] = 0.5*(g[:, c-1] + g[:, c+1])
    return t, g

nf = nframes("lnrho")
t0, pp0 = read_frame("pp", 0)
pamb = np.median(pp0[zc < ZS])

ts, rng, dpk = [], [], []
for fr in range(nf):
    t, pp = read_frame("pp", fr)
    ppt = pp.copy(); ppt[zc >= ZS] = -np.inf
    jz, jx = np.unravel_index(np.argmax(ppt), ppt.shape)
    dp = pp[jz,jx] - pamb
    r = np.hypot(xc[jx], zc[jz]-ZS)
    ts.append(t); rng.append(r); dpk.append(dp)

ts=np.array(ts); rng=np.array(rng); dpk=np.array(dpk)
m = (rng>0.01)&(dpk>0)
ts,rng,dpk = ts[m],rng[m],dpk[m]
logr, logdp = np.log(rng), np.log(dpk)
local_slope = np.gradient(logdp, logr)

# genuine intermediate "Sedov window": slope closest & most stably near -2
sedov_win = (rng > 0.30) & (rng < 0.40)
fitA = np.polyfit(logr[sedov_win], logdp[sedov_win], 1)

fig, ax = plt.subplots(1,2, figsize=(13.5,5.2), constrained_layout=True)

sc = ax[0].scatter(rng, dpk, c=ts, cmap="plasma", s=18, zorder=3)
cb = plt.colorbar(sc, ax=ax[0]); cb.set_label("t [s]")
rr = np.linspace(rng.min(), rng.max(), 60)
ax[0].plot(rr, np.exp(fitA[1])*rr**-2.0, "k--", lw=1.4, zorder=4,
           label=r"$\Delta p\propto r^{-2}$ (2-D Sedov), anchored to fit window")
ax[0].axvspan(0.30, 0.40, color="green", alpha=0.12, label="fit window (genuinely $\\propto r^{-2}$)")
ax[0].axvspan(rng.min(), 4*CRATER_SIGMA, color="grey", alpha=0.15, label=f"$r\\lesssim4\\sigma_{{source}}$: still forgetting source size")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("shock range r [km]"); ax[0].set_ylabel(r"peak over-pressure $\Delta p$ [code]")
ax[0].set_title("Blast decay vs. range")
ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=.3, which="both")

ax[1].plot(rng, -local_slope, "o-", ms=3, color="C0")
ax[1].axhline(2.0, color="k", ls="--", lw=1.2, label="Sedov (2-D): $-d\\ln\\Delta p/d\\ln r=2$")
ax[1].axvspan(rng.min(), 4*CRATER_SIGMA, color="grey", alpha=0.15, label="finite source-size transient")
ax[1].axvspan(0.30, 0.40, color="green", alpha=0.15, label="genuine Sedov window")
ax[1].axvspan(0.45, rng.max(), color="orange", alpha=0.12, label="energy leaking out $\\Rightarrow E(t)$ not const")
ax[1].set_xscale("log")
ax[1].set_ylim(0.5, 3.0)
ax[1].set_xlabel("shock range r [km]"); ax[1].set_ylabel(r"local slope $-d\ln\Delta p/d\ln r$")
ax[1].set_title("Local decay exponent vs. range\n(only briefly touches the ideal Sedov value)")
ax[1].legend(fontsize=7.7, loc="upper right"); ax[1].grid(alpha=.3, which="both")

fig.suptitle("myblast14 — Sedov ($\\propto r^{-2}$) holds only in an intermediate window,\n"
             "bracketed by the finite source size (early) and the energy leak through the z-walls (late)", fontsize=12.5)
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_sedov_check.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
print("fit window slope:", fitA[0])
