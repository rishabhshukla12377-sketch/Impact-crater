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
dx = xc[1]-xc[0]; dz = zc[1]-zc[0]
SEAM = [k*NL - 1 for k in range(1, NPX+1)]

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
    gg = g.copy()
    for r in SEAM: gg[r,:] = g[r-1,:] if r+1>=NG else 0.5*(g[r-1,:]+g[r+1,:])
    for c in SEAM: gg[:,c] = g[:,c-1] if c+1>=NG else 0.5*(g[:,c-1]+g[:,c+1])
    return t, gg

def bilinear(g, x, z):
    fx = (x-X0)/dx; fz = (z-Z0)/dz
    ix0 = np.clip(np.floor(fx).astype(int), 0, NG-2); iz0 = np.clip(np.floor(fz).astype(int), 0, NG-2)
    tx = fx-ix0; tz = fz-iz0
    return (g[iz0,ix0]*(1-tx)*(1-tz) + g[iz0,ix0+1]*tx*(1-tz)
          + g[iz0+1,ix0]*(1-tx)*tz + g[iz0+1,ix0+1]*tx*tz)

fr_snap = 20
t, pp = read_frame("pp", fr_snap)
_, lnr = read_frame("lnrho", fr_snap)
_, shock = read_frame("shock", fr_snap)
rho = np.exp(lnr)

good = np.ones((NG,NG), dtype=bool)
for r in SEAM: good[max(r-2,0):r+3,:] = False
for c in SEAM: good[:, max(c-2,0):c+3] = False
sm = np.where(good, shock, -1)
jz, jx = np.unravel_index(np.argmax(sm), sm.shape)
r_peak = np.hypot(xc[jx], zc[jz]-ZS)
theta = np.arctan2(zc[jz]-ZS, xc[jx])
print(f"t={t:.3f}  shock peak={sm[jz,jx]:.6f} at r={r_peak:.4f} km, theta={np.degrees(theta):.1f} deg")

step = min(dx,dz)*0.4
rr = np.arange(r_peak-0.05, r_peak+0.05, step)
xr = rr*np.cos(theta); zr = ZS + rr*np.sin(theta)
rho_r = bilinear(rho, xr, zr)
pp_r  = bilinear(pp,  xr, zr)
shock_r = np.clip(bilinear(shock, xr, zr), 0, None)

thresh = 0.5*shock_r.max()   # FWHM
active = shock_r > thresh
w_km = rr[active].max()-rr[active].min() if active.any() else 0.0
ncells = w_km/step

plt.rcParams.update({"font.size":13,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False})
fig, ax1 = plt.subplots(figsize=(9,5.8), constrained_layout=True)
ax1.plot(rr, rho_r, color="C0", lw=2.4, marker="o", ms=3, label=r"$\rho(r)$ (density jump = the shock)")
ax1.set_xlabel(f"range from impact point r [km]   (t={t:.2f}s)")
ax1.set_ylabel(r"density $\rho$ [code]", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")

ax2 = ax1.twinx()
ax2.plot(rr, shock_r/shock_r.max(), color="C3", lw=2.4, marker="o", ms=3,
          label=r"shock viscosity indicator (normalized to its peak)")
ax2.set_ylabel("shock indicator\n(normalized)", color="C3")
ax2.tick_params(axis="y", labelcolor="C3")
ax2.set_ylim(-0.05,1.15)

if active.any():
    ax1.axvspan(rr[active].min(), rr[active].max(), color="red", alpha=0.13,
                label=f"FWHM: {ncells:.0f} cells $\\approx$ {100*w_km/r_peak:.0f}% of shock radius R")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper left")
ax1.set_title(f"Shock transition width $\\approx${ncells:.0f} cells ({100*w_km/r_peak:.0f}% of R) —\nthin compared to the flow, not razor-sharp")
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_shock_width.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out, " width(km)=", w_km, " ncells=", ncells)
