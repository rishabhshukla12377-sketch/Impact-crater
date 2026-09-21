import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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

good = np.ones((NG,NG), dtype=bool)
for r in SEAM: good[max(r-2,0):r+3,:] = False
for c in SEAM: good[:, max(c-2,0):c+3] = False

frames = [8,12,16,20,24,28,32,36,40]
step = min(dx,dz)*0.6
half_win = 0.06

records = []   # (t, r_rel, rho, shock_norm, r_peak, fwhm_km)
for fr in frames:
    t, shock = read_frame("shock", fr)
    _, lnr = read_frame("lnrho", fr)
    rho = np.exp(lnr)
    sm = np.where(good, shock, -1)
    jz, jx = np.unravel_index(np.argmax(sm), sm.shape)
    r_peak = np.hypot(xc[jx], zc[jz]-ZS)
    theta = np.arctan2(zc[jz]-ZS, xc[jx])
    rr = np.arange(r_peak-half_win, r_peak+half_win, step)
    xr = rr*np.cos(theta); zr = ZS + rr*np.sin(theta)
    rho_r = bilinear(rho, xr, zr)
    shock_r = np.clip(bilinear(shock, xr, zr), 0, None)
    sn = shock_r/shock_r.max() if shock_r.max()>0 else shock_r
    half = sn > 0.5
    fwhm_km = (rr[half].max()-rr[half].min()) if half.any() else 0.0
    records.append((t, rr-r_peak, rho_r, sn, r_peak, fwhm_km))
    print(f"t={t:.3f}  r_peak={r_peak:.4f}  fwhm_km={fwhm_km:.4f}  fwhm/R={100*fwhm_km/r_peak:.1f}%")

plt.rcParams.update({"font.size":11,"figure.facecolor":"white","savefig.facecolor":"white"})
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection="3d")

for (t, rrel, rho_r, sn, r_peak, fwhm_km) in records:
    pts = np.array([rrel, np.full_like(rrel, t), rho_r]).T.reshape(-1,1,3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap="inferno", norm=plt.Normalize(0,1))
    lc.set_array(sn[:-1])
    lc.set_linewidth(3)
    ax.add_collection3d(lc)

ax.set_xlim(-0.06,0.06); ax.set_ylim(records[0][0]-0.01, records[-1][0]+0.01)
ax.set_zlim(2.8, 4.4)
ax.set_xlabel("r $-$ r$_{shock}$(t)  [km]")
ax.set_ylabel("t [s]")
ax.set_zlabel(r"density $\rho$ [code]")
ax.set_title("The shock keeps a near-constant relative width ($\\approx$9.5% of R) as it expands\n(color = shock-viscosity indicator, normalized per frame; t=0.04-0.20s)")
ax.view_init(elev=22, azim=-60)

m = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0,1))
m.set_array([])
cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1)
cb.set_label("shock indicator (normalized to frame peak)")

out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_shock_width_3d.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
