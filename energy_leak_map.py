import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
GAMMA = 5.0/3.0; GRAV = 9.8e-3
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
dx = xc[1]-xc[0]
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]
K = 2   # cells inboard of the true edge (avoids the single-point slice-output artifact)

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
ts = []
top_map = np.empty((nf, NG))
bot_map = np.empty((nf, NG))

for fr in range(nf):
    t, lnr = read_frame("lnrho", fr)
    _, ux = read_frame("uu1", fr); _, uz = read_frame("uu3", fr); _, pp = read_frame("pp", fr)
    rho = np.exp(lnr); spd2 = ux*ux + uz*uz
    spec_e = 0.5*spd2 + GAMMA/(GAMMA-1)*pp/rho + GRAV*zc[:,None]
    top_map[fr,:] = rho[-1-K,:]*uz[-1-K,:]*spec_e[-1-K,:]     # energy flux density [energy/(km . s)] leaving through top
    bot_map[fr,:] = -rho[K,:]*uz[K,:]*spec_e[K,:]             # leaving through bottom
    ts.append(t)
ts = np.array(ts)

vmax = np.percentile(np.abs(np.concatenate([top_map, bot_map])), 99)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True, sharey=True)
ext = [X0, X1, ts[0], ts[-1]]
im0 = ax[0].imshow(top_map, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax[0].set_title("Energy outflow density through TOP face (z$\\approx$1.6)")
ax[0].set_xlabel("x [km]"); ax[0].set_ylabel("t [s]")
ax[0].set_xlim(-0.7,0.7)
cb0 = fig.colorbar(im0, ax=ax[0]); cb0.set_label("outward energy flux [code/(km s)]\n(red = leaving domain)")

im1 = ax[1].imshow(bot_map, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax[1].set_title("Energy outflow density through BOTTOM face (z$\\approx$0)")
ax[1].set_xlabel("x [km]")
ax[1].set_xlim(-0.7,0.7)
cb1 = fig.colorbar(im1, ax=ax[1]); cb1.set_label("outward energy flux [code/(km s)]\n(red = leaving domain)")

fig.suptitle("myblast14 — where (x) and when (t) energy is leaving through the open z-boundaries", fontsize=13)
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_energy_leak_map.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
