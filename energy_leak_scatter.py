import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
GAMMA = 5.0/3.0; GRAV = 9.8e-3
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]
K = 2
XSTRIDE = 12   # downsample x for a legible scatter

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
xs = xc[::XSTRIDE]
T_all, X_all, TOP_all, BOT_all = [], [], [], []

for fr in range(nf):
    t, lnr = read_frame("lnrho", fr)
    _, ux = read_frame("uu1", fr); _, uz = read_frame("uu3", fr); _, pp = read_frame("pp", fr)
    rho = np.exp(lnr); spd2 = ux*ux + uz*uz
    spec_e = 0.5*spd2 + GAMMA/(GAMMA-1)*pp/rho + GRAV*zc[:,None]
    top = (rho[-1-K,:]*uz[-1-K,:]*spec_e[-1-K,:])[::XSTRIDE]
    bot = (-rho[K,:]*uz[K,:]*spec_e[K,:])[::XSTRIDE]
    T_all.append(np.full(xs.shape, t)); X_all.append(xs)
    TOP_all.append(top); BOT_all.append(bot)

T_all=np.concatenate(T_all); X_all=np.concatenate(X_all)
TOP_all=np.concatenate(TOP_all); BOT_all=np.concatenate(BOT_all)
vmax = np.percentile(np.abs(np.concatenate([TOP_all,BOT_all])), 98)

fig, ax = plt.subplots(1,2, figsize=(14,5.6), constrained_layout=True, sharey=True, sharex=True)
s = 14 + 60*np.clip(np.abs(TOP_all)/vmax, 0, 1)
sc0 = ax[0].scatter(T_all, X_all, c=TOP_all, s=s, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     edgecolors="none", alpha=0.75)
ax[0].set_title("Energy outflow through TOP face (z$\\approx$1.6)")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("x [km]")
cb0 = fig.colorbar(sc0, ax=ax[0]); cb0.set_label("outward energy flux [code/(km s)]\n(red = leaving domain, blue = re-entering)")

s2 = 14 + 60*np.clip(np.abs(BOT_all)/vmax, 0, 1)
sc1 = ax[1].scatter(T_all, X_all, c=BOT_all, s=s2, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     edgecolors="none", alpha=0.75)
ax[1].set_title("Energy outflow through BOTTOM face (z$\\approx$0)")
ax[1].set_xlabel("t [s]")
cb1 = fig.colorbar(sc1, ax=ax[1]); cb1.set_label("outward energy flux [code/(km s)]\n(red = leaving domain, blue = re-entering)")

for a in ax: a.set_ylim(-0.7,0.7)
fig.suptitle("myblast14 — where (x) and when (t) energy leaves through the open z-boundaries\n(dot size $\\propto$ |flux|)", fontsize=13)
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_energy_leak_scatter.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
