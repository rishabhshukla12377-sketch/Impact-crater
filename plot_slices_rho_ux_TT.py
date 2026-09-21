import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
PAYLOAD = 131096
FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]   # last local column of each x-proc block (known slice-output artifact)

def nframes(fld): return os.path.getsize(f"{D}/proc0/slice_{fld}.xz")//FRAME
def read_frame(fld, fr, declean=True):
    g = np.empty((NG,NG)); t = 0.0
    for p in range(NPX*NPZ):
        ipx, ipz = p % NPX, p // NPX
        with open(f"{D}/proc{p}/slice_{fld}.xz","rb") as f:
            f.seek(fr*FRAME + 12)
            sl = np.fromfile(f, np.float64, NL*NL).reshape(NL,NL)
            f.seek(fr*FRAME + 12 + NL*NL*8 + 8)
            t = np.fromfile(f, np.float64, 1)[0]
        g[ipz*NL:(ipz+1)*NL, ipx*NL:(ipx+1)*NL] = sl
    if declean:
        for c in SEAM_COLS:
            g[:, c] = 0.5*(g[:, c-1] + g[:, c+1])
    return t, g

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

nf = nframes("lnrho")
frames = [10, 30, nf-1]                     # t = 0.05, 0.15, 0.50
fig, ax = plt.subplots(len(frames), 3, figsize=(12.5, 4.1*len(frames)), constrained_layout=True)

fields = [
    ("lnrho", lambda a: np.exp(a), r"$\rho$ [code]", "viridis", dict(vmin=0, vmax=3.4)),
    ("uu1",   lambda a: a,          r"$u_x$ [km/s]",  "RdBu_r", dict(vmin=-1.5, vmax=1.5)),
    ("TT",    lambda a: np.log10(np.clip(a,1e-3,None)), r"$\log_{10}\,T$ [K]", "inferno", dict(vmin=0, vmax=3.6)),
]

ext = [X0, X1, Z0, Z1]
for r, fr in enumerate(frames):
    for c, (fld, xf, label, cmap, kw) in enumerate(fields):
        t, raw = read_frame(fld, fr)
        dat = xf(raw)
        a = ax[r, c]
        im = a.imshow(dat, origin="lower", extent=ext, aspect="equal", cmap=cmap, interpolation="nearest", **kw)
        a.axhline(1.0, color="w", lw=0.7, ls=":", alpha=0.8)
        a.set_xlim(-0.55, 0.55); a.set_ylim(0.45, 1.35)
        a.set_title(f"{label},  t = {t:.3f} s", fontsize=11.5)
        if r == len(frames)-1:
            a.set_xlabel("x [km]")
        if c == 0:
            a.set_ylabel("z [km]")
        a.xaxis.set_major_locator(MaxNLocator(5))
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=9)

fig.suptitle("myblast14 — run 1 slices  (crater_energy=0.30, frac_kin=0.5, $\\gamma$=5/3, Earth g)",
             fontsize=13, y=1.02)

out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_run1_slices_rho_ux_TT.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
