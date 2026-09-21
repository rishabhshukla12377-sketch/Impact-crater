import numpy as np, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14"

def load(path):
    rows = list(csv.DictReader(open(path)))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return d

old = load(f"{BASE}/postproc/myblast14_postproc.csv")
new = load(f"{BASE}/fixed_bc_no_sponge_run/postproc/fixed_bc_no_sponge_run_postproc.csv")

plt.rcParams.update({"font.size":12,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False})
fig, ax = plt.subplots(2,2, figsize=(12,9), constrained_layout=True)

ax[0,0].plot(old["t"], old["d_axis"]*1e3, color="C3", lw=2, label="before fix (leaky bcz)")
ax[0,0].plot(new["t"], new["d_axis"]*1e3, color="C0", lw=2, label="after fix (closed bcz)")
ax[0,0].set_xlabel("t [s]"); ax[0,0].set_ylabel("axial crater depth [m]")
ax[0,0].set_title("Crater depth vs. time"); ax[0,0].legend(fontsize=9)

ax[0,1].plot(old["t"], old["R"]*1e3, color="C3", lw=2, label="before fix")
ax[0,1].plot(new["t"], new["R"]*1e3, color="C0", lw=2, label="after fix")
ax[0,1].set_xlabel("t [s]"); ax[0,1].set_ylabel("crater half-width R [m]")
ax[0,1].set_title("Crater half-width vs. time"); ax[0,1].legend(fontsize=9)

m_old = (old["shock_rng"]>0.01) & (old["dp_peak"]>0)
m_new = (new["shock_rng"]>0.01) & (new["dp_peak"]>0)
ax[1,0].loglog(old["shock_rng"][m_old], old["dp_peak"][m_old], "o-", ms=3, color="C3", label="before fix")
ax[1,0].loglog(new["shock_rng"][m_new], new["dp_peak"][m_new], "o-", ms=3, color="C0", label="after fix")
rr = np.linspace(0.02, 1.0, 50)
p0 = old["dp_peak"][m_old][3]
r0 = old["shock_rng"][m_old][3]
ax[1,0].loglog(rr, p0*(rr/r0)**-2.0, "k--", lw=1, label=r"$\propto r^{-2}$ (2-D Sedov)")
ax[1,0].set_xlabel("range from impact point r [km]"); ax[1,0].set_ylabel(r"peak over-pressure $\Delta p$")
ax[1,0].set_title("Blast decay: before vs. after"); ax[1,0].legend(fontsize=9)

ax[1,1].plot(old["t"], old["KE"], color="C3", lw=2, label="before fix")
ax[1,1].plot(new["t"], new["KE"], color="C0", lw=2, label="after fix")
ax[1,1].set_xlabel("t [s]"); ax[1,1].set_ylabel("kinetic energy [code units]")
ax[1,1].set_title("Kinetic energy vs. time"); ax[1,1].legend(fontsize=9)

fig.suptitle("myblast14 -- crater metrics & Sedov decay: leaky-BC vs. fixed-BC run", fontsize=13.5)
out = f"{BASE}/analysis/figures/myblast14_crater_sedov_before_after.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)

print(f"\nfinal crater depth: before={old['d_axis'][-1]*1e3:.1f}m  after={new['d_axis'][-1]*1e3:.1f}m")
print(f"final crater half-width: before={old['R'][-1]*1e3:.1f}m  after={new['R'][-1]*1e3:.1f}m")
print(f"peak crater depth: before={old['d_axis'].max()*1e3:.1f}m @t={old['t'][np.argmax(old['d_axis'])]:.2f}   after={new['d_axis'].max()*1e3:.1f}m @t={new['t'][np.argmax(new['d_axis'])]:.2f}")
print(f"final KE: before={old['KE'][-1]:.4f}  after={new['KE'][-1]:.4f}")
print(f"final rim uplift: before={old['rim_h'][-1]*1e3:.1f}m  after={new['rim_h'][-1]*1e3:.1f}m")
