import numpy as np, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/analysis/energy_drift_data.csv")))
t = np.array([float(r["t"]) for r in rows])
dEdt = np.array([float(r["dEsum_dt"]) for r in rows])
rng = np.array([float(r["r_shock"]) for r in rows])
uz_bot = np.array([float(r["uz_bot_max"]) for r in rows])
uz_top = np.array([float(r["uz_top_max"]) for r in rows])

m = t > 0.15
plt.rcParams.update({"font.size":12,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False})
fig, ax1 = plt.subplots(figsize=(9,5.8), constrained_layout=True)
ax1.plot(t[m], dEdt[m], color="C3", lw=2.2, label="dE$_{sum}$/dt (drift rate)")
ax1.set_xlabel("t [s]"); ax1.set_ylabel("dE/dt [code/s]", color="C3")
ax1.tick_params(axis="y", labelcolor="C3")
ax1.set_ylim(0, 0.022)

ax2 = ax1.twinx()
ax2.plot(t[m], rng[m], color="C0", lw=1.8, ls="--", label="shock range r(t)")
ax2.plot(t[m], uz_bot[m]*20, color="C2", lw=1.8, label="|u$_z$| at bottom wall $\\times$20")
ax2.axhline(1.0, color="grey", lw=1, ls=":")
ax2.text(t[m][0], 1.0, " domain half-size (x=±1, or z=0 from surface)", fontsize=8, color="grey", va="bottom")
ax2.set_ylabel("shock range [km]  /  scaled |u$_z$|", color="C0")
ax2.tick_params(axis="y", labelcolor="C0")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper left")
ax1.set_title("Post-transient drift rate climbs again once the shock\nreaches the bottom wall (t$\\gtrsim$0.36s) -- a reflection effect on top of a small steady baseline")
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/analysis/figures/myblast14_energy_drift_zoom.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

# quantify: baseline drift (0.16-0.34s) vs late drift (0.40-0.50s)
b = (t>=0.16)&(t<=0.34)
l = (t>=0.40)&(t<=0.50)
print(f"baseline dE/dt (t=0.16-0.34, before bottom-wall interaction): mean={dEdt[b].mean():.5f}  range=[{dEdt[b].min():.5f},{dEdt[b].max():.5f}]")
print(f"late dE/dt      (t=0.40-0.50, after bottom-wall interaction): mean={dEdt[l].mean():.5f}  range=[{dEdt[l].min():.5f},{dEdt[l].max():.5f}]")
print(f"ratio late/baseline: {dEdt[l].mean()/dEdt[b].mean():.2f}x")
