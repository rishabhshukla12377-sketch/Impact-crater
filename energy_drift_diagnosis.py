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

ts=[]; Esum=[]; KEs=[]; rng=[]
K = 3   # a few cells inboard, to see wave arrival cleanly without the single-point edge artifact
uz_top=[]; uz_bot=[]; ux_left=[]; ux_right=[]

for fr in range(nf):
    t, lnr = read_frame("lnrho", fr)
    _, ux = read_frame("uu1", fr); _, uz = read_frame("uu3", fr); _, pp = read_frame("pp", fr)
    rho = np.exp(lnr); spd2 = ux*ux+uz*uz
    ke = 0.5*(rho*spd2).sum()*dA
    eint = (pp/(GAMMA-1)).sum()*dA - Eint_pre
    egrv = (rho*GRAV*zc[:,None]).sum()*dA - Egrv_t0
    Esum.append(ke+eint+egrv); KEs.append(ke)
    ppt = pp.copy(); ppt[zc >= ZS] = -np.inf
    jz, jx = np.unravel_index(np.argmax(ppt), ppt.shape)
    rng.append(np.hypot(xc[jx], zc[jz]-ZS))
    uz_top.append(np.abs(uz[-1-K,:]).max())
    uz_bot.append(np.abs(uz[K,:]).max())
    ux_left.append(np.abs(ux[:,K]).max())
    ux_right.append(np.abs(ux[:,-1-K]).max())
    ts.append(t)

ts=np.array(ts); Esum=np.array(Esum); KEs=np.array(KEs); rng=np.array(rng)
uz_top=np.array(uz_top); uz_bot=np.array(uz_bot); ux_left=np.array(ux_left); ux_right=np.array(ux_right)

dEdt = np.gradient(Esum, ts)

# first time each wall shows non-negligible motion (arrival of a wave/wall interaction)
def first_cross(arr, thresh):
    idx = np.argmax(arr > thresh)
    return ts[idx] if arr[idx] > thresh else None
thr = 0.02
t_top = first_cross(uz_top, thr); t_bot = first_cross(uz_bot, thr)
t_left = first_cross(ux_left, thr); t_right = first_cross(ux_right, thr)
print(f"wall-motion first exceeds {thr}: top={t_top} bot={t_bot} left={t_left} right={t_right}")

print(f"\n{'t':>7} {'Esum':>9} {'dE/dt':>10} {'KE':>9} {'r_shock':>8} {'uz_top':>8} {'uz_bot':>8} {'ux_L':>8} {'ux_R':>8}")
for i in range(0, len(ts), 4):
    print(f"{ts[i]:7.3f} {Esum[i]:9.5f} {dEdt[i]:10.5f} {KEs[i]:9.5f} {rng[i]:8.3f} {uz_top[i]:8.4f} {uz_bot[i]:8.4f} {ux_left[i]:8.4f} {ux_right[i]:8.4f}")

# save full numeric table to csv
import csv
with open("/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/analysis/energy_drift_data.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["t","Esum","dEsum_dt","KE","r_shock","uz_top_max","uz_bot_max","ux_left_max","ux_right_max"])
    for i in range(len(ts)):
        w.writerow([ts[i],Esum[i],dEdt[i],KEs[i],rng[i],uz_top[i],uz_bot[i],ux_left[i],ux_right[i]])
print("\nwrote analysis/energy_drift_data.csv")

plt.rcParams.update({"font.size":12,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False})
fig, ax = plt.subplots(2,1, figsize=(9,8.5), sharex=True, constrained_layout=True)
ax[0].plot(ts, Esum, color="C0", lw=2.2)
ax[0].axhline(E_DEP, color="grey", ls=":", lw=1)
ax[0].set_ylabel("tracked energy\n[code units]")
ax[0].set_title("Energy budget (fixed-BC rerun)")

ax[1].plot(ts, dEdt, color="C3", lw=1.8, label="dE$_{sum}$/dt (instantaneous drift rate)")
ax[1].axhline(0, color="grey", lw=0.8, ls=":")
for tw, lab, c in [(t_top,"top wave arrival","C1"), (t_bot,"bottom wave arrival","C2"),
                    (t_left,"left wave arrival","C4"), (t_right,"right wave arrival","C5")]:
    if tw is not None:
        ax[1].axvline(tw, color=c, ls="--", lw=1.2, alpha=0.8, label=f"{lab} (t={tw:.2f}s)")
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("dE/dt [code/s]")
ax[1].legend(fontsize=8, loc="upper left")
ax[1].set_title("Drift rate vs. wave/shock arrival at each wall")

out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/analysis/figures/myblast14_energy_drift_diagnosis.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
