import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/data"
NPX = NPZ = 8; NL = 128; NG = NPX*NL
X0,X1,Z0,Z1 = -1.0, 1.0, 0.0, 1.6
GAMMA = 5.0/3.0; GRAV = 9.8e-3
PAYLOAD = 131096; FRAME = 4 + PAYLOAD + 4
xc = np.linspace(X0,X1,NG); zc = np.linspace(Z0,Z1,NG)
dx = xc[1]-xc[0]; dz = zc[1]-zc[0]; dA = dx*dz
SEAM_COLS = [k*NL - 1 for k in range(1, NPX)]
E_DEP, F_KIN = 0.30, 0.50
K = 2

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
ts=[]; KE=[]; EINT=[]; EGRV=[]; Ftop=[]; Fbot=[]; Fleft=[]; Fright=[]
_, lnr0 = read_frame("lnrho",0); _, pp0 = read_frame("pp",0)
rho0 = np.exp(lnr0)
Eint_t0 = (pp0/(GAMMA-1)).sum()*dA
Egrv_t0 = (rho0*GRAV*zc[:,None]).sum()*dA
Eint_pre = Eint_t0 - (1-F_KIN)*E_DEP

for fr in range(nf):
    t, lnr = read_frame("lnrho", fr)
    _, ux = read_frame("uu1", fr); _, uz = read_frame("uu3", fr); _, pp = read_frame("pp", fr)
    rho = np.exp(lnr); spd2 = ux*ux+uz*uz
    KE.append(0.5*(rho*spd2).sum()*dA)
    EINT.append((pp/(GAMMA-1)).sum()*dA - Eint_pre)
    EGRV.append((rho*GRAV*zc[:,None]).sum()*dA - Egrv_t0)
    spec_e = 0.5*spd2 + GAMMA/(GAMMA-1)*pp/rho + GRAV*zc[:,None]
    Ftop.append((rho[-1-K,:]*uz[-1-K,:]*spec_e[-1-K,:]).sum()*dx)
    Fbot.append((-rho[K,:]*uz[K,:]*spec_e[K,:]).sum()*dx)
    Fleft.append((-rho[:,K]*ux[:,K]*spec_e[:,K]).sum()*dz)
    Fright.append((rho[:,-1-K]*ux[:,-1-K]*spec_e[:,-1-K]).sum()*dz)
    ts.append(t)

ts=np.array(ts); KE=np.array(KE); EINT=np.array(EINT); EGRV=np.array(EGRV)
Ftop=np.array(Ftop); Fbot=np.array(Fbot); Fleft=np.array(Fleft); Fright=np.array(Fright)
Esum = KE+EINT+EGRV
def cum(F): return np.concatenate([[0], np.cumsum(0.5*(F[1:]+F[:-1])*np.diff(ts))])
cum_top, cum_bot, cum_left, cum_right = cum(Ftop), cum(Fbot), cum(Fleft), cum(Fright)
cum_total = cum_top+cum_bot+cum_left+cum_right
missing = Esum[0]-Esum

# fractional attribution over time (of the outflow that has happened so far)
eps = 1e-9
frac_top = 100*cum_top/np.maximum(cum_total,eps)
frac_bot = 100*cum_bot/np.maximum(cum_total,eps)

plt.rcParams.update({"font.size":11,"figure.facecolor":"white","savefig.facecolor":"white"})
fig, ax = plt.subplots(1,2, figsize=(13.5,5.4), constrained_layout=True)

ax[0].plot(ts, KE, label="kinetic $E_K$")
ax[0].plot(ts, EINT, label=r"internal $-$ pre-impact")
ax[0].plot(ts, EGRV, label=r"grav. PE $-$ pre-impact")
ax[0].plot(ts, Esum, "k--", lw=1.6, label="sum (tracked total)")
ax[0].axhline(E_DEP, color="grey", lw=0.8, ls=":")
ax[0].text(ts[-1], E_DEP, f" deposited {E_DEP:.2f}", ha="right", va="bottom", fontsize=8, color="grey")
ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("energy [code units]")
ax[0].set_title("Domain-integrated energy budget\n(NOT conserved: drops 43% by t=0.5)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

ax[1].plot(ts, missing, "k--", lw=2.0, label="missing energy = $E_{sum}(0)-E_{sum}(t)$", zorder=5)
ax[1].plot(ts, cum_top, lw=2.0, color="C3", label="cum. outflow: TOP / atmosphere (z$\\approx$1.6)")
ax[1].plot(ts, cum_bot, lw=2.0, color="C0", label="cum. outflow: BOTTOM / target base (z$\\approx$0)")
ax[1].plot(ts, cum_right, lw=1.2, color="C2", label="cum. outflow: right (x$\\approx$+1)")
ax[1].plot(ts, cum_left, lw=1.2, color="C1", label="cum. outflow: left (x$\\approx$-1)")
ax[1].plot(ts, cum_total, "r:", lw=1.4, label="sum of 4 faces")
callout_pts = [0.10, 0.30, 0.50]
ypos = [0.14, 0.155, 0.17]
for tm, yp in zip(callout_pts, ypos):
    i = np.argmin(np.abs(ts-tm))
    ax[1].annotate(f"t={ts[i]:.2f}: top={frac_top[i]:.0f}%, bottom={frac_bot[i]:.0f}%",
                    xy=(ts[i], cum_total[i]), xytext=(ts[i]+0.01, yp),
                    fontsize=8, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="dimgrey"))
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("energy [code units]")
ax[1].set_title("Cumulative energy outflow by face\n(atmosphere leak dominates early; base leak dominates late)")
ax[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.62)); ax[1].grid(alpha=.3)

fig.suptitle("myblast14 — the deposited energy is NOT conserved: it leaks through BOTH open z-faces\n"
             "(atmosphere/top first, target-base/bottom only once the shock gets there)", fontsize=12.5)
out = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_energy_conservation_check.png"
fig.savefig(out, dpi=145, bbox_inches="tight")
print("wrote", out)
for tm in [0.05,0.10,0.15,0.25,0.35,0.45,0.50]:
    i = np.argmin(np.abs(ts-tm))
    print(f"t={ts[i]:.3f}  missing={missing[i]:.4f}  cum_top={cum_top[i]:.4f} ({frac_top[i]:.0f}%)  cum_bot={cum_bot[i]:.4f} ({frac_bot[i]:.0f}%)  cum_total={cum_total[i]:.4f}")
