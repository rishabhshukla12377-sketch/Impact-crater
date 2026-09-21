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
ts=[]; Esum=[]; Ftop=[]; Fbot=[]; Fside=[]
_, lnr0 = read_frame("lnrho",0); _, pp0 = read_frame("pp",0)
rho0 = np.exp(lnr0)
Eint_t0 = (pp0/(GAMMA-1)).sum()*dA
Egrv_t0 = (rho0*GRAV*zc[:,None]).sum()*dA
Eint_pre = Eint_t0 - (1-F_KIN)*E_DEP

for fr in range(nf):
    t, lnr = read_frame("lnrho", fr)
    _, ux = read_frame("uu1", fr); _, uz = read_frame("uu3", fr); _, pp = read_frame("pp", fr)
    rho = np.exp(lnr); spd2 = ux*ux+uz*uz
    ke = 0.5*(rho*spd2).sum()*dA
    eint = (pp/(GAMMA-1)).sum()*dA - Eint_pre
    egrv = (rho*GRAV*zc[:,None]).sum()*dA - Egrv_t0
    Esum.append(ke+eint+egrv)
    spec_e = 0.5*spd2 + GAMMA/(GAMMA-1)*pp/rho + GRAV*zc[:,None]
    Ftop.append((rho[-1-K,:]*uz[-1-K,:]*spec_e[-1-K,:]).sum()*dx)
    Fbot.append((-rho[K,:]*uz[K,:]*spec_e[K,:]).sum()*dx)
    Fs = (-rho[:,K]*ux[:,K]*spec_e[:,K]).sum()*dz + (rho[:,-1-K]*ux[:,-1-K]*spec_e[:,-1-K]).sum()*dz
    Fside.append(Fs)
    ts.append(t)

ts=np.array(ts); Esum=np.array(Esum)
Ftop=np.array(Ftop); Fbot=np.array(Fbot); Fside=np.array(Fside)
def cum(F): return np.concatenate([[0], np.cumsum(0.5*(F[1:]+F[:-1])*np.diff(ts))])
cum_top, cum_bot, cum_side = cum(Ftop), cum(Fbot), cum(Fside)

plt.rcParams.update({"font.size":13,"figure.facecolor":"white","savefig.facecolor":"white",
                      "axes.spines.top":False, "axes.spines.right":False})

# ---- Plot 1: simple conservation check ----
fig1, ax1 = plt.subplots(figsize=(7.5,5.5), constrained_layout=True)
ax1.axhline(E_DEP, color="grey", ls="--", lw=1.5, label="energy put in at t=0 (constant, should stay flat)")
ax1.plot(ts, Esum, color="C3", lw=2.5, label="energy actually tracked in the box")
loss_pct = 100*(Esum[0]-Esum[-1])/Esum[0]
ax1.annotate(f"−{loss_pct:.0f}% by\nt={ts[-1]:.1f}s", xy=(ts[-1], Esum[-1]), xytext=(ts[-1]-0.11, Esum[-1]-0.05),
             fontsize=12, color="C3", fontweight="bold")
ax1.set_xlabel("t [s]"); ax1.set_ylabel("energy [code units]")
ax1.set_title("Deposited energy is not staying in the box")
ax1.set_ylim(0, 0.34)
ax1.legend(fontsize=10, loc="lower left")
out1 = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_energy_conservation_simple.png"
fig1.savefig(out1, dpi=150)
print("wrote", out1)

# ---- Plot 2: simple leak attribution ----
fig2, ax2 = plt.subplots(figsize=(7.5,5.5), constrained_layout=True)
ax2.plot(ts, cum_top, color="C3", lw=2.5, label="lost through the top (into the atmosphere)")
ax2.plot(ts, cum_bot, color="C0", lw=2.5, label="lost through the bottom (target base)")
ax2.plot(ts, cum_side, color="grey", lw=1.5, ls=":", label="lost through left+right walls (~0)")
ax2.set_xlabel("t [s]"); ax2.set_ylabel("cumulative energy lost [code units]")
ax2.set_title("Where the missing energy goes")
ax2.legend(fontsize=10, loc="upper left")
out2 = "/cfs/klemming/scratch/r/rshbhsh/Impact/myblast14/myblast14_energy_leak_simple.png"
fig2.savefig(out2, dpi=150)
print("wrote", out2)
