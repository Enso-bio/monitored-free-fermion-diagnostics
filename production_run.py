"""Production run: generate all data and figures for the paper."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monitored_fermions as mf
import time, json

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})

FIG = "figs/"
DATA = {}
t_start = time.time()

# ----------------------------------------------------------------------
# 1) Half-chain sweep vs measurement probability p, several sizes
# ----------------------------------------------------------------------
plist = np.round(np.linspace(0.0, 0.6, 13), 4)
Ls = [24, 32, 48]
ntraj, T, Teq = 32, 90, 45

sweepS, sweepF, sweepM = {}, {}, {}
for L in Ls:
    S, F, M = mf.sweep(L, plist, ntraj, T, Teq, blocks=[L // 2], seed=1234 + L)
    sweepS[L] = S[L // 2]; sweepF[L] = F[L // 2]; sweepM[L] = M[L // 2]
    print(f"[{time.time()-t_start:6.1f}s] sweep L={L} done")

DATA["plist"] = plist.tolist()
DATA["sweepS"] = {L: sweepS[L].tolist() for L in Ls}
DATA["sweepF"] = {L: sweepF[L].tolist() for L in Ls}
DATA["sweepM"] = {L: sweepM[L].tolist() for L in Ls}

# Figure 2: S_{L/2} vs p
fig, ax = plt.subplots(figsize=(3.4, 2.7))
markers = ["o", "s", "^", "D"]
for L, mk in zip(Ls, markers):
    ax.plot(plist, sweepS[L] / (L / 2), mk + "-", ms=4, lw=1.0, label=f"$L={L}$")
ax.set_xlabel(r"measurement probability $p$")
ax.set_ylabel(r"$S_{L/2}\,/\,(L/2)$")
ax.legend(frameon=False, fontsize=8)
ax.set_title(r"entropy density vs monitoring", fontsize=9)
fig.tight_layout()
fig.savefig(FIG + "fig_scaling_p.pdf")
fig.savefig(FIG + "fig_scaling_p.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------
# 2) Subsystem scaling S(ell) at fixed L, weak vs strong monitoring
# ----------------------------------------------------------------------
Lb = 64
Ub = mf._expm_herm(mf.single_particle_h(Lb))
blocks = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
p_show = [0.10, 0.20, 0.50]   # log, log, area
ntraj_b, Tb, Teqb = 28, 110, 55
subS = {}
for p in p_show:
    acc = np.zeros(len(blocks))
    for k in range(ntraj_b):
        s, f, m = mf.run_trajectory(Lb, p, Ub, Tb, Teqb, np.random.default_rng(900 + k), blocks)
        acc += np.array([s[b] for b in blocks])
    subS[p] = acc / ntraj_b
    print(f"[{time.time()-t_start:6.1f}s] subsystem sweep p={p} done")

# conformal chord length for PBC
chord = (Lb / np.pi) * np.sin(np.pi * np.array(blocks) / Lb)
mask = np.array(blocks) >= 8
DATA["blocks"] = blocks
DATA["chord"] = chord.tolist()
DATA["subS"] = {p: subS[p].tolist() for p in p_show}

# effective (non-universal, p-dependent) logarithmic coefficient alpha, S ~ alpha ln(chord)
alpha = {}
for p in p_show:
    coef = np.polyfit(np.log(chord[mask]), subS[p][mask], 1)
    alpha[p] = float(coef[0])
DATA["alpha"] = alpha

fig, ax = plt.subplots(figsize=(3.4, 2.7))
colors = {0.10: "C0", 0.20: "C2", 0.50: "C3"}
labels = {0.10: r"$p=0.10$ (log.)", 0.20: r"$p=0.20$ (log.)", 0.50: r"$p=0.50$ (area)"}
for p in p_show:
    ax.plot(np.log(chord), subS[p], "o-", ms=4, lw=1.0, color=colors[p], label=labels[p])
# show log fits for the two logarithmic curves
for p in (0.10, 0.20):
    coef = np.polyfit(np.log(chord[mask]), subS[p][mask], 1)
    xx = np.log(chord[mask])
    ax.plot(xx, np.polyval(coef, xx), "--", color=colors[p], lw=0.8)
ax.set_xlabel(r"$\ln[\,(L/\pi)\sin(\pi \ell/L)\,]$")
ax.set_ylabel(r"$S(\ell)$")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.set_title(r"subsystem scaling ($L=64$)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG + "fig_subsystem.pdf")
fig.savefig(FIG + "fig_subsystem.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------
# 3) Diagnostics: charge fluctuations F and effective entangling modes M
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
for L, mk in zip(Ls, markers):
    axes[0].plot(plist, sweepF[L], mk + "-", ms=4, lw=1.0, label=f"$L={L}$")
    axes[1].plot(plist, sweepM[L], mk + "-", ms=4, lw=1.0, label=f"$L={L}$")
axes[0].set_xlabel(r"$p$"); axes[0].set_ylabel(r"$F_{L/2}=\sum_k \nu_k(1-\nu_k)$")
axes[1].set_xlabel(r"$p$"); axes[1].set_ylabel(r"$\mathcal{M}_{L/2}$ (eff. modes)")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("bipartite charge fluctuations", fontsize=9)
axes[1].set_title("effective entangling modes", fontsize=9)
fig.tight_layout()
fig.savefig(FIG + "fig_diagnostic.pdf")
fig.savefig(FIG + "fig_diagnostic.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------
# 4) Entanglement growth from a product state
# ----------------------------------------------------------------------
Lg = 48
Ug = mf._expm_herm(mf.single_particle_h(Lg))
Tg = 90
ps_g = [0.05, 0.15, 0.40]
ntraj_g = 40
growth = {p: np.zeros(Tg) for p in ps_g}
for p in ps_g:
    acc = np.zeros(Tg)
    for k in range(ntraj_g):
        rng = np.random.default_rng(500 + k)
        C = mf.neel_C(Lg); Uc = Ug.conj(); UT = Ug.T
        blk = list(range(Lg // 2))
        for t in range(Tg):
            C = Uc @ C @ UT
            for m in np.nonzero(rng.random(Lg) < p)[0]:
                C = mf.measure_site(C, int(m), rng)
            C = 0.5 * (C + C.conj().T)
            s, _, _ = mf.block_observables(C, blk)
            acc[t] += s
    growth[p] = acc / ntraj_g
    print(f"[{time.time()-t_start:6.1f}s] growth p={p} done")

DATA["growth_t"] = list(range(Tg))
DATA["growth"] = {p: growth[p].tolist() for p in ps_g}

fig, ax = plt.subplots(figsize=(3.4, 2.7))
for p in ps_g:
    ax.plot(range(Tg), growth[p], "-", lw=1.2, label=f"$p={p}$")
ax.set_xlabel(r"time step $t$")
ax.set_ylabel(r"$S_{L/2}(t)$")
ax.legend(frameon=False, fontsize=8)
ax.set_title(r"entanglement growth ($L=48$)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG + "fig_growth.pdf")
fig.savefig(FIG + "fig_growth.png", dpi=200)
plt.close(fig)

with open("data_summary.json", "w") as fh:
    json.dump(DATA, fh, indent=1)

print(f"[{time.time()-t_start:6.1f}s] ALL DONE. alpha(p) = " +
      ", ".join(f"{p}:{a:.3f}" for p, a in DATA['alpha'].items()))
