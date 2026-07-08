"""
Version 2 production run, addressing referee-style criticism:

  (1) per-trajectory recording -> real error bars (SE over independent trajectories)
  (2) statistical performance: coefficient of variation of S, F, M vs p
  (3) M vs F scatter across trajectories and measurement rates
      -> does M carry information beyond F?
  (4) subsystem scaling at L = 48, 64, 96 with model comparison
      (pure log vs log + linear) incl. AIC and parameter errors
  (5) autocorrelation time of S(t) within a trajectory
  (6) transient-doubling convergence check
  (7) growth curves with SE bands

Regenerates ALL paper figures. Output: data_v2.json + figs/*.pdf/png
"""
import numpy as np
import json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monitored_fermions as mf

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})
FIG = "figs/"
OUT = {}
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# ----------------------------------------------------------------------
# 1) Sweeps with per-trajectory recording (no p=0: singular unitary limit)
# ----------------------------------------------------------------------
plist = np.round(np.arange(0.05, 0.6001, 0.05), 3)          # 0.05 ... 0.60
sizes  = {24: 64, 32: 64, 48: 128}                            # L -> ntraj
T, Teq = 90, 45

sweep = {}   # sweep[L][p] = array (ntraj, 3) columns S, F, M
for L, ntraj in sizes.items():
    U = mf._expm_herm(mf.single_particle_h(L))
    sweep[L] = {}
    for ip, p in enumerate(plist):
        arr = np.zeros((ntraj, 3))
        for k in range(ntraj):
            rng = np.random.default_rng(10_000 * L + 1000 * ip + k)
            s, f, m = mf.run_trajectory(L, float(p), U, T, Teq, rng, [L // 2])
            arr[k] = [s[L // 2], f[L // 2], m[L // 2]]
        sweep[L][float(p)] = arr
    log(f"sweep L={L} ({ntraj} traj) done")

OUT["plist"] = plist.tolist()
OUT["sweep_mean"] = {L: {str(p): sweep[L][p].mean(0).tolist() for p in sweep[L]} for L in sweep}
OUT["sweep_se"]   = {L: {str(p): (sweep[L][p].std(0, ddof=1) / np.sqrt(len(sweep[L][p]))).tolist()
                          for p in sweep[L]} for L in sweep}

# ---- Fig: entropy vs p with error bars --------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.7))
markers = {24: "o", 32: "s", 48: "^"}
for L in sizes:
    mean = np.array([sweep[L][float(p)][:, 0].mean() for p in plist])
    se   = np.array([sweep[L][float(p)][:, 0].std(ddof=1) / np.sqrt(sizes[L]) for p in plist])
    ax.errorbar(plist, mean / (L / 2), yerr=se / (L / 2), fmt=markers[L] + "-",
                ms=4, lw=1.0, capsize=2, label=f"$L={L}$")
ax.set_xlabel(r"measurement probability $p$")
ax.set_ylabel(r"$S_{L/2}\,/\,(L/2)$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG + "fig_scaling_p.pdf"); fig.savefig(FIG + "fig_scaling_p.png", dpi=200)
plt.close(fig)

# ---- Fig: F and M vs p with error bars --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
for L in sizes:
    for j, axi in enumerate(axes):
        col = j + 1  # 1 = F, 2 = M
        mean = np.array([sweep[L][float(p)][:, col].mean() for p in plist])
        se   = np.array([sweep[L][float(p)][:, col].std(ddof=1) / np.sqrt(sizes[L]) for p in plist])
        axi.errorbar(plist, mean, yerr=se, fmt=markers[L] + "-", ms=4, lw=1.0,
                     capsize=2, label=f"$L={L}$")
axes[0].set_xlabel(r"$p$"); axes[0].set_ylabel(r"$F_{L/2}$")
axes[1].set_xlabel(r"$p$"); axes[1].set_ylabel(r"$\mathcal{M}_{L/2}$")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("number fluctuations", fontsize=9)
axes[1].set_title("spectrum participation", fontsize=9)
fig.tight_layout()
fig.savefig(FIG + "fig_diagnostic.pdf"); fig.savefig(FIG + "fig_diagnostic.png", dpi=200)
plt.close(fig)
log("sweep figures done")

# ----------------------------------------------------------------------
# 2) Statistical performance: coefficient of variation (L=48, 128 traj)
# ----------------------------------------------------------------------
L = 48
cv = {name: [] for name in ("S", "F", "M")}
for p in plist:
    arr = sweep[L][float(p)]
    for j, name in enumerate(("S", "F", "M")):
        cv[name].append(arr[:, j].std(ddof=1) / arr[:, j].mean())
OUT["cv_L48"] = {k: list(map(float, v)) for k, v in cv.items()}

fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.plot(plist, cv["S"], "o-", ms=4, lw=1.0, label=r"$S_{L/2}$")
ax.plot(plist, cv["F"], "s-", ms=4, lw=1.0, label=r"$F_{L/2}$")
ax.plot(plist, cv["M"], "^-", ms=4, lw=1.0, label=r"$\mathcal{M}_{L/2}$")
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\sigma_O\,/\,\langle O\rangle$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG + "fig_statistics.pdf"); fig.savefig(FIG + "fig_statistics.png", dpi=200)
plt.close(fig)
log("CV figure done")

# ----------------------------------------------------------------------
# 3) M vs F scatter across trajectories (L=48)
# ----------------------------------------------------------------------
p_show = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
allF, allM, allp = [], [], []
for p in p_show:
    arr = sweep[48][p]
    allF.append(arr[:, 1]); allM.append(arr[:, 2]); allp.append(np.full(len(arr), p))
allF = np.concatenate(allF); allM = np.concatenate(allM); allp = np.concatenate(allp)

# pooled quadratic fit in log-log
lx, ly = np.log(allF), np.log(allM)
coef = np.polyfit(lx, ly, 2)
resid = ly - np.polyval(coef, lx)
group = {}
for p in p_show:
    r = resid[allp == p]
    group[p] = (float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))))
# within-p Pearson correlation of (F, M)
corr = {p: float(np.corrcoef(sweep[48][p][:, 1], sweep[48][p][:, 2])[0, 1]) for p in p_show}
OUT["scatter_resid_mean_se"] = {str(p): group[p] for p in p_show}
OUT["scatter_within_p_corr"] = {str(p): corr[p] for p in p_show}
OUT["scatter_pooled_R2"] = float(1 - resid.var() / ly.var())

fig, ax = plt.subplots(figsize=(3.4, 2.9))
cmap = plt.cm.viridis
for i, p in enumerate(p_show):
    arr = sweep[48][p]
    ax.plot(arr[:, 1], arr[:, 2], "o", ms=2.5, alpha=0.6,
            color=cmap(i / (len(p_show) - 1)), label=f"$p={p}$")
xx = np.linspace(lx.min(), lx.max(), 100)
ax.plot(np.exp(xx), np.exp(np.polyval(coef, xx)), "k--", lw=1.0, label="pooled fit")
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel(r"$F_{L/2}$")
ax.set_ylabel(r"$\mathcal{M}_{L/2}$")
ax.legend(frameon=False, fontsize=6.5, ncol=2)
fig.tight_layout()
fig.savefig(FIG + "fig_scatter.pdf"); fig.savefig(FIG + "fig_scatter.png", dpi=200)
plt.close(fig)
log("scatter figure done; pooled R2 = %.4f" % OUT["scatter_pooled_R2"])

# ----------------------------------------------------------------------
# 4) Subsystem scaling, L = 48, 64, 96, with model comparison
# ----------------------------------------------------------------------
sub_cfg = {
    48: dict(blocks=[4, 6, 8, 10, 12, 16, 20, 24], T=110, Teq=55),
    64: dict(blocks=[4, 6, 8, 10, 12, 16, 20, 24, 28, 32], T=140, Teq=70),
    96: dict(blocks=[4, 6, 8, 12, 16, 20, 24, 32, 40, 48], T=200, Teq=100),
}
p_sub = [0.10, 0.20, 0.50]
ntraj_sub = 24
subdata = {}   # subdata[L][p] = (mean S(ell), se S(ell))
for L, cfg in sub_cfg.items():
    U = mf._expm_herm(mf.single_particle_h(L))
    subdata[L] = {}
    for p in p_sub:
        arr = np.zeros((ntraj_sub, len(cfg["blocks"])))
        for k in range(ntraj_sub):
            rng = np.random.default_rng(77_000 + 100 * L + int(1000 * p) + k)
            s, f, m = mf.run_trajectory(L, p, U, cfg["T"], cfg["Teq"], rng, cfg["blocks"])
            arr[k] = [s[b] for b in cfg["blocks"]]
        subdata[L][p] = (arr.mean(0), arr.std(0, ddof=1) / np.sqrt(ntraj_sub))
    log(f"subsystem L={L} done")

def fit_compare(L, p):
    """Fit S = a ln x + b  vs  S = a ln x + c*ell + b. Return dict of results."""
    cfg = sub_cfg[L]
    ell = np.array(cfg["blocks"], float)
    chord = (L / np.pi) * np.sin(np.pi * ell / L)
    S, Se = subdata[L][p]
    mask = ell >= 8
    x = np.log(chord[mask]); e = ell[mask]; y = S[mask]
    n = mask.sum()
    # model 1: log only
    X1 = np.column_stack([x, np.ones(n)])
    b1, res1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    ssr1 = float(np.sum((y - X1 @ b1) ** 2))
    cov1 = ssr1 / (n - 2) * np.linalg.inv(X1.T @ X1)
    aic1 = n * np.log(ssr1 / n) + 2 * 2
    # model 2: log + linear
    X2 = np.column_stack([x, e, np.ones(n)])
    b2, res2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    ssr2 = float(np.sum((y - X2 @ b2) ** 2))
    cov2 = ssr2 / (n - 3) * np.linalg.inv(X2.T @ X2)
    aic2 = n * np.log(ssr2 / n) + 2 * 3
    return dict(alpha1=float(b1[0]), alpha1_err=float(np.sqrt(cov1[0, 0])),
                alpha2=float(b2[0]), alpha2_err=float(np.sqrt(cov2[0, 0])),
                vol2=float(b2[1]), vol2_err=float(np.sqrt(cov2[1, 1])),
                ssr1=ssr1, ssr2=ssr2, aic1=float(aic1), aic2=float(aic2), n=int(n))

fits = {L: {p: fit_compare(L, p) for p in p_sub} for L in sub_cfg}
OUT["subsystem_fits"] = {str(L): {str(p): fits[L][p] for p in p_sub} for L in sub_cfg}
for L in sub_cfg:
    for p in p_sub:
        f = fits[L][p]
        log(f"  L={L} p={p}: alpha={f['alpha1']:.2f}({f['alpha1_err']:.2f}) | "
            f"log+lin: alpha={f['alpha2']:.2f}({f['alpha2_err']:.2f}) "
            f"vol={f['vol2']:.3f}({f['vol2_err']:.3f}) dAIC={f['aic2']-f['aic1']:+.1f}")

OUT["subsystem"] = {str(L): {str(p): dict(blocks=sub_cfg[L]["blocks"],
                                          S=subdata[L][p][0].tolist(),
                                          Se=subdata[L][p][1].tolist())
                             for p in p_sub} for L in sub_cfg}

# ---- Fig: subsystem scaling, two panels -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=False)
colsL = {48: "C0", 64: "C2", 96: "C1"}
for ax, p in zip(axes, [0.10, 0.50]):
    for L in sub_cfg:
        ell = np.array(sub_cfg[L]["blocks"], float)
        chord = (L / np.pi) * np.sin(np.pi * ell / L)
        S, Se = subdata[L][p]
        ax.errorbar(np.log(chord), S, yerr=Se, fmt="o", ms=3.5, lw=0.9,
                    capsize=2, color=colsL[L], label=f"$L={L}$")
        if p < 0.5:
            f = fits[L][p]
            mask = ell >= 8
            xx = np.log(chord[mask])
            ax.plot(xx, f["alpha1"] * xx +
                    (S[mask] - f["alpha1"] * xx).mean(), "--", lw=0.8, color=colsL[L])
    ax.set_xlabel(r"$\ln[\,(L/\pi)\sin(\pi \ell/L)\,]$")
    ax.set_title(f"$p={p}$", fontsize=9)
axes[0].set_ylabel(r"$S(\ell)$")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG + "fig_subsystem.pdf"); fig.savefig(FIG + "fig_subsystem.png", dpi=200)
plt.close(fig)
log("subsystem figure done")

# ----------------------------------------------------------------------
# 5) Autocorrelation time of S(t) within a trajectory (L=48, p=0.1, 0.4)
# ----------------------------------------------------------------------
def s_series(L, p, T, Teq, seed):
    U = mf._expm_herm(mf.single_particle_h(L))
    rng = np.random.default_rng(seed)
    C = mf.neel_C(L); Uc = U.conj(); UT = U.T
    blk = list(range(L // 2))
    out = []
    for t in range(T):
        C = Uc @ C @ UT
        for m in np.nonzero(rng.random(L) < p)[0]:
            C = mf.measure_site(C, int(m), rng)
        C = 0.5 * (C + C.conj().T)
        if t >= Teq:
            s, _, _ = mf.block_observables(C, blk)
            out.append(s)
    return np.array(out)

def tau_int(x):
    x = x - x.mean()
    n = len(x)
    if x.var() == 0:
        return 1.0
    acf = np.correlate(x, x, "full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    tau = 1.0
    for k in range(1, n // 3):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]
    return float(tau)

taus = {}
for p in (0.1, 0.4):
    tt = [tau_int(s_series(48, p, 500, 100, 31_000 + k)) for k in range(8)]
    taus[p] = (float(np.mean(tt)), float(np.std(tt, ddof=1) / np.sqrt(len(tt))))
    log(f"autocorrelation p={p}: tau = {taus[p][0]:.1f} +- {taus[p][1]:.1f} steps")
OUT["tau_int"] = {str(p): taus[p] for p in taus}

# ----------------------------------------------------------------------
# 6) Transient-doubling check (L=48)
# ----------------------------------------------------------------------
check = {}
for p in (0.1, 0.4):
    vals = {}
    for tag, (Tc, Teqc) in {"base": (90, 45), "double": (180, 90)}.items():
        U = mf._expm_herm(mf.single_particle_h(48))
        a = []
        for k in range(64):
            rng = np.random.default_rng(52_000 + int(1000 * p) + k)
            s, f, m = mf.run_trajectory(48, p, U, Tc, Teqc, rng, [24])
            a.append(s[24])
        a = np.array(a)
        vals[tag] = (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))))
    check[p] = vals
    log(f"transient check p={p}: base {vals['base'][0]:.3f}({vals['base'][1]:.3f})"
        f"  double {vals['double'][0]:.3f}({vals['double'][1]:.3f})")
OUT["transient_check"] = {str(p): check[p] for p in check}

# ----------------------------------------------------------------------
# 7) Growth with SE bands (L=48)
# ----------------------------------------------------------------------
Lg, Tg, ntraj_g = 48, 90, 40
Ug = mf._expm_herm(mf.single_particle_h(Lg))
ps_g = [0.05, 0.15, 0.40]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
growth_out = {}
for p in ps_g:
    series = np.zeros((ntraj_g, Tg))
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
            series[k, t] = s
    mean = series.mean(0); se = series.std(0, ddof=1) / np.sqrt(ntraj_g)
    ax.plot(range(Tg), mean, "-", lw=1.2, label=f"$p={p}$")
    ax.fill_between(range(Tg), mean - se, mean + se, alpha=0.25)
    growth_out[str(p)] = dict(mean=mean.tolist(), se=se.tolist())
    log(f"growth p={p} done")
OUT["growth"] = growth_out
ax.set_xlabel(r"time step $t$")
ax.set_ylabel(r"$S_{L/2}(t)$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG + "fig_growth.pdf"); fig.savefig(FIG + "fig_growth.png", dpi=200)
plt.close(fig)

with open("data_v2.json", "w") as fh:
    json.dump(OUT, fh, indent=1)

# per-trajectory raw data for reproducibility
np.savez_compressed(
    "pertraj_sweeps.npz",
    plist=plist,
    **{f"L{L}_p{p}": sweep[L][float(p)] for L in sizes for p in plist},
)
log("ALL DONE")
