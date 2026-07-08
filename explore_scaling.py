import numpy as np
import monitored_fermions as mf

Lb = 64
Ub = mf._expm_herm(mf.single_particle_h(Lb))
blocks = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
chord = (Lb / np.pi) * np.sin(np.pi * np.array(blocks) / Lb)

for p in [0.0, 0.1, 0.15, 0.2, 0.3, 0.5]:
    acc = np.zeros(len(blocks))
    ntraj = 24
    for k in range(ntraj):
        s, f, m = mf.run_trajectory(Lb, p, Ub, 110, 55, np.random.default_rng(900 + k), blocks)
        acc += np.array([s[b] for b in blocks])
    S = acc / ntraj
    mask = np.array(blocks) >= 8
    # log fit
    cl = np.polyfit(np.log(chord[mask]), S[mask], 1)
    pred = np.polyval(cl, np.log(chord[mask]))
    ss_res = np.sum((S[mask] - pred) ** 2)
    ss_tot = np.sum((S[mask] - S[mask].mean()) ** 2)
    r2_log = 1 - ss_res / ss_tot
    # linear (volume) fit
    cv = np.polyfit(np.array(blocks)[mask], S[mask], 1)
    predv = np.polyval(cv, np.array(blocks)[mask])
    r2_lin = 1 - np.sum((S[mask] - predv) ** 2) / ss_tot
    print(f"p={p:4.2f}  S_half={S[-1]:5.2f}  logslope={cl[0]:5.2f} R2log={r2_log:5.3f}  R2lin={r2_lin:5.3f}")
