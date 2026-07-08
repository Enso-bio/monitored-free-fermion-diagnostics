import numpy as np
import monitored_fermions as mf

L = 32
U = mf._expm_herm(mf.single_particle_h(L))

# purity check
rng = np.random.default_rng(3)
C = mf.neel_C(L); Uc = U.conj(); UT = U.T
for t in range(40):
    C = Uc @ C @ UT
    for m in np.nonzero(rng.random(L) < 0.2)[0]:
        C = mf.measure_site(C, int(m), rng)
    C = 0.5 * (C + C.conj().T)
ev = np.linalg.eigvalsh(C)
print('full-C eigenvalue min/max:', round(float(ev.min()), 4), round(float(ev.max()), 4))
print('purity dev sum nu(1-nu):', round(float(np.sum(ev * (1 - ev))), 4))

for p in [0.0, 0.05, 0.2, 0.5, 0.9]:
    acc = 0.0
    for k in range(20):
        s, f, m = mf.run_trajectory(L, p, U, 80, 40, np.random.default_rng(100 + int(p * 1000) + k), [L // 2])
        acc += s[L // 2]
    print('p=', p, '  S_half=', round(acc / 20, 3))
