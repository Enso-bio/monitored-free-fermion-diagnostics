"""
Monitored free-fermion chain: Gaussian (correlation-matrix) simulation.

State: pure fermionic Gaussian state, fully specified by the single-particle
correlation matrix C_{ij} = <c_i^dagger c_j>.

Dynamics per discrete step:
  1) Unitary hopping:  C -> conj(U) C U^T,  U = exp(-i h dt),
     with single-particle Hamiltonian h_{ij} = -J (delta_{i,j+1}+delta_{i+1,j})
     plus periodic-boundary hopping.
  2) Stochastic projective measurement of the local occupation n_l on each site
     independently with probability p. The Gaussian update rules (verified on a
     2-site test case) are, for measured site m:
        outcome n_m=1 (prob C_mm):  C_ij -> C_ij - C_im C_mj / C_mm ,  then row/col m -> 0, C_mm=1
        outcome n_m=0 (prob 1-C_mm):C_ij -> C_ij + C_im C_mj / (1-C_mm), then row/col m -> 0, C_mm=0

Observables (from the reduced correlation matrix C_A of a contiguous block A):
  eigenvalues nu_k in [0,1]
  von Neumann entropy   S_A   = -sum [ nu ln nu + (1-nu) ln(1-nu)]
  bipartite charge var  F_A   =  sum nu(1-nu)
  effective entangling modes  M_A = exp(-sum w_k ln w_k), w_k = nu(1-nu)/F_A
"""

import numpy as np

def single_particle_h(L, J=1.0, pbc=True):
    h = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        h[i, i + 1] = -J
        h[i + 1, i] = -J
    if pbc:
        h[0, L - 1] = -J
        h[L - 1, 0] = -J
    return h

def neel_C(L):
    C = np.zeros((L, L), dtype=complex)
    for i in range(0, L, 2):
        C[i, i] = 1.0
    return C

def measure_site(C, m, rng):
    pocc = np.real(C[m, m])
    pocc = min(max(pocc, 0.0), 1.0)
    if rng.random() < pocc:
        # outcome occupied
        col = C[:, m].copy()
        row = C[m, :].copy()
        C -= np.outer(col, row) / pocc
        C[m, :] = 0.0
        C[:, m] = 0.0
        C[m, m] = 1.0
    else:
        col = C[:, m].copy()
        row = C[m, :].copy()
        C += np.outer(col, row) / (1.0 - pocc)
        C[m, :] = 0.0
        C[:, m] = 0.0
        C[m, m] = 0.0
    return C

def block_observables(C, block):
    CA = C[np.ix_(block, block)]
    CA = 0.5 * (CA + CA.conj().T)
    nu = np.linalg.eigvalsh(CA)
    nu = np.clip(np.real(nu), 1e-12, 1.0 - 1e-12)
    S = float(np.sum(-(nu * np.log(nu) + (1 - nu) * np.log(1 - nu))))
    g = nu * (1 - nu)
    F = float(np.sum(g))
    if F > 1e-12:
        w = g / F
        w = np.clip(w, 1e-300, None)
        M = float(np.exp(-np.sum(w * np.log(w))))
    else:
        M = 0.0
    return S, F, M

def run_trajectory(L, p, U, T, T_eq, rng, blocks):
    """Return time-averaged (over t>T_eq) observables for each block size."""
    C = neel_C(L)
    Uc = U.conj()
    UT = U.T
    accS = {b: 0.0 for b in blocks}
    accF = {b: 0.0 for b in blocks}
    accM = {b: 0.0 for b in blocks}
    nrec = 0
    for t in range(T):
        C = Uc @ C @ UT
        # stochastic measurements
        draw = rng.random(L) < p
        for m in np.nonzero(draw)[0]:
            C = measure_site(C, int(m), rng)
        C = 0.5 * (C + C.conj().T)
        if t >= T_eq:
            nrec += 1
            for b in blocks:
                blk = list(range(0, b))
                S, F, M = block_observables(C, blk)
                accS[b] += S
                accF[b] += F
                accM[b] += M
    for b in blocks:
        accS[b] /= nrec
        accF[b] /= nrec
        accM[b] /= nrec
    return accS, accF, accM

def sweep(L, plist, ntraj, T, T_eq, blocks, seed=0):
    rng = np.random.default_rng(seed)
    h = single_particle_h(L)
    U = _expm_herm(h)  # dt=1
    outS = {b: np.zeros(len(plist)) for b in blocks}
    outF = {b: np.zeros(len(plist)) for b in blocks}
    outM = {b: np.zeros(len(plist)) for b in blocks}
    for ip, p in enumerate(plist):
        aS = {b: 0.0 for b in blocks}
        aF = {b: 0.0 for b in blocks}
        aM = {b: 0.0 for b in blocks}
        for _ in range(ntraj):
            s, f, m = run_trajectory(L, p, U, T, T_eq, rng, blocks)
            for b in blocks:
                aS[b] += s[b]; aF[b] += f[b]; aM[b] += m[b]
        for b in blocks:
            outS[b][ip] = aS[b] / ntraj
            outF[b][ip] = aF[b] / ntraj
            outM[b][ip] = aM[b] / ntraj
    return outS, outF, outM

def _expm_herm(h):
    w, V = np.linalg.eigh(h)
    return (V * np.exp(-1j * w)) @ V.conj().T

if __name__ == "__main__":
    import time
    t0 = time.time()
    L = 32
    U = _expm_herm(single_particle_h(L))
    rng = np.random.default_rng(1)
    s, f, m = run_trajectory(L, 0.1, U, 60, 30, rng, [L // 2])
    print("timing one trajectory L=32 T=60:", round(time.time() - t0, 3), "s")
    print("S,F,M half-chain:", s[L // 2], f[L // 2], m[L // 2])
