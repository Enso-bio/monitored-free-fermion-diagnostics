# Number fluctuations & entanglement-spectrum participation — monitored free fermions

Theoretical quantum paper (REVTeX / APS style) with figures generated from real
correlation-matrix simulations. **Version 2** — substantially revised after a
referee-style critique (see "What changed in v2" below).

## Contents
- `main.tex` — the manuscript. Bibliography is **embedded** (`thebibliography`),
  fully self-contained, no BibTeX needed.
- `figs/` — 6 figures in PDF (for LaTeX) and PNG (quick viewing).
- `monitored_fermions.py` — the Gaussian (correlation-matrix) simulator.
- `analysis_v2.py` — **regenerates all data and figures** (~4 min, numpy + matplotlib).
- `data_v2.json` — summary statistics behind the figures.
- `pertraj_sweeps.npz` — trajectory-resolved raw data (for reproducibility).
- `production_run.py`, `data_summary.json` — v1 run (superseded, kept for history).
- `validate.py`, `explore_scaling.py`, `check_tex.py` — sanity checks.

## Compile the PDF
No LaTeX engine on this machine. Either:
1. **Overleaf:** upload `main.tex` + the `figs/` folder, Recompile. Done.
2. **Local:** `pdflatex main` twice (or `tectonic main.tex`).

Replace the placeholder author block (marked `TODO`) before any use.

## What changed in v2 (honest-claims upgrades)
The v1 draft was critiqued along referee lines; v2 addresses every substantive point:

1. **M-vs-F information test (new Fig. 5).** Per-trajectory scatter shows M is
   ~a deterministic function of F (pooled R² = 0.997). Reported as a **negative
   result**: M adds little independent information in this model. The paper is
   reframed around this finding.
2. **Statistical performance quantified (new Fig. 6).** The old unsupported
   "smoother" claim is replaced by measured coefficients of variation
   (128 trajectories): M is up to ~2× quieter than S at strong monitoring;
   F is *not* quieter than S (old half-false claim corrected).
3. **Real error bars everywhere** — standard errors over 24–128 independent
   trajectories; autocorrelation time measured (τ ≈ 4–6 steps); transient-doubling
   convergence check reported in the text.
4. **Multi-L subsystem scaling (L = 48, 64, 96) + model comparison table.**
   Pure-log vs log+linear fits with AIC: at p = 0.1 the pure log is statistically
   disfavored and α drifts with L — stated openly, supports the crossover picture.
5. **Terminology fixed:** "charge-resolved" removed (that term means
   symmetry-resolved entanglement, Goldstein–Sela); now "number-fluctuation-based".
6. **Cost claims fixed:** all three observables share the diagonalization cost;
   only F escapes it (Tr C − Tr C²) and only F is experimentally motivated.
7. **p = 0 treated as a singular limit** and excluded from scaling figures.
8. **Growth overshoot at p = 0.05 explained** (coherent quench peak at the
   ballistic time, degraded by monitoring) instead of ignored.
9. Data-availability statements unified; obsolete `showpacs` removed.

## Reproduce everything
```
python analysis_v2.py
```
Fixed seeds; deterministic output.
