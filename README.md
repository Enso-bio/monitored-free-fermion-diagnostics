# Number Fluctuations and Entanglement-Spectrum Participation in Monitored Free Fermions

This repository contains the simulation code, analysis scripts, data files, and
figures associated with the manuscript:

**Number Fluctuations and Entanglement-Spectrum Participation in Monitored Free Fermions**

The project studies finite-size diagnostics of entanglement in monitored
free-fermion chains using trajectory-resolved Gaussian correlation-matrix
simulations.

## Repository contents

- `main.tex` — manuscript source file in REVTeX/APS style. The bibliography is
  embedded using `thebibliography`, so no BibTeX file is required.
- `figs/` — figures used in the manuscript, provided in PDF format for LaTeX
  compilation and PNG format for quick viewing.
- `monitored_fermions.py` — Gaussian correlation-matrix simulator for the
  monitored free-fermion chain.
- `analysis_v2.py` — main analysis script used to regenerate the numerical data
  and figures.
- `data_v2.json` — processed summary data used in the manuscript figures.
- `pertraj_sweeps.npz` — trajectory-resolved data for the measurement-rate
  sweeps.
- `validate.py` — basic numerical validation checks.
- `explore_scaling.py` — auxiliary scaling exploration script.
- `check_tex.py` — simple manuscript consistency checks for citations, figures,
  labels, and LaTeX environments.

## Requirements

The numerical scripts require Python 3 with:

- `numpy`
- `matplotlib`

No specialized quantum-simulation package is required.

## Reproducing the numerical results

To regenerate the data and figures, run:

```bash
python analysis_v2.py
