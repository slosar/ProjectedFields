# ProjectedFields

Code accompanying

> K. Hong, R. Pund, A. Slosar,
> *Towards field-level likelihood for projected fields: Evolved projected fields from initial projected fields*
> https://arxiv.org/abs/2510.14089

The paper asks how much information about the *initial* projected (2D) matter field survives in
the *evolved* projected field. The mean evolved projected field turns out to be a 2D Lagrangian
perturbation theory (LPT) evolution of the initial projected field, damped by
`exp(-½ k² Σ²)`. Here `Σ²` measures the information lost to the bulk (non-projected) modes. Section 4 of the paper
uses this to build a field-level posterior for the initial projected field and samples it with
Hamiltonian Monte Carlo.

This repository contains everything needed to reproduce results in the paper except the simulation data, which are available upon request.

**Data.** The GADGET-4 snapshots (100 simulations plus the projected-modes-only and
no-projected-modes runs) and the projected 2D maps derived from them are available from the
authors upon request. See [docs/data.md](docs/data.md) for the expected layout.

## Contents

| Directory | What | Paper |
|---|---|---|
| `simulations/` | GADGET-4 patch for fixed projected initial modes, parameter files, output lists, batch scripts | Sec. 3 |
| `profield/` | Snapshot loading, projection and power spectra (nbodykit), 2D LPT evolution | Secs. 2–3, App. B |
| `notebooks/` | `projected_fields_tutorial.ipynb` (projected fields, power spectra, LPT damping and Σ²); `make_mask.ipynb` (the survey window) | Secs. 2–3, 4 |
| `inference/` | Differentiable 2D LPT in PyTorch, the field-level posterior, the HMC sampler, the ensemble runs and the figures of Sec. 4 | Sec. 4 |
| `docs/` | Documentation and the 2LPT derivation for an Einstein–de Sitter universe | |
| `figures/` | Output directory of `inference/make_paper_figs.py` | |

## Quick start (Section 4)

```bash
pip install -r requirements.txt            # torch, numpy, scipy, h5py, matplotlib
export PROFIELD_DATA=/path/to/data         # default: ./data

# 1. project the snapshots to 2D maps (or ask us for the projected maps directly)
inference/tools/project_all.sh /path/to/gadget/run/dir

# 2. all runs of Section 4 and its figures (about 3 h on a small GPU)
inference/run_all.sh

# or only the figures, from existing results in $PROFIELD_DATA/results
python inference/make_paper_figs.py
```

[docs/inference.md](docs/inference.md) describes the posterior and the individual scripts, and
[docs/figures.md](docs/figures.md) maps every figure of the paper to the code that makes it.

## Documentation

- [docs/simulations.md](docs/simulations.md): the simulation suite and how to rerun it
- [docs/data.md](docs/data.md): data products and directory layout
- [docs/inference.md](docs/inference.md): the field-level posterior and sampler (Sec. 4)
- [docs/figures.md](docs/figures.md): figure → script

