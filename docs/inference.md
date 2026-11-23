# Field-level inference (Section 4)

## Posterior

Only the initial projected field `Δ₀` is sampled:

```
-ln P(Δ₀ | Δ_obs) = ½ Σ_k |Δ₀|² / P₀  +  ½ Σ_k |W·Δ_obs − W·(S ∗ Z(Δ₀))|²(k) / N(k)
N = (1 − S²) P_tot,        S = exp(−½ k² Σ²)
```

- `Z` is the 2D LPT forward model (1LPT or 2LPT). It displaces a particle sheet and deposits it with
  a differentiable CIC (`projected_likelihood_torch.py`).
- `P₀` is the linear projected power at `z = 63`, and `P_tot` is the total non-linear projected
  power. Both are fixed inputs. The fiducial `P_tot` is the mean over the suite of the x and y
  projections (`field_sampler.suite_ptot`).
- `Σ²` is fixed to 8.2 `(h⁻¹ Mpc)²` (from the scan in `model_checks.py`).
- `W` is the survey window (`inference/window_holes.npz`, `f_sky = 0.84`; made with
  `notebooks/make_mask.ipynb`). The masked residual is weighted with `N`, which is the `f_sky`
  approximation applied consistently (see Appendix C of the paper).
- Modes outside the Nyquist circle and the DC mode are zero. The field is parametrised by a
  whitened real field, so the prior is a unit Gaussian and HMC is preconditioned by it.

## Scripts

All are run from the repository root and read and write `$PROFIELD_DATA` (default `./data`).

| Script | Does | Output |
|---|---|---|
| `inference/field_sampler.py` | Posterior (`FieldPosterior`), batched HMC (`hmc`), data loaders, `P₀`/`P_tot` | library |
| `inference/projected_likelihood_torch.py` | Cosmology, growth factor, linear power, differentiable 2D LPT | library |
| `inference/model_checks.py` | Checks of the model at the true ICs (mean, residual power, Gaussianity) and the `Σ²` scan | `results/checks.npz` |
| `inference/free_ptot.py` | Shows that `P_tot` cannot be a free parameter (noise-variance collapse) | `results/free_ptot.npz` |
| `inference/run_ensemble.py` | Samples many realizations or chains; stores calibration summaries | `results/<tag>.npz` |
| `inference/make_paper_figs.py` | Figures of Section 4 and the numbers quoted in the text | `figures/*.pdf`, stdout |
| `inference/run_all.sh` | Everything above in order | |
| `inference/tools/project_snapshot.py` | One GADGET snapshot → projected maps | `projected/<name>.npz` |
| `inference/tools/project_all.sh` | All snapshots needed | `projected/` |

Main options of `run_ensemble.py`: `--axis {x,y,z}`, `--sims 0-99`, `--chains`, `--lpt {1,2}`,
`--mask {holes,none}`, `--mask-noise {plain,fsky}`, `--ptot {suite,data}`, `--n-burn`, `--n-keep`,
`--init prior`. See `python inference/run_ensemble.py --help`.

The z projections of all simulations share `Δ₀` (truth: `onlyPF_ics`). The x and y projections
have independent `Δ₀` (truth: each simulation's own `z = 63` state). This gives 200 independent
realizations for the calibration tests.

## Requirements and run time

PyTorch with a GPU is recommended, though the code also runs on a CPU. The full `run_all.sh` takes
about 3 h on an RTX 3050 laptop GPU. A 100-realization ensemble takes about 2 min on an A100.

## Conventions

- FFT normalisation: `fft2(field) · (L / N²)` everywhere.
- `projected_likelihood_torch.py` also contains `ProjectedFieldLikelihood`, the optimisation-based
  likelihood of an earlier version of the paper. It is kept for reference and is not used by the
  Section 4 scripts.
