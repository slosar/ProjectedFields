# Simulations

The suite of Section 3: 100 dark-matter-only GADGET-4 simulations with `128³` particles in a
periodic box of `250 h⁻¹ Mpc`. All of them share the same projected initial modes (`k_z = 0`)
and have different bulk modes (`k_z ≠ 0`). Two more runs isolate the two parts:

| Run | Parameter file | Initial conditions |
|---|---|---|
| `output_00` … `output_99` | `param_base.txt`, with `Seed` incremented by the run index (`set_param.py`) | fixed `k_z = 0` modes (`PFSeed`), random `k_z ≠ 0` modes (`Seed`) |
| `output_onlyPF` | `param_onlyPF.txt` (`ZeroModes 1`) | projected modes only |
| `output_zeroPF` | `param_noPF.txt` (`ZeroProjected 1`) | bulk modes only |

Cosmology: `Ω_m = 0.308`, `Ω_Λ = 0.692`, `Ω_b = 0.0482`, `h = 0.678`, `σ_8 = 0.9`, `n_s = 1`, with the
Efstathiou approximation to the linear power spectrum (`ShapeGamma 0.21`). The ICs use 2LPT and
start at `z = 63`.

## GADGET-4 modification

`simulations/gadget4/fix_projected_fields.patch` adds a compile-time option
`FIX_PROJECTED_FIELDS` and three parameters (`PFSeed`, `ZeroModes`, `ZeroProjected`) to the
N-GenIC initial-condition generator of GADGET-4. With this option the random numbers of the
`k_z = 0` modes are drawn from a generator seeded with `PFSeed`, so they are identical in all
runs. The patch applies to commit `BASE_COMMIT` of https://github.com/rugvedpund/gadget4, a
fork of the public GADGET-4.

```bash
git clone https://github.com/rugvedpund/gadget4 && cd gadget4
git checkout $(cat ../simulations/gadget4/BASE_COMMIT)
git apply ../simulations/gadget4/fix_projected_fields.patch
cp ../simulations/gadget4/Config.sh . && make
```

## Running

`simulations/run.sh` runs one simulation, and `simulations/series.job` is the HTCondor job that
runs a series of them. Both are the scripts used at the BNL SDCC cluster and contain its paths.

```bash
python set_param.py <run-dir> <index>      # writes param_<index>.txt from param_base.txt
mpiexec -np 16 ./Gadget4 param_<index>.txt
```

## Output lists: careful

The two halves of the suite were run with different output lists:

| Simulations | `z = 63` | `z ≈ 1` |
|---|---|---|
| 00–49 | `snapshot_ics_000.hdf5` (`snapshot_000` is `z = 49`) | `snapshot_010.hdf5` |
| 50–99 | `snapshot_000.hdf5` | `snapshot_007.hdf5` |

`simulations/gadget4/outputs.txt` is the list of simulations 50–99; the 20-entry list of
simulations 00–49 has not been preserved. Always check `Header/Redshift` rather than trusting
the snapshot index. The loaders in `inference/` do this.

## Analysis of the snapshots (Sections 2–3)

`profield/projectedfields.py` loads the snapshots with nbodykit, projects them, computes power
spectra and makes the panel figures. It is driven by an ini file:
`python profield/projectedfields.py simulations/config.ini` (edit the `[IO]` paths first;
section `[paper1]` is the field figure).
`notebooks/projected_fields_tutorial.ipynb` walks through the same analysis with
`profield/data.py` and `profield/lpt2d.py`: loading, 2D LPT evolution of the initial projected
field, and the measurement of the damping `Σ²(z)`.
