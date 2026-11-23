# Data

The data are **not** part of this repository. They are available from the authors upon request.

There are three levels of data products:

1. **GADGET-4 snapshots** (HDF5): `output_NN/snapshot_XXX.hdf5` for the 100 simulations, plus
   `output_onlyPF/` and `output_zeroPF/`. Only the `z = 63` and `z ≈ 1` snapshots are needed for
   Section 4 (see the output-list warning in [simulations.md](simulations.md)).
2. **Projected maps**: `$PROFIELD_DATA/projected/<name>.npz`, made by
   `inference/tools/project_all.sh`. Each file holds `128 × 128` float32 maps `x`, `y`, `z` of the
   overdensity projected along that axis (CIC, `250 h⁻¹ Mpc`) and the `redshift`. Names:
   - `sim{NN}_z1`: simulation NN at `z ≈ 1`
   - `sim{NN}_ics` (00–49) / `sim{NN}_z63` (50–99): simulation NN at `z = 63`
   - `onlyPF_ics`, `onlyPF_z1`: the projected-modes-only run

   About 250 small files. These are all that Section 4 needs.
3. **Sampler results**: `$PROFIELD_DATA/results/<tag>.npz`, made by `inference/run_all.sh`
   (about 1 GB in total). `inference/make_paper_figs.py` makes the figures and the numbers quoted in the text from these.

`PROFIELD_DATA` defaults to `./data` (ignored by git).

Sections 2–3 used nbodykit field caches written by `profield/projectedfields.py`
(`output_fields_<prefix>.npz`, see `simulations/config.ini`). They can be regenerated from the
snapshots.
