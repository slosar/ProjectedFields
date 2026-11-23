#!/bin/bash
# Project the GADGET-4 snapshots used in Section 4 to 2D maps along x, y and z.
# Usage (from the repository root):  inference/tools/project_all.sh <gadget-run-dir>
# where <gadget-run-dir> contains output_00 ... output_99 and output_onlyPF.
# Writes $PROFIELD_DATA/projected/<name>.npz (default PROFIELD_DATA=data).
#
# CAREFUL: the output lists differ between simulations 00-49 and 50-99.
#   00-49: z~1 is snapshot_010, snapshot_000 is z=49, the z=63 ICs are snapshot_ics_000;
#   50-99: z~1 is snapshot_007, snapshot_000 is z=63.
# The loaders in inference/field_sampler.py check the redshift stored in every map.
set -e
SIMS=${1:?usage: $0 <gadget-run-dir>}
OUT=${PROFIELD_DATA:-data}/projected
PY=${PYTHON:-python}
mkdir -p $OUT
proj() { [ -f $OUT/$2.npz ] || $PY inference/tools/project_snapshot.py $SIMS/$1 $OUT/$2.npz; }

proj output_onlyPF/snapshot_ics_000.hdf5 onlyPF_ics
proj output_onlyPF/snapshot_010.hdf5 onlyPF_z1
for i in $(seq -w 0 99); do
    if [ $i -lt 50 ]; then
        proj output_$i/snapshot_010.hdf5 sim${i}_z1
        proj output_$i/snapshot_ics_000.hdf5 sim${i}_ics
    else
        proj output_$i/snapshot_007.hdf5 sim${i}_z1
        proj output_$i/snapshot_000.hdf5 sim${i}_z63
    fi
done
