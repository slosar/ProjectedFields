#!/bin/bash
# All runs behind Section 4 of the paper, followed by its figures.
# Run from the repository root.  Needs the projected maps in $PROFIELD_DATA/projected
# (see inference/tools/project_all.sh and docs/data.md).
# About 3 h on a small GPU; each 100-realization ensemble takes ~2 min on an A100.
set -e
PY=${PYTHON:-python}
RES=${PROFIELD_DATA:-data}/results
mkdir -p $RES/logs
run() { [ -f $RES/$1.npz ] || $PY inference/run_ensemble.py --tag "$@" > $RES/logs/$1.log 2>&1; }

[ -f $RES/checks.npz ] || $PY inference/model_checks.py      # model checks at the true ICs, Sigma^2 scan
[ -f $RES/free_ptot.npz ] || $PY inference/free_ptot.py      # why P_tot cannot be left free

# showcase realization (sim 88, z projection): 8 chains each, long, with stored samples
run r88_lpt1 --axis z --sims 88 --chains 8 --n-keep 2000 --save-samples 16 --lpt 1 --init prior
run r88_lpt2 --axis z --sims 88 --chains 8 --n-keep 2000 --save-samples 16 --lpt 2 --init prior
run r88_nomask_lpt1 --axis z --sims 88 --chains 8 --n-keep 2000 --mask none --lpt 1 --init prior

# ensembles: x and y projections give 200 independent Delta_0;
# z projections are 100 bulk-mode realizations of a single Delta_0
E="--n-burn 200 --n-keep 500"
for ax in x y z; do
    run ${ax}_mask_lpt1      --axis $ax --sims 0-99 --lpt 1 $E
    run ${ax}_nomask_lpt1    --axis $ax --sims 0-99 --lpt 1 --mask none $E
    run ${ax}_mask_lpt2      --axis $ax --sims 0-99 --lpt 2 $E
    run ${ax}_mask_lpt1_fsky --axis $ax --sims 0-99 --lpt 1 --mask-noise fsky $E   # mask weighting of the previous version
done
run x_mask_lpt1_pdata --axis x --sims 0-99 --lpt 1 --ptot data $E   # P_tot from the data itself

$PY inference/make_paper_figs.py | tee $RES/summary.txt
