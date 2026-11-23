#!/bin/bash

POSITIONAL_ARGS=()
NP=1
OFFSET=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--path)
            GADGET_PATH="$2"
            shift 2
            ;;
        -np|--num_procs)
            NP="$2"
            shift 2
            ;;
        -pp|--post_processing)
            PP="$2"
            shift 2
            ;;
        -r|--run_simulations)
            RUN="$2"
            shift 2
            ;;
        -o|--offset)
            OFFSET="$2"
            shift 2
            ;;
        -*|--*)
            echo "Unknown option $i"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "$POSITIONAL_ARGS[@]"

if [[ -z $GADGET_PATH ]]; then
    echo "No path to Gadget4 simulation directory provided."
    exit 1
fi

if [[ -n $RUN ]]
then
    RUN=$((RUN+OFFSET))
    cd /astro/u/khong/projfields/code/
    python set_param.py $GADGET_PATH $RUN
fi 

cd /astro/u/khong/g/$GADGET_PATH
if [[ -z $PP ]]
then
    PP=0
fi
echo "Current working directory is `pwd`"
echo "Using $NP processes."
#module load anaconda3
#conda activate bnl
if [[ -n $RUN ]]
then
    printf -v rr "%02i" $RUN
    mpiexec -np $NP ./Gadget4 param_${rr}.txt $PP
else
    mpiexec -np $NP ./Gadget4 param.txt $PP
fi

