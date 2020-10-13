#!/bin/bash
echo running L2X
echo "$@"
curr_user=$1
shift 1

if [ "$curr_user" = "mahmoud" ] ; then
    . /home/mahmoudm/anaconda3/etc/profile.d/conda.sh
    conda activate tf_new_py3
    cd l2x
elif [ "$curr_user" = " " ] ; then
    conda activate gpu_env
    cd l2x
fi

python explain.py "$@"

