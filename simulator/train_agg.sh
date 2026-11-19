#!/bin/bash
# stream A / B split of training jobs for intermediate aggregation
cd /home/claude/work/wsnsim
jobs_A="0.25:0 0.25:0.9 0.5:0 1:0.9"
jobs_B="0.5:0.9 0.75:0 0.75:0.9 0:0"
J=$1; [ "$J" = A ] && L=$jobs_A || L=$jobs_B
for job in $L; do c=${job%%:*}; g=${job##*:}
  for s in 1 2 3; do
    out=models/c${c}_g${g/./}_m$s.npz
    [ -f $out ] || python3 train.py $c $g $s $out > logs/c${c}_g${g}_m$s.log 2>&1
  done
done
