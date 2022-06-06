#!/bin/bash



start_time="$(date -u +%s)"


data="wikipedia"
mode="t"
n_runs=5


echo "****************************************************************************************************************"
echo "*** Running CAW-N method ***"
echo "dataset: $data"
echo "mode: $mode"
echo "n_runs: $n_runs"
echo "Start Time: $(date)"
echo "****************************************************************************************************************"


if [ "$mode" = "i" ]; then
  python main.py -d $data --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0 --n_runs "$n_runs" --gpu 0

elif [ "$mode" = "t" ]; then
  python main.py -d $data --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --pos_dim 172 --walk_pool attn --seed 123 --n_runs "$n_runs" --gpu 0

else
  echo "Undefined mode!"
fi



end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "****************************************************************************************************************"
echo "Total elapsed time: $elapsed seconds."



