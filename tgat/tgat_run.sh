#!/bin/bash


prefix="TGAT"
mode="self_sup_link"
n_runs=5

for data in wikipedia 
do
  start_time="$(date -u +%s)"
  echo "****************************************************************************************************************"
  echo "*** Running tgat_run.sh: TGAT method execution ***"
  echo "dataset: $data"
  echo "prefix: $prefix"
  echo "mode: $mode"
  echo "n_runs: $n_runs"
  echo "Start Time: $(date)"
  echo "****************************************************************************************************************"

  if [ "$mode" = "self_sup_link" ]; then
    python -u learn_edge.py -d $data --bs 200 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix $prefix --n_runs $n_runs
  elif [ "$mode" = "sup_node" ]; then
    python -u learn_node.py -d $data --bs 100 --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix $prefix --n_runs $n_runs
  else
    echo "Undefined mode!"
  fi

  end_time="$(date -u +%s)"
  elapsed="$(($end_time-$start_time))"
  echo "******************************************************"
  echo "Method: $prefix, Data: $data: Elapsed Time: $elapsed seconds."
  echo "****************************************************************************************************************"
  echo ""
  echo ""

done