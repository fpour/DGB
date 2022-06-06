#!/bin/bash


#####################################
# parameters & methods
#####################################
prefix="TGAT"
n_runs=5


#####################################
# commands
#####################################

for data in wikipedia 
do
  for neg_sample in rnd
  do
    start_time="$(date -u +%s)"
    echo "****************************************************************************************************************"
    echo "*** Running tgat_run.sh: TGAT method execution ***"
    echo "dataset: $data"
    echo "prefix: $prefix"
    echo "neg_sample: $neg_sample"
    echo "n_runs: $n_runs"
    echo "Start Time: $(date)"
    echo "****************************************************************************************************************"

    python tgat_test_trained_model_learn_edge.py -d $data --uniform --n_degree 20 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix $prefix --n_runs $n_runs --neg_sample $neg_sample

    end_time="$(date -u +%s)"
    elapsed="$(($end_time-$start_time))"
    echo "******************************************************"
    echo "Method: $prefix, Data: $data: Elapsed Time: $elapsed seconds."
    echo "****************************************************************************************************************"
    echo ""
    echo ""
  done
done

