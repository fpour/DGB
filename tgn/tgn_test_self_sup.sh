#!/bin/bash


#####################################
# commands
#####################################
n_runs=5

for data in wikipedia
do
  for method in jodie dyrep tgn
  do
    for neg_sample in rnd
    do
      echo "****************************************************************************************************************"
      echo "dataset: $data"
      echo "method: $method"
      echo "neg_sample: $neg_sample"
      echo "n_runs: $n_runs"
      echo "Start Time: $(date)"
      echo "****************************************************************************************************************"

      start_time="$(date -u +%s)"

      if [ "${method}" = "tgn" ]
        then
          python tgn_test_trained_model_self_sup.py -d $data --use_memory --model $method --gpu 0 --neg_sample $neg_sample --n_runs $n_runs
        elif [ "${method}" = "jodie" ]
        then
          python tgn_test_trained_model_self_sup.py -d "$data" --use_memory --memory_updater rnn --embedding_module time --model $method --gpu 0 --neg_sample $neg_sample --n_runs $n_runs
        elif [ "${method}" = "dyrep" ]
        then
          python tgn_test_trained_model_self_sup.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --model $method --gpu 0 --neg_sample $neg_sample --n_runs $n_runs
        else
          echo "Undefined task!"
        fi

      end_time="$(date -u +%s)"
      elapsed="$(($end_time-$start_time))"
      echo "******************************************************"
      echo "Method: $method, NEG_SAMPLE: $neg_sample, Data: $data, Elapsed Time: $elapsed seconds."
      echo "****************************************************************************************************************"
      echo ""

    done
  done
done


