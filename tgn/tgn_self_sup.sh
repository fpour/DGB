#!/bin/bash



#####################################
# commands
#####################################
n_runs=5

for data in wikipedia 
do
  for method in jodie dyrep tgn
  do
    if [ "${method}" = "tgn" ]
    then
      prefix="${method}_attn"
    else
      prefix="${method}_rnn"
    fi

    echo "****************************************************************************************************************"
    echo "dataset: $data"
    echo "prefix: $prefix"
    echo "n_runs: $n_runs"
    echo "Start Time: $(date)"
    echo "****************************************************************************************************************"

    start_time="$(date -u +%s)"

    if [ "${method}" = "tgn" ]
      then
        echo ">>> train_self_supervised; TGN; data: $data"
        python train_self_supervised.py -d $data --use_memory --prefix "$prefix" --n_runs "$n_runs" --gpu 0
      elif [ "${method}" = "jodie" ]
      then
        echo ">>> train_self_supervised; jodie_rnn; data: $data"
        python train_self_supervised.py -d $data --use_memory --memory_updater rnn --embedding_module time --prefix "$prefix" --n_runs "$n_runs" --gpu 0
      elif [ "${method}" = "dyrep" ]
      then
          echo ">>> train_self_supervised; dyrep_rnn; data: $data"
          python train_self_supervised.py -d "$data" --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix "$prefix" --n_runs "$n_runs" --gpu 0
      elif [ "${method}" = "preproc" ]
      then
          echo ">>> Preprocessing data!"
          python utils/preprocess_data.py --data $data
      else
        echo "Undefined task!"
      fi

    end_time="$(date -u +%s)"
    elapsed="$(($end_time-$start_time))"
    echo "******************************************************"
    echo "Method: $method, Data: $data: Elapsed Time: $elapsed seconds."
    echo "****************************************************************************************************************"
    echo ""
    echo ""

  done
done
