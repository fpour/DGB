#!/bin/bash


function print_args() {
  p_arr=("$@")
  for each in "${p_arr[@]}"
  do
    echo "$each"
  done
}

echo "**********************************************************************************"
echo ">>> *** EdgeBank *** <<<"
echo "**********************************************************************************"
echo "`date`"

n_runs=5
neg_sample=rnd


for data in wikipedia
do
  for mem_mode in unlim_mem repeat_freq time_window
  do
    if [ "${mem_mode}" = "time_window" ]
    then
      for w_mode in fixed avg_reoccur
      do

        start_time="$(date -u +%s)"

        echo "================================================"
        arguments=( "ARGS:" "Data: ${data}" "Memory: ${mem_mode}" "w_mode: ${w_mode}" "n_runs: $n_runs" )
        print_args "${arguments[@]}"
        echo "================================================"

        python link_pred/edge_bank_baseline.py --data "$data" --mem_mode "$mem_mode" --w_mode "$w_mode" --n_runs "$n_runs" --neg_sample "$neg_sample"

        end_time="$(date -u +%s)"
        elapsed="$(($end_time-$start_time))"
        echo "================================================"
        arguments=( "Method: EBank" "NEG_SAMPLE: ${neg_sample}" "Data: ${data}" "mem_mode: ${mem_mode}" "w_mode: ${w_mode}" "Elapsed Time: ${elapsed} seconds" )
        print_args "${arguments[@]}"
        echo "================================================================================================"
        echo ""

      done
    else

      start_time="$(date -u +%s)"

      echo "================================================"
      arguments=( "ARGS:" "Data: ${data}" "Memory: ${mem_mode}"  "n_runs: $n_runs" )
      print_args "${arguments[@]}"
      echo "================================================"

      python link_pred/edge_bank_baseline.py --data "$data" --mem_mode "$mem_mode" --n_runs "$n_runs" --neg_sample "$neg_sample"

      end_time="$(date -u +%s)"
      elapsed="$(($end_time-$start_time))"
      echo "================================================"
      arguments=( "Method: EBank" "NEG_SAMPLE: ${neg_sample}" "Data: ${data}" "mem_mode: ${mem_mode}" "w_mode: ${w_mode}" "Elapsed Time: ${elapsed} seconds" )
      print_args "${arguments[@]}"
      echo "================================================================================================"
      echo ""

    fi
  done
done
