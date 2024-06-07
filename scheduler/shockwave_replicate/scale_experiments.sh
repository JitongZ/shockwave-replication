#!/bin/bash

# # define paths
# label="tune1_shockwave"
# pickle_dir="results/${label}_pickle"

# # create ./results folder if not exist
# mkdir -p ./results

# # Array of num_gpus values
# num_gpus_values=("64" "128" "256")
# policies=("max_min_fairness" "shockwave")
# # num_gpus_values=("64" "128" "256")
# # policies=("shockwave")

# for num_gpus in "${num_gpus_values[@]}"; do
#   for policy_name in "${policies[@]}"; do
#     python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
#     --trace_file ../traces/reproduce/220_0.2_5_100_25_4_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic.trace \
#     --policy $policy_name \
#     --config ../configurations/scale_${num_gpus}gpus.json \
#     --throughputs_file ../wisr_throughputs.json \
#     --pickle_output_dir ../../shockwave_replicate/${pickle_dir} \
#     --cluster_spec $num_gpus:0:0 \
#     --seed 0 --solver ECOS \
#     --time_per_iteration 120 \
#     > ./results/${label}_test_shockwave_trace_${num_gpus}gpus_${policy_name}.txt 2>&1
#   done
# done

# python3 ./plot_scale_experiment.py --pickle_dir ${pickle_dir} --plot_name ${label}_gpu_metrics_plots



### 
# Create ./results folder if not exist
mkdir -p ./results

# Array of num_gpus values
num_gpus_values=("64" "128" "256")
policies=("max_min_fairness" "shockwave")

# Directory containing the trace files
trace_dir="../traces/reproduce"

# Loop through all .trace files in the directory
for trace_file in ${trace_dir}/*.trace; do
  # Extract base name of the trace file
  trace_base=$(basename "$trace_file")
  
  # Extract first four numbers for the label
  label_numbers=$(echo "$trace_base" | grep -oP '^\d+_\d+\.\d+_\d+_\d+')
  label="traces_${label_numbers}"
  pickle_dir="results/${label}_pickle"

  for num_gpus in "${num_gpus_values[@]}"; do
    for policy_name in "${policies[@]}"; do
      python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
      --trace_file "$trace_file" \
      --policy $policy_name \
      --config ../configurations/scale_${num_gpus}gpus.json \
      --throughputs_file ../wisr_throughputs.json \
      --pickle_output_dir ../../shockwave_replicate/${pickle_dir} \
      --cluster_spec $num_gpus:0:0 \
      --seed 0 --solver ECOS \
      --time_per_iteration 120 \
      > ./results/${label}_test_shockwave_trace_${num_gpus}gpus_${policy_name}.txt 2>&1
    done
  done

  python3 ./plot_scale_experiment.py --pickle_dir ${pickle_dir} --plot_name ${label}_gpu_metrics_plots
done
