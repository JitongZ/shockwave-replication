#!/bin/bash

# Array of num_gpus values
num_gpus_values=("64" "128" "256")

for num_gpus in "${num_gpus_values[@]}"
do
  python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
  --trace_file ../traces/shockwave/220_0.2_5_100_25_4_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic.trace \
  --policy "max_min_fairness" \
  --throughputs_file ../shockwave_wisr_throughputs.json \
  --cluster_spec $num_gpus:0:0 \
  --seed 0 --solver ECOS \
  --time_per_iteration 120 \
  > ./results/test_shockwave_trace_${num_gpus}gpus.txt 2>&1
done

python3 ./plot_scale_experiment.py