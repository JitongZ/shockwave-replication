# python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
#   --trace_file ../traces/shockwave/220_0.2_5_100_25_4_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic.trace \
#   --policy "max_min_fairness" \
#   --throughputs_file ../shockwave_wisr_throughputs.json \
#   --cluster_spec 64:0:0 \
#   --seed 0 --solver ECOS \
#   --time_per_iteration 120 \
#   # --config ../scale_64gpus.json \
#   # --pickle_output_dir ./results/pickles \
#   > ./results/test_max_min_fairness.txt 2>&1

# python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
  # --trace_file ../traces/physical_cluster/artifact_evaluation.trace \
  # --policy "max_min_fairness" \
  # --throughputs_file ../simulation_throughputs.json \
  # --cluster_spec 64:0:0 \
  # --seed 0 --solver ECOS \
  # --time_per_iteration 120 \
  # > ./results/test_max_min_fairness.txt 2>&1

# can't use /home/cs144/cs244/gavel/scheduler/traces/physical_cluster/artifact_evaluation.trace, 
# with shockwave_wisr_throughputs.json
# can use simulation_throughputs 


# 
python3 ../scripts/drivers/simulate_scheduler_with_trace.py \
  --trace_file ../traces/shockwave/220_0.2_5_100_25_4_0,0.5,0.5_0.6,0.3,0.09,0.01_multigpu_dynamic.trace \
  --policy "max_min_fairness" \
  --throughputs_file ../shockwave_wisr_throughputs.json \
  --cluster_spec 64:0:0 \
  --seed 0 --solver ECOS \
  --time_per_iteration 120 \
  > ./results/test_shockwave_trace.txt 2>&1

 