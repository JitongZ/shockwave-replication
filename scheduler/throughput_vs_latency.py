import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

import job
import policies
import scheduler

np.random.seed(42)

def get_policy(policy_name):
    if policy_name == "isolated":
        policy = policies.IsolatedPolicy()
    elif policy_name == "max_min_fairness":
        policy = policies.MaxMinFairnessPolicy()
    elif policy_name == "max_min_fairness_packed":
        policy = policies.MaxMinFairnessPolicyWithPacking()
    elif policy_name == "min_total_duration":
        policy = policies.MinTotalDurationPolicy()
    elif policy_name == "min_total_duration_packed":
        policy = policies.MinTotalDurationPolicyWithPacking()
    elif policy_name == "fifo":
        policy = policies.FIFOPolicy()
    else:
        raise Exception("Unknown policy!")
    return policy

def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            job_type, command, num_steps_arg, total_steps, arrival_time, scale_factor = \
                    line.split('\t')
            jobs.append(job.Job(job_id=None,
                                job_type=job_type,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=int(total_steps),
                                duration=None,
                                scale_factor=int(scale_factor)))
            arrival_times.append(int(arrival_time))
            #arrival_times.append(0)
    return jobs, arrival_times


def debug_packing_for_rounds():
    debug_traces = os.listdir('traces/generated/msr/debug')
    throughputs_file = 'combined_throughputs.json'
    cluster_spec = {'v100': 28}
    x = {}
    colors = {}
    ranges = [(80, 90)]#, (50, 80), (50, 100), (50, 90), (80, 100)]

    all_colors = np.random.rand(100)

    with open('debug_packing_for_rounds.results', 'w') as f:
        f.write('Trace,Average JCT without rounds,Average JCT with rounds,'
                'Utilization without rounds, Utilization with rounds\n')
        f.flush()
        for debug_trace in debug_traces:
            trace_path = os.path.join('traces/generated/msr/debug', debug_trace)
            all_jobs, all_arrival_times = parse_trace(trace_path)
            for (i, j) in ranges:
                #debug_trace = 'msr_debug_%d_to_%d.trace' % (i*10, (i*10+20))
                jobs = all_jobs[i:j]
                arrival_times = all_arrival_times[i:j]
                results = {}
                for use_rounds in [True, False]:
                    x[(i, j, use_rounds)] = []
                    colors[(i, j, use_rounds)] = []
                    policy = get_policy('max_min_fairness_packed')
                    results[use_rounds] = {}
                    sched = \
                        scheduler.Scheduler(policy,
                                            schedule_in_rounds=use_rounds,
                                            throughputs_file=throughputs_file,
                                            emulate=True)
                    sched.emulate(cluster_spec, arrival_times, jobs,
                                  ideal=False)
                    start_times, end_times = sched.get_job_start_and_end_times()
                    for k, (start_time, end_time) in enumerate(zip(start_times, end_times)):
                        x[(i, j, use_rounds)].append(start_time)
                        x[(i, j, use_rounds)].append(end_time)
                        colors[(i, j, use_rounds)].append(all_colors[i+k])
                        colors[(i, j, use_rounds)].append(all_colors[i+k])
                    #results[use_rounds]['utilization'] = \
                    #        sched.get_cluster_utilization()
                    #results[use_rounds]['average_jct'] = sched.get_average_jct()
                """
                f.write('%s,%3f,%.3f,%.3f,%.3f\n' % (debug_trace.split('.trace')[0],
                                                     results[False]['average_jct'],
                                                     results[True]['average_jct'],
                                                     results[False]['utilization'],
                                                     results[True]['utilization']))
                """


    for k, (i, j) in enumerate(ranges):
        for m, use_rounds in enumerate([True, False]):
            y = [k*2+m for _ in range(len(x[(i, j, use_rounds)]))]
            if use_rounds:
                label = 'Jobs %d to %d with rounds' % (i, j)
            else:
                label = 'Jobs %d to %d without rounds' % (i, j)
            plt.scatter(x[(i, j, use_rounds)], y, c=colors[(i, j, use_rounds)], label=label)
    plt.show()


def main():
    # TODO: convert to command line arguments
    schedule_in_rounds = True
    throughputs_file = 'combined_throughputs.json'
    num_v100s = 4
    policy_names = ['max_min_fairness_packed'] 
    ratios = [
            {'v100': 1, 'p100': 0, 'k80': 0},
            #{'v100': 1, 'p100': 1, 'k80': 0},
            #{'v100': 1, 'p100': 1, 'k80': 1},
            #{'v100': 2, 'p100': 1, 'k80': 0},
        ]
    lams = [16]
    for ratio in ratios:
        cluster_spec = {}
        total_gpu_fraction = sum([ratio[gpu_type] for gpu_type in ratio])
        for gpu_type in ratio:
            cluster_spec[gpu_type] = int(ratio[gpu_type] / total_gpu_fraction * num_v100s)
        output_file = ('latency_vs_throughput_cluster_ratio_'
                       '%d_%d_%d.csv') % (ratio['v100'], ratio['p100'],
                                          ratio['k80'])
        with open(output_file, 'w') as f:
            f.write('# v100,# p100,# k80,Policy,Lambda,Utilization,Average JCT\n')
            for policy_name in policy_names:
                for lam in lams:
                    # TODO: can this be moved to previous for loop?
                    policy = get_policy(policy_name)
                    trace = 'traces/generated/microbenchmark/arrival_rate_%d.trace' % (lam)
                    #trace = 'traces/generated/msr/msr_debug_truncated.trace'
                    jobs, arrival_times = parse_trace(trace)
                    sched = \
                        scheduler.Scheduler(policy,
                                            schedule_in_rounds=schedule_in_rounds,
                                            throughputs_file=throughputs_file,
                                            emulate=True)
                    sched.emulate(cluster_spec, arrival_times, jobs,
                                  ideal=False)
                    utilization = sched.get_cluster_utilization()
                    average_jct = sched.get_average_jct()
                    f.write('%d,%d,%d,%s,%d,%.3f,%.3f\n' % (cluster_spec['v100'],
                                                            cluster_spec['p100'],
                                                            cluster_spec['k80'],
                                                            policy.name,
                                                            lam,
                                                            utilization,
                                                            average_jct))
                    f.flush()


if __name__=='__main__':
    debug_packing_for_rounds()
    #main()
