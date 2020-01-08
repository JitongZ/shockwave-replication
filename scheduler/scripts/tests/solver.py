import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import argparse
import datetime
import numpy as np
import random
import time

import utils
from job import Job
from job_id_pair import JobIdPair
from job_table import JobTable

def print_allocation(allocation, jobs):
    n = len(jobs)
    apps = sorted(set([job.job_type for job in jobs]))
    num_variables_per_job = 1 + len(apps)
    job_ids = sorted([job.job_id for job in jobs])
    job_id_to_application = {job.job_id : job.job_type for job in jobs}
    for i in range(0, n * num_variables_per_job, num_variables_per_job):
        job_id = job_ids[i // num_variables_per_job]
        print('Job %s (%s):' % (str(job_id),
                                job_id_to_application[job_id]))
        print('\tIsolated allocation: %.5f %.5f %.5f' % (allocation[i,0],
                                                         allocation[i,1],
                                                         allocation[i,2]))
        for j in range(1, num_variables_per_job):
            print('\tAllocation with %s: %.5f %.5f %.5f' % (apps[j-1],
                                                            allocation[i+j,0],
                                                            allocation[i+j,1],
                                                            allocation[i+j,2]))


def generate_job(rng, oracle_throughputs, generate_multi_gpu_jobs=False,
                  generate_multi_priority_jobs=False,
                  run_dir='/tmp', job_template=None, job_id=None):
    """Generates a new job for simulation."""
    if job_template is None:
        job_template = rng.choice(JobTable)
    job_type = job_template.model
    run_time = 60 * (10 ** random.uniform(2, 4))
    num_steps = \
        run_time * oracle_throughputs['v100'][job_type]['null']
    assert(run_time > 0)
    assert(num_steps > 0)
    if job_template.needs_data_dir:
        command = job_template.command % (run_dir, run_dir)
    else:
        command = job_template.command % (run_dir)

    scale_factor = 1
    if generate_multi_gpu_jobs:
        # NOTE: Distribution modified for test purposes.
        r = rng.uniform(0, 1)
        if 0.2 <= r <= 0.6:
            scale_factor = 2
        elif 0.6 <= r <= 0.9:
            scale_factor = 4
        elif 0.9 <= r:
            scale_factor = 8

    priority_weight = 1.0
    if generate_multi_priority_jobs:
        # NOTE: Distribution modified for test purposes.
        r = rng.uniform(0, 1)
        if 0.0 <= r <= 0.4:
            priority_weight = 5.0

    job = Job(job_id=job_id,
              job_type=job_type,
              command=command,
              num_steps_arg=job_template.num_steps_arg,
              total_steps=num_steps,
              duration=None,
              scale_factor=scale_factor,
              priority_weight=priority_weight)

    return job

def get_job_throughputs(jobs, oracle_throughputs, worker_types):
    throughputs = {}
    for i, job in enumerate(jobs):
        throughputs[job.job_id] = {}
        for worker_type in worker_types:
            throughputs[job.job_id][worker_type] = \
                oracle_throughputs[worker_type][job.job_type]['null']
        for j, other_job in enumerate(jobs[i+1:]):
            merged_job_id = JobIdPair(job.job_id[0], other_job.job_id[0])
            throughputs[merged_job_id] = {}
            for worker_type in worker_types:
                throughputs[merged_job_id][worker_type] = \
                    oracle_throughputs[worker_type][job.job_type][other_job.job_type]
    return throughputs

def get_app_throughputs(jobs, oracle_throughputs, worker_types):
    apps = set([job.job_type for job in jobs])
    app_throughputs = {}
    for app in apps:
        app_throughputs[app] = {}
        for worker_type in worker_types:
            app_throughputs[app][worker_type] = {}
            app_throughputs[app][worker_type][None] = \
                oracle_throughputs[worker_type][app]['null']
            for other_app in apps:
                app_throughputs[app][worker_type][other_app] = \
                    oracle_throughputs[worker_type][app][other_app][0]
    return app_throughputs

def get_allocation_v1(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types, scale_factors, priority_weights,
                      flatten=True):
    throughputs = get_job_throughputs(jobs, oracle_throughputs, worker_types)
    unflattened_allocation = \
        policy.get_allocation(throughputs, scale_factors, priority_weights,
                              cluster_spec)
    if not flatten:
        return unflattened_allocation

    n = len(jobs)
    apps = sorted(set([job.job_type for job in jobs]))
    num_variables_per_job = 1 + len(apps)
    flattened_allocation = np.zeros((n * num_variables_per_job,
                                     len(worker_types)), dtype=np.float32)
    job_ids = sorted([job.job_id for job in jobs])
    for i, job_id in enumerate(job_ids):
        job_offset = i * num_variables_per_job
        for k, worker_type in enumerate(worker_types):
            flattened_allocation[job_offset,k] = \
                unflattened_allocation[job_id][worker_type]
        for j, other_job_id in enumerate(job_ids):
            if i == j:
                continue
            merged_job_id = JobIdPair(job_id[0], other_job_id[0])
            app_idx = apps.index(jobs[j].job_type)
            for k, worker_type in enumerate(worker_types):
                flattened_allocation[job_offset+1+app_idx, k] = \
                    unflattened_allocation[merged_job_id][worker_type]
    return flattened_allocation

def get_allocation_v2(policy, jobs, oracle_throughputs, cluster_spec,
                      worker_types, scale_factors, priority_weights,
                      flatten=True):
    job_id_to_application = {job.job_id : job.job_type for job in jobs}
    app_throughputs = get_app_throughputs(jobs, oracle_throughputs,
                                          worker_types)
    flattened_allocation = policy.get_allocation_v2(app_throughputs,
                                                    job_id_to_application,
                                                    scale_factors,
                                                    priority_weights,
                                                    cluster_spec)
    if flatten:
        return flattened_allocation

    # TODO: Unflatten allocation
    unflattened_allocation = {}
    return unflattned_allocation

def main(args):
    rng = random.Random()
    rng.seed(0)
    v100s, p100s, k80s = args.cluster_spec.split(':')
    cluster_spec = {
        'v100': int(v100s),
        'p100': int(p100s),
        'k80': int(k80s),
    }
    worker_types = sorted(cluster_spec.keys())
    oracle_throughputs = utils.read_all_throughputs_json(args.throughputs_file)
    jobs = []
    for i in range(args.num_jobs):
        job_template = JobTable[i % len(JobTable)]
        job_id = JobIdPair(i, None)
        jobs.append(generate_job(rng, oracle_throughputs,
                                 args.generate_multi_gpu_jobs,
                                 args.generate_multi_priority_jobs,
                                 job_template=job_template,
                                 job_id=job_id))
    policy = utils.get_policy('max_min_fairness_packed')
    scale_factors = {
        job.job_id: job.scale_factor for job in jobs
    }
    priority_weights = {
        job.job_id: job.priority_weight for job in jobs
    }
    start = datetime.datetime.now()
    v1_allocation = get_allocation_v1(policy, jobs, oracle_throughputs,
                                      cluster_spec, worker_types,
                                      scale_factors, priority_weights)
    v1_runtime = datetime.datetime.now() - start
    start = datetime.datetime.now()
    v2_allocation = get_allocation_v2(policy, jobs, oracle_throughputs,
                                      cluster_spec, worker_types,
                                      scale_factors, priority_weights)
    v2_runtime = datetime.datetime.now() - start
    print('v1 allocation:')
    print_allocation(v1_allocation, jobs)
    print('')
    print('v2 allocation:')
    print_allocation(v2_allocation, jobs)
    print('')
    print('v1 runtime:', v1_runtime.seconds + v1_runtime.microseconds / 1.0e6)
    print('v2 runtime:', v2_runtime.seconds + v2_runtime.microseconds / 1.0e6)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='')
    parser.add_argument('--num_jobs', type=int, default=10, help='Num jobs')
    parser.add_argument('--throughputs-file', type=str,
                        default=('oracle_throughputs.json'),
                        help='Oracle throughputs file')
    parser.add_argument('--generate-multi-gpu-jobs', action='store_true',
                        default=False,
                        help=('If set, generates multi-GPU jobs according to '
                              'a pre-defined distribution'))
    parser.add_argument('--generate-multi-priority-jobs', action='store_true',
                        default=False,
                        help=('If set, generates some jobs with higher priority'))
    parser.add_argument('-c', '--cluster-spec', type=str, default='3:3:3',
                        help=('Cluster specification in the form of '
                              '#v100s:#p100s:#k80s'))
    args = parser.parse_args()
    main(args)

