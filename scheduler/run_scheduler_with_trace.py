import argparse
import datetime
import queue
import time
import datetime

import job
import policies
import scheduler

def get_policy(policy_name):
    if policy_name == "isolated":
        policy = policies.IsolatedPolicy()
    elif policy_name == "ks":
        policy = policies.KSPolicy()
    elif policy_name == "ks_packed":
        policy = policies.KSPolicyWithPacking()
    elif policy_name == "fifo":
        policy = policies.FIFOPolicy()
    elif policy_name == "max_throughput":
        policy = policies.MaximumThroughputPolicy()
    else:
        raise Exception("Unknown policy!")
    return policy

def parse_trace(trace_file):
    jobs = []
    with open(trace_file, 'r') as f:
        for line in f:
            command, num_steps_arg, total_steps, arrival_time, duration = \
                    line.split('\t')
            jobs.append((job.Job(job_id=None,
                                job_type=None,
                                command=command,
                                num_steps_arg=num_steps_arg,
                                total_steps=int(total_steps),
                                duration=int(duration)),
                        int(arrival_time)))
    return jobs

def main(args):
    jobs = parse_trace(args.trace_file)
    job_queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    policy = get_policy(args.policy)
    sched = scheduler.Scheduler(policy, job_packing=False)
    start_time = datetime.datetime.now()
    while not job_queue.empty():
        job, arrival_time = job_queue.get()
        current_time = datetime.datetime.now()
        elapsed_seconds = (current_time - start_time).seconds
        remaining_time = arrival_time - elapsed_seconds
        if remaining_time > 0:
            time.sleep(remaining_time)
        job_id = sched.add_job(job)
        print('%s] [Dispatched] Job ID: %s' % (str(datetime.datetime.now()),
                                               str(job_id)))

    sleep_seconds = 30
    while not sched.is_done():
        time.sleep(sleep_seconds)

    print("Total time taken: %d seconds" % (datetime.datetime.now() - start_time).seconds)
    s.shutdown()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run scheduler with trace')
    parser.add_argument('-t', '--trace_file', type=str, required=True,
                        help='Trace file')
    parser.add_argument('-p', '--policy', type=str, default='fifo',
                        choices=['isolated', 'ks', 'ks_packed', 'fifo',
                                 'max_throughput'],
                        help='Scheduler policy')
    main(parser.parse_args())
