from __future__ import print_function

import heapq
import cvxpy as cp
import numpy as np
import os
# from preconditions import preconditions
import queue
import sys
import threading
import time
import datetime
import random
import math

# TODO: clean these up
from job import Job
import job_id_pair
import job_queue
from job_table import JobTable
from runtime.rpc import scheduler_server, scheduler_client
import utils

SCHEDULER_PORT = 50060
SLEEP_SECONDS = 2
INFINITY = float("inf")
DEFAULT_THROUGHPUT = INFINITY
DEFAULT_NUM_STEPS = 100     # Default number of steps in each iteration.
EMA_ALPHA = .25 # Alpha parameter for exponential moving average.
MAX_FAILED_ATTEMPTS = 5

class Scheduler:

    def __init__(self, policy, schedule_in_rounds, emulate=False,
                 throughputs_file=None, seed=42, time_per_iteration=1920):

        print('Running scheduler with policy=%s, schedule_in_rounds=%r, '
               'seed=%d, time_per_iteration=%d' % (policy.name,
                                                   schedule_in_rounds,
                                                   seed,
                                                   time_per_iteration))

        # Flag to control whether scheduling should occur in rounds.
        self._schedule_in_rounds = schedule_in_rounds

        # Flag to control whether scheduler runs in emulation mode.
        self._emulate = emulate
        # Initialize seeds.
        self._initialize_seeds(seed)
        # Initialize time in seconds each iteration should run for.
        self._time_per_iteration = time_per_iteration

        # Latest emulated timestamp.
        self._current_timestamp = 0
        # Start and last processed timestamp for each job_id.
        self._per_job_start_timestamps = {}
        self._per_job_latest_timestamps = {}
        # Job completion times.
        self._job_completion_times = {}
        # Queue of events that need to be processed at specific timestamps.
        self._event_queue = []

        # List of worker IDs.
        self._worker_ids = []
        # List of worker types.
        self._worker_types = set()
        # Mapping of worker ID to worker type, and worker type to worker ID.
        self._worker_id_to_worker_type_mapping = {}
        self._worker_type_to_worker_id_mapping = {}
        # Policy instance.
        self._policy = policy
        # Should jobs be packed.
        self._job_packing = 'Packing' in policy.name
        # RPC clients.
        self._cluster_spec = {}
        self._worker_connections = {}
        # Next job_id to assign.
        self._job_id_counter = 0
        # Next worker_id to assign.
        self._worker_id_counter = 0
        # Lock to ensure worker_id assignment is thread-safe.
        self._scheduler_lock = threading.Lock()
        # List of available worker IDs.
        self._available_worker_ids = queue.Queue()
        # Throughputs for all current incomplete applications.
        self._throughputs = {}
        # Allocations for all current incomplete applications.
        self._allocation = {}
        # Iterations run on each worker_id, for all current incomplete
        # applications.
        self._steps_run_so_far = {}
        # Total number of iterations run for each incomplete job across
        # all worker types.
        self._total_steps_run = {}
        # Time run so far on each worker_id, for all current incomplete
        # applications.
        self._job_time_so_far = {}
        # Time spent running any application on each worker, for all current
        # incomplete applications.
        self._worker_time_so_far = {}
        # Cumulative time spent running any application on each worker.
        self._cumulative_worker_time_so_far = {}
        # Number of jobs to compute fair share.
        self._num_jobs = 0
        # Commands to run for all current incomplete applications.
        self._jobs = {}
        # Priority queues for each worker_type.
        self._per_worker_type_job_queue = {}
        self._priorities = {}
        self._deficits = {}
        # Number of failures per job.
        self._num_failures_per_job = {}
        # Timestamp when data structures recording elapsed time was last reset.
        self._last_reset_time = 0
        # Flag indicating when to update the allocation.
        self._need_to_update_allocation = False
        # Throughputs for all job types (pre-measured).
        if throughputs_file is not None:
            self._all_throughputs = utils.read_all_throughputs_json(
                throughputs_file)
        else:
            self._all_throughputs = {}
        # Currently running jobs.
        self._running_jobs = set()
        # The timestamp when each worker entered the cluster.
        self._worker_start_times = {}
        # Verbose flag.
        self._verbose = False
        # Data structures for debugging.
        self._micro_tasks_per_job = {}
        self._all_jobs = []

        port = SCHEDULER_PORT
        callbacks = {
            'RegisterWorker': self._register_worker_callback,
            'Done': self._done_callback,
        }

        if not self._emulate:
            self.server_thread = threading.Thread(
                target=scheduler_server.serve,
                args=(port, callbacks))
            self.server_thread.daemon = True
            self.server_thread.start()

            self.start_scheduling_thread()


    def _initialize_seeds(self, seed):
        np.random.seed(seed)
        random.seed(seed+1)

        self._job_generator = random.Random()
        self._job_generator.seed(seed+2)

        self._interarrival_time_generator = random.Random()
        self._interarrival_time_generator.seed(seed+3)


    def start_scheduling_thread(self):
        self.scheduler_thread = threading.Thread(
            target=self.schedule,
            args=())
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()


    def _update_throughput(self, job_id, worker_type, num_steps,
                           execution_time):
        # Adjust the job throughput using an exponential moving average
        # between the old value and the new measurement.
        # TODO: fix for job pairs.
        old_throughput = self._throughputs[job_id][worker_type]
        new_throughput = num_steps / execution_time
        if old_throughput != INFINITY:
            new_throughput *= EMA_ALPHA
            new_throughput += (1 - EMA_ALPHA) * old_throughput
        self._throughputs[job_id][worker_type] = new_throughput
        print(('[DEBUG] Job %s throughput on worker type %s: '
               '%.3f -> %.3f') % (job_id, worker_type, old_throughput,
                                  self._throughputs[job_id][worker_type]))

    """
    ======================================================================
       Public-facing scheduler methods.
    ======================================================================
    """

    def add_job(self, job, timestamp=None):
        """Adds a new job to the scheduler.

        Enables users to schedule a new job. Updates the internal
        allocation of workers to jobs. An allocation is of the form
        {job: <fraction of allocations on different workers>}.

        Args:
            job: Job object to schedule. Contains information about the command
                 to run, as well as the number of steps to run the command for.
            timestamp (optional): Timestamp at which job is to be added
                                  (defaults to current_timestamp() if not
                                  specified).

        Returns:
            The job_id of the newly added job.
        """

        with self._scheduler_lock:
            current_timestamp = self.get_current_timestamp()
            job_id = job_id_pair.JobIdPair(self._job_id_counter, None)
            self._job_id_counter += 1
            job._job_id = job_id
            self._jobs[job_id] = job
            self._steps_run_so_far[job_id] = {}
            self._job_time_so_far[job_id] = {}
            self._throughputs[job_id] = {}
            self._num_failures_per_job[job_id] = 0
            self._total_steps_run[job_id] = 0
            for worker_type in self._worker_types:
                self._steps_run_so_far[job_id][worker_type] = 0
                self._throughputs[job_id][worker_type] = \
                    self._compute_throughput(job.job_type, worker_type)
                if self._job_packing:
                    self._populate_job_combination_metadata(job_id,
                                                            worker_type)
                self._job_time_so_far[job_id][worker_type] = \
                        (self._time_per_iteration / 2.0)
            self._per_job_start_timestamps[job_id] = current_timestamp
            self._per_job_latest_timestamps[job_id] = None
            if self._schedule_in_rounds:
                self._add_to_priorities(job_id)
            else:
                self._add_to_queue(job_id)
            self._need_to_update_allocation = True
            if timestamp is None:
                timestamp = self.get_current_timestamp()
            self._per_job_start_timestamps[job_id] = timestamp
            print('%s]\t[Job dispatched]\tJob ID: %s' % (timestamp, job_id))

        return job_id

    def remove_job(self, job_id):
        """Removes a job from the scheduler.

        Enables users to remove a previously scheduled job. Updates
        the internal allocation of workers to jobs.

        Args:
            job_id: The job_id of the job to remove.
        """

        job_id = job_id_pair.JobIdPair(job_id, None)
        with self._scheduler_lock:
            duration = self._per_job_latest_timestamps[job_id] - \
                self._per_job_start_timestamps[job_id]
            self._job_completion_times[job_id] = duration
            print("Job %d completed\n\tStart timestamp: %.2f\n\t"
                  "End timestamp: %.2f\nDuration: %.2f %s\n"
                  "Number of active jobs: %d\n" % (
                      job_id[0],
                      self._per_job_start_timestamps[job_id],
                      self._per_job_latest_timestamps[job_id],
                      duration, "seconds", len(self._jobs))
                  )

            del self._jobs[job_id]
            del self._steps_run_so_far[job_id]
            del self._total_steps_run[job_id]
            del self._job_time_so_far[job_id]
            del self._throughputs[job_id]
            del self._num_failures_per_job[job_id]
            if self._job_packing:
                to_delete = []
                for other_job_id in self._throughputs:
                    if (other_job_id.is_pair() and
                        job_id.overlaps_with(other_job_id)):
                        for only_other_job_id in other_job_id.singletons():
                            if only_other_job_id != job_id:
                                for worker_type in self._worker_types:
                                    self._steps_run_so_far[only_other_job_id][worker_type] += \
                                            self._steps_run_so_far[other_job_id][worker_type]
                        to_delete.append(other_job_id)
                for other_job_id in to_delete:
                    del self._throughputs[other_job_id]
                    del self._steps_run_so_far[other_job_id]
                    del self._job_time_so_far[other_job_id]

            if self._schedule_in_rounds:
                self._remove_from_priorities(job_id)
            else:
                self._remove_from_queue(job_id)

            self._need_to_update_allocation = True

    def num_workers(self):
        """Returns the number of workers the scheduler is connected to."""

        n = 0
        with self._scheduler_lock:
            for worker_type in self._cluster_spec:
                n += self._cluster_spec[worker_type]
            return n

    def is_done(self):
        """Returns whether the scheduler is done with all its assigned work."""
        with self._scheduler_lock:
            return len(self._jobs) == 0

    def shutdown(self):
        """Sends a shutdown signal to every worker and ends the scheduler."""
        with self._scheduler_lock:
            for worker_id in self._worker_connections:
                self._worker_connections[worker_id].shutdown()
        # TODO: Any other cleanup?

    """
    ======================================================================
       Scheduler's main schedule() and emulate() methods.
    ======================================================================
    """

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _schedule_job_on_worker(self, worker_id):
        """Attempts to schedule a job on worker WORKER_ID.

           Args:
            worker_id: The ID of the worker to schedule a job on.

           Returns:
            The job ID of the scheduled job, or None if no job was scheduled.
        """

        worker_type = self._worker_id_to_worker_type_mapping[worker_id]
        self._update_queue()
        if self._per_worker_type_job_queue[worker_type].size() == 0:
            self._add_available_worker_id(worker_id)
            return None

        queued_job = self._per_worker_type_job_queue[worker_type][0]
        job_id = queued_job.job_id
        priority = queued_job.priority

        # TODO: Do we need to include a check along the lines below?
        # if self._allocation[job_id][worker_type] < threshold:
        #     self._add_available_worker_id(worker_id)
        #     return None
        # threshold here is tuned according to the number of users, etc.
        # Check is meant to make sure very small allocations don't get
        # GPU time.

        for single_job_id in job_id.singletons():
            self._remove_from_queue(single_job_id)

        # Actually execute the scheduled job_id(s) on the right
        # worker_id.
        for single_job_id in job_id.singletons():
            num_steps = self._get_num_steps(job_id, worker_type, single_job_id)
            if num_steps <= 0:
                raise ValueError('num_steps should be greater'
                                 'than 0, is %d' % (num_steps))
            self._per_job_latest_timestamps[single_job_id] = \
                    self.get_current_timestamp()
        worker_types = []
        for x in self._allocation[job_id]:
            worker_types.append(x)
        worker_types = sorted(worker_types)
        allocation_str = ''
        for x in worker_types:
            allocation_str += ' [%4s %f]' % (x, self._allocation[job_id][x])
        print(('%s]\t[Micro-task scheduled]\tJob ID: %s\t'
               'Worker type: %s\tWorker ID: %d\t'
               'Priority: %f\tDeficit: %f\t'
               'Allocation:%s') % (self.get_current_timestamp(),
                                   job_id, worker_type,
                                   worker_id,
                                   priority,
                                   self._deficits[worker_type][job_id],
                                   allocation_str))

        return job_id

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper(self, worker_type,
                                         already_scheduled_jobs):
        """Solves a Knapsack-like DP problem to determine which applications /
           jobs should run on the specified worker_type in the upcoming round.

           Returns:
             A list of job IDs to schedule on the passed-in worker_type in
             the upcoming round.
        """

        # Only iterate through job_ids that haven't been scheduled yet.
        job_ids = []
        # TODO: Handle job packing.
        for job_id in self._jobs.keys():
            if job_id not in already_scheduled_jobs:
                job_ids.append(job_id)

        num_workers = len(self._worker_type_to_worker_id_mapping[worker_type])

        # DP table initialization.
        A = []
        parent_pointers = {}
        for i in range(len(job_ids)):
            job = self._jobs[job_ids[i]]
            job_id = job.job_id
            scale_factor = job.scale_factor
            A.append([])
            for j in range(num_workers):
                if (j+1) >= scale_factor:
                    A[-1].append(self._priorities[worker_type][job_id])
                else:
                    A[-1].append(0.0)

        # Solve Knapsack-like DP problem to determine which applications to
        # run on available workers of the passed-in worker_type.
        if len(A) == 0:
            return []

        for j in range(len(A[0])):
            for i in range(len(A)):
                job = self._jobs[job_ids[i]]
                job_id = job.job_id
                scale_factor = job.scale_factor
                parent_pointer = None

                # If application i is not in the optimal subset of applications
                # to run on workers of type `worker_type`.
                if i > 0 and A[i-1][j] >= A[i][j]:
                    A[i][j] = A[i-1][j]
                    parent_pointer = (i-1, j)

                # If the optimal subset of applications to run on workers of
                # type `worker_type` need only `j-1` GPUs instead of `j`.
                if j > 0 and A[i][j-1] >= A[i][j]:
                    A[i][j] = A[i][j-1]
                    parent_pointer = (i, j-1)

                # If application `i` is in the optimal subset of applications
                # to run on <= j GPUs of type `worker_type`.
                if (i > 0) and (j >= scale_factor):
                    new_priority_sum = (A[i-1][j-scale_factor] +
                        self._priorities[worker_type][job_id])
                    if new_priority_sum > A[i][j]:
                        A[i][j] = new_priority_sum
                        parent_pointer = (i-1, j-scale_factor)

                parent_pointers[(i, j)] = parent_pointer

        # Now route through parent_pointers backward to get the applications
        # that are active in this round.
        (i, j) = (len(job_ids)-1, num_workers-1)
        scheduled_jobs_on_worker_type = []
        while (i, j) in parent_pointers:
            if parent_pointers[(i, j)] is None:
                break
            (i_prime, j_prime) = parent_pointers[(i, j)]
            if (i_prime < i) and (j_prime < j):
                scheduled_jobs_on_worker_type.append((job_ids[i], j-j_prime))
            (i, j) = (i_prime, j_prime)
        scheduled_jobs_on_worker_type.append((job_ids[i], j+1))

        return scheduled_jobs_on_worker_type

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper_v2(self, worker_type,
                                            already_scheduled_jobs):
        """Greedily selects the jobs to run in the next round by iterating
           through the job list in sorted priority order.

           Assumes only single-GPU jobs.

           Returns:
             A list of job IDs to schedule on the passed-in worker_type in
             the upcoming round.
        """
        already_scheduled_jobs_set = set(already_scheduled_jobs)
        scheduled_jobs_on_worker_type = []
        num_workers = len(self._worker_type_to_worker_id_mapping[worker_type])

        entries = []
        for job_id in self._priorities[worker_type]:
            entries.append((job_id, self._priorities[worker_type][job_id],
                            self._deficits[worker_type][job_id],
                            self._allocation[job_id][worker_type]))

        sorted_job_queue = sorted(entries,
                                  key=lambda x: (x[1], x[2], x[3]),
                                  reverse=True)

        for job_id, *_ in sorted_job_queue:
            if len(scheduled_jobs_on_worker_type) == num_workers:
                break
            # Don't schedule jobs that have already been scheduled.
            if (job_id not in already_scheduled_jobs_set and
                (not job_id.is_pair() or
                 (job_id.singletons()[0] not in already_scheduled_jobs_set and
                  job_id.singletons()[1] not in already_scheduled_jobs_set))):
                # Don't schedule jobs with 0 throughput.
                if ((job_id.is_pair() and
                    (self._throughputs[job_id][worker_type][0] <= 0 or
                     self._throughputs[job_id][worker_type][1] <= 0)) or
                    (not job_id.is_pair() and
                     self._throughputs[job_id][worker_type] <= 0)):
                        continue
                else:
                    new_priority = 0
                if self._priorities[worker_type][job_id] == 0.0:
                    print('WARNING: scheduling job %s with 0 priority' % (job_id))
                already_scheduled_jobs_set.add(job_id)
                for single_job_id in job_id.singletons():
                    already_scheduled_jobs_set.add(single_job_id)
                scheduled_jobs_on_worker_type.append((job_id, 1))

        return scheduled_jobs_on_worker_type

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers_helper_v3(self, worker_type,
                                            already_scheduled_jobs):
        """Solves a Knapsack-like ILP problem to determine which applications /
           jobs should run on the specified worker_type in the upcoming round.
            Returns:
             A list of job IDs to schedule on the passed-in worker_type in
             the upcoming round.
        """

        # Get the job combinations that don't overlap with already scheduled
        # jobs.
        already_scheduled_jobs_set = set(already_scheduled_jobs)
        job_ids = []
        priorities = []
        scale_factors = []
        for job_id in self._priorities[worker_type]:
            if (job_id not in already_scheduled_jobs_set and
                (not job_id.is_pair() or
                 (job_id.singletons()[0] not in already_scheduled_jobs_set and
                  job_id.singletons()[1] not in already_scheduled_jobs_set))):
                job_ids.append(job_id)
                priorities.append(self._priorities[worker_type][job_id])
                if job_id.is_pair():
                    scale_factors.append(1)  # TODO: Fix this! Not necessarily true.
                else:
                    scale_factors.append(self._jobs[job_id].scale_factor)

        if len(job_ids) == 0:
            return []

        single_job_ids = []
        for job_id in job_ids:
            if not job_id.is_pair():
                single_job_ids.append(job_id)

        masks = []
        for single_job_id in single_job_ids:
            mask = []
            # Each mask has dimension (num_job_combinations,).
            for job_id in job_ids:
                if job_id in single_job_ids:
                    if job_id != single_job_id:
                        mask.append(0.0)
                    else:
                        mask.append(1.0)
                else:
                    if not single_job_id.overlaps_with(job_id):
                        mask.append(0.0)
                    else:
                        mask.append(1.0)
            masks.append(mask)

        priorities = np.array(priorities)
        scale_factors = np.array(scale_factors)

        x = cp.Variable(len(job_ids), boolean=True)
        objective = cp.Maximize(cp.sum(cp.multiply(priorities, x)))

        num_workers = len(self._worker_type_to_worker_id_mapping[worker_type])
        constraints = [
            cp.sum(cp.multiply(scale_factors, x)) <= num_workers,
        ]
        for mask in masks:
            constraints.append(cp.sum(cp.multiply(mask, x)) <= 1)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver='ECOS_BB')

        if cvxprob.status != "optimal":
            print('WARNING: Assignment of jobs to available workers in round not optimal!')

        scheduled_jobs_on_worker_type = []
        for i in range(len(job_ids)):
            if x.value[i] > 0.9:
                scheduled_jobs_on_worker_type.append((job_ids[i],
                                                      scale_factors[i]))

        return scheduled_jobs_on_worker_type

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _schedule_jobs_on_workers(self):
        """Attempts to schedule jobs on as many alive workers as possible.

           Returns:
             A list of job IDs and tuple of worker IDs for each scheduled job
             in the coming round.
        """
        # TODO: See if any code needs to be borrowed from _schedule_job_on_worker
        # from master.

        # Update priorities before trying to figure out applications to run
        # in the upcoming round.
        self._update_priorities()

        already_scheduled_jobs = []
        scheduled_jobs = []

        to_remove = []
        worker_types = ["v100", "p100", "k80"]
        for i, worker_type in enumerate(worker_types):
            if worker_type not in self._worker_type_to_worker_id_mapping:
                to_remove.append(i)
        for i in reversed(to_remove):
            worker_types.pop(i)

        for worker_type in worker_types:
            worker_ids = self._worker_type_to_worker_id_mapping[worker_type]
            worker_id_ptr = 0
            scheduled_jobs_on_worker_type = \
                    self._schedule_jobs_on_workers_helper_v3(
                            worker_type, already_scheduled_jobs)

            for (job_id, scale_factor) in scheduled_jobs_on_worker_type:
                # Make sure a job is only scheduled on a single worker_type in
                # a given round.
                already_scheduled_jobs.append(job_id)
                if job_id.is_pair():
                    for single_job_id in job_id.singletons():
                        already_scheduled_jobs.append(single_job_id)

                # For now, ignore locality. Place job_id on the first
                # `scale_factor` workers of the desired type.
                # assert(scale_factor == self._jobs[job_id].scale_factor)
                worker_id_ptrs = [worker_id_ptr + i for i in range(scale_factor)]
                scheduled_jobs.append((job_id,
                                       tuple([worker_ids[i] for i in worker_id_ptrs])))
                worker_id_ptr += scale_factor

                for single_job_id in job_id.singletons():
                    num_steps = self._get_num_steps(job_id, worker_type,
                                                    single_job_id)
                    if num_steps <= 0:
                        raise ValueError('Num steps should be greater'
                                         'than 0, is %d' % (num_steps))
                    self._per_job_latest_timestamps[single_job_id] = \
                        self.get_current_timestamp()
                    self._running_jobs.add(single_job_id)
                worker_types = []
                for x in self._allocation[job_id]:
                    worker_types.append(x)
                worker_types = sorted(worker_types)
                allocation_str = ''
                for x in worker_types:
                    allocation_str += ' [%4s %f]' % (x, self._allocation[job_id][x])
                print(('%s]\t[Micro-task scheduled]\tJob ID: %s\t'
                       'Worker type: %s\tWorker ID(s): %s\t'
                       'Priority: %f\tDeficit: %f\t'
                       'Allocation: %s') % (self.get_current_timestamp(),
                                           job_id, worker_type,
                                           tuple([worker_ids[i] for i in worker_id_ptrs]),
                                           self._priorities[worker_type][job_id],
                                           self._deficits[worker_type][job_id],
                                           allocation_str))
            if worker_id_ptr < len(worker_ids):
                print(('WARNING: %d GPUs of type %s left unused. '
                       'Number of active jobs: %d') % (len(worker_ids) - worker_id_ptr,
                                                       worker_type,
                                                       len(self._jobs)))

        return scheduled_jobs

    def _get_num_steps(self, job_id, worker_type, single_job_id=None):
        if job_id.is_pair():
            assert(single_job_id is not None)
            index = job_id.as_tuple().index(single_job_id[0])
            num_steps = int(self._throughputs[job_id][worker_type][index] *
                            self._time_per_iteration)
        else:
            num_steps = int(self._throughputs[job_id][worker_type] *
                            self._time_per_iteration)
        return min(num_steps,
                   self._get_remaining_steps(single_job_id))

    def _get_job_steps_and_finish_times(self, job_id, worker_type):
        """Returns the number of steps to execute and and latest finish time(s)
           for a job or job pair."""
        max_finish_time = None
        all_num_steps = []
        for i, single_job_id in enumerate(job_id.singletons()):
            num_steps = self._get_num_steps(job_id, worker_type, single_job_id)
            all_num_steps.append(num_steps)
            if job_id.is_pair():
                throughput = self._throughputs[job_id][worker_type][i]
            else:
                throughput = self._throughputs[job_id][worker_type]
            if throughput <= 0.0:
                print(single_job_id)
                print(worker_type)
                raise Exception("Throughput should not be less than 0!")
            else:
                execution_time = num_steps / throughput
                finish_time = (self.get_current_timestamp() + \
                                (num_steps / throughput))
            if (max_finish_time is None or
                finish_time > max_finish_time):
                max_finish_time = finish_time
            self._running_jobs.add(single_job_id)
        return all_num_steps, max_finish_time

    def _emulate_ideal(self, queued_jobs):
        """Emulates the passed-in policy ``ideally''.

           Determines the timestamps at which ``events'' occur --
           ``add_job'' or ``remove_job''. Now, given the time between
           these events, the amount of time each application / job spends
           on each worker type can be computed. Note that this method
           makes no determination whether such a time assignment is actually
           achievable.

           Args:
            queued_jobs: A list of jobs sorted by their arrival times
            into the cluster. All jobs in this list have not occurred
            yet.
        """

        # Find the next arrival timestamp (timestamp at which a job is to be
        # added as specified in the trace).
        next_arrival_timestamp = None
        time_to_next_arrival_timestamp = None
        if len(queued_jobs) > 0:
            next_arrival_timestamp = queued_jobs[0][0]
            time_to_next_arrival_timestamp = \
                next_arrival_timestamp - self._current_timestamp

        # Find the next departure timestamp (timestamp at which a job completes).
        # Compute departure timestamps for all jobs, and find the minimum.
        time_to_next_departure_timestamp = None
        if self._allocation is not None:
            first_job_id_to_depart = None
            for job_id in self._allocation:
                if job_id.is_pair():
                    continue
                if job_id not in self._throughputs:  # TODO: Is this needed?
                    continue
                # Effective throughput with the computed allocation, which is a
                # weighted average of the throughputs and allocations.
                steps_per_time = 0.0
                for other_job_id in self._allocation:
                    if not job_id.overlaps_with(other_job_id):
                        continue
                    for worker_type in self._allocation[other_job_id]:
                        # TODO: scale_factor needed here?
                        if other_job_id.is_pair():
                            # Determine which part of the tuple corresponds to
                            # the current job_id.
                            i = 0
                            if other_job_id.singletons()[1] == job_id:
                                i = 1
                            assert other_job_id.singletons()[i] == job_id
                            steps_per_time += (
                                self._allocation[other_job_id][worker_type] *
                                self._throughputs[other_job_id][worker_type][i])
                        else:
                            steps_per_time += (
                                self._allocation[other_job_id][worker_type] *
                                self._throughputs[other_job_id][worker_type])
                # Can now compute the finish_time for this job_id using the
                # effective throughput computed above.
                if steps_per_time == 0.0:
                    true_finish_time = INFINITY
                else:
                    true_finish_time = self._get_remaining_steps(job_id) /\
                        steps_per_time + 1
                # Only update time_to_next_departure_timestamp if earlier than
                # time_to_next_arrival_timestamp.
                if (time_to_next_departure_timestamp is None) or \
                    (time_to_next_departure_timestamp > true_finish_time):
                    if time_to_next_arrival_timestamp is None or \
                        true_finish_time < time_to_next_arrival_timestamp:
                        time_to_next_departure_timestamp = true_finish_time
                        first_job_id_to_depart = job_id

            # Now, compute the time to the next event (next_arrival_timestamp if
            # before next_departure_timestamp, otherwise next_departure_timestamp).
            next_timestamp = next_arrival_timestamp
            if time_to_next_departure_timestamp is not None:
                next_timestamp = self._current_timestamp + \
                    time_to_next_departure_timestamp
            time_to_next_timestamp = next_timestamp - self._current_timestamp

            # Update step and time counts for all jobs on the different worker types.
            steps_run = 0
            for job_id in self._allocation:
                if job_id.is_pair():
                    continue
                if job_id not in self._throughputs:
                    continue
                for other_job_id in self._allocation:
                    if not job_id.overlaps_with(other_job_id):
                        continue
                    for worker_type in self._allocation[other_job_id]:
                        if other_job_id.is_pair():
                            # Determine which part of the tuple corresponds to
                            # the current job_id.
                            i = 0
                            if other_job_id.singletons()[1] == job_id:
                                i = 1
                            assert other_job_id.singletons()[i] == job_id
                            throughput = \
                                self._throughputs[other_job_id][worker_type][i]
                        elif other_job_id in self._throughputs:
                            throughput = self._throughputs[other_job_id][worker_type]
                        else:
                            raise Exception("other_job_id should be in self._throughputs!")
                        # TODO: scale_factor needed here?
                        time_run = (time_to_next_timestamp *
                                    self._allocation[other_job_id][worker_type])
                        steps_run = time_run * throughput
                        self._steps_run_so_far[job_id][worker_type] += steps_run
                        self._total_steps_run[job_id] += steps_run
                        self._job_time_so_far[job_id][worker_type] += time_run
                        self._worker_time_so_far[worker_type] += time_run
            # TODO: Check if this is updated correctly.
            self._current_timestamp = next_timestamp

            return first_job_id_to_depart
        return None

    def _sample_arrival_time_delta(self, rate_parameter):
        """Samples job interarrival rate from a Poisson distribution according
           to the specified rate parameter."""
        return -math.log(1.0 - self._interarrival_time_generator.random()) / rate_parameter

    def _generate_job(self, fixed_job_duration=None, run_dir='/tmp'):
        """Generates a new job for emulation."""
        job_template = self._job_generator.choice(JobTable)
        job_type = job_template.model
        if fixed_job_duration:
            print('Running for fixed duration %d minutes' % (fixed_job_duration / 60.0))
            run_time = fixed_job_duration
        else:
            run_time = 60 * (10 ** self._job_generator.uniform(2, 4))
        num_steps = run_time * self._all_throughputs['v100'][job_type]['null']
        assert(run_time > 0)
        assert(num_steps > 0)
        if job_template.needs_data_dir:
            command = job_template.command % (run_dir, run_dir)
        else:
            command = job_template.command % (run_dir)

        job = Job(job_id=None,
                  job_type=job_type,
                  command=command,
                  num_steps_arg=job_template.num_steps_arg,
                  total_steps=num_steps,
                  duration=None,
                  scale_factor=1)

        return job

    def emulate(self, cluster_spec, arrival_times=None, jobs=None,
                ideal=False, lam=None, jobs_to_complete=None,
                measurement_window=None, fixed_job_duration=None,
                num_total_jobs=None, debug=False):
        """Emulates the scheduler execution.

           Emulation can be performed using a trace or with continuously
           generated synthetic data. Emulation is terminated when either
               1) All jobs in the specified trace complete.
               2) A specific subset of jobs complete.
               3) All jobs in a specific time window complete.

           Currently, the cluster specification must be statically
           specified from the beginning of execution.

           Args:
            cluster_spec: A dictionary of worker type to worker count.
            arrival_times: The arrival times of a set of pre-generated jobs.
            jobs: A set of pre-generated jobs.
            ideal: If True, emulates ideal behavior. This is only available
                   when running from a trace.
            lam: 1 / the rate parameter to be passed in to the Poisson process
                 used to generate arrival times.
            jobs_to_complete: A set of `JobIdPair`s that must be completed
                              before terminating the emulation. Mutually
                              exclusive with 'measurement_window'.
            measurement_window: A tuple specifying the beginning and ending
                                timestamps of the window in which to collect
                                jobs for the termination condition. Mutually
                                exclusive with `jobs_to_complete`.
            fixed_job_duration: If set, all generated jobs will have this
                                duration if run exclusively on a v100.
            num_total_jobs: If set, only `num_total_jobs` jobs will be generated.
            Returns:
                If `measurement_window` is specified, returns the jobs
                collected in the window. Otherwise returns None.
        """

        from_trace = arrival_times is not None and jobs is not None
        if num_total_jobs is not None:
            remaining_jobs = num_total_jobs
        if from_trace:
            remaining_jobs = len(jobs)
            queued_jobs = []
        else:
            if self._all_throughputs is None:
                raise ValueError('Scheduler must be initialized with a '
                                 'throughputs file.')
            elif lam is None:
                raise ValueError('\'lam\' must be specified when running '
                                 'without trace.')
        if ideal:
            if not from_trace:
                raise ValueError('Can only emulate in ideal mode with a trace.')
        if (jobs_to_complete is not None and
                measurement_window is not None):
            raise ValueError('Only one of \'jobs_to_complete\' or '
                             '\'measurement_window\' can be set.')
        if (not from_trace and jobs_to_complete is None and
            num_total_jobs is None and measurement_window is None):
            raise ValueError('One of \'jobs_to_complete\', '
                             '\'measurement_window\', or \'num_total_jobs\' must be set.')

        running_jobs = []
        num_jobs_generated = 0
        completed_jobs = set()
        jobs_to_measure = set()

        # Set up the cluster according to the provided spec.
        worker_types = sorted([worker_type for worker_type in cluster_spec])
        for worker_type in worker_types:
            for i in range(cluster_spec[worker_type]):
                self._register_worker_callback(worker_type)

        if from_trace:
            # Add all jobs to the queue.
            for i in range(1, len(arrival_times)):
                assert(arrival_times[i] >= arrival_times[i-1])

            for (arrival_time, job) in zip(arrival_times, jobs):
                queued_jobs.append((arrival_time, job))

        if ideal:
            self._current_timestamp = queued_jobs[0][0]

        last_job_arrival_time = None
        next_job_arrival_time = 0
        no_dispatched_or_running_jobs = False
        current_round_start_time = 0
        current_round_end_time = None
        num_completed_jobs = 0
        while True:
            if debug:
                input('Press Enter to continue...')
            if (jobs_to_complete is not None and
                  jobs_to_complete.issubset(completed_jobs)):
                break
            elif (measurement_window is not None and
                  self._current_timestamp >= measurement_window[1] and
                  jobs_to_measure.issubset(completed_jobs)):
                break
            elif (num_total_jobs is not None and
                    remaining_jobs <= 0):
                break
            elif from_trace:
                if remaining_jobs == 0:
                    break
                elif len(queued_jobs) > 0:
                    next_job_arrival_time = queued_jobs[0][0]
                else:
                    next_job_arrival_time = None

            # Jump to the next event's timestamp.
            if ideal:
                first_job_id_to_depart = self._emulate_ideal(queued_jobs)
                if first_job_id_to_depart is not None:
                    # First_job_id_to_depart should have no steps remaining.
                    assert(self._get_remaining_steps(first_job_id_to_depart) /
                        self._jobs[first_job_id_to_depart].total_steps <= 0.01)
                    self._per_job_latest_timestamps[first_job_id_to_depart] = \
                        self.get_current_timestamp()
                    # Remove job and update remaining_jobs counter.
                    self.remove_job(first_job_id_to_depart[0])
                    remaining_jobs -= 1

            else:
                if self._schedule_in_rounds:
                    # If scheduling in rounds, find the time when the latest job
                    # completes, which signals the finishing of the round.
                    max_timestamp = 0
                    if (len(running_jobs) > 0 and
                        -running_jobs[0][0] > max_timestamp):
                        max_timestamp = -running_jobs[0][0]
                        if current_round_end_time is not None:
                            current_round_start_time = current_round_end_time
                        current_round_end_time = max_timestamp
                    if max_timestamp > 0:
                        self._current_timestamp = max_timestamp
                    else:
                        self._current_timestamp = next_job_arrival_time
                else:
                    # Otherwise, find the time when the first job completes,
                    # which signals that a worker is available.
                    min_timestamp = INFINITY
                    if (len(running_jobs) > 0 and
                        running_jobs[0][0] < min_timestamp):
                        min_timestamp = running_jobs[0][0]
                    if (next_job_arrival_time is not None and
                        next_job_arrival_time < min_timestamp):
                        min_timestamp = next_job_arrival_time
                    if min_timestamp is not INFINITY:
                        self._current_timestamp = min_timestamp
                        no_dispatched_or_running_jobs = False
                    else:
                        if no_dispatched_or_running_jobs:
                            print('ERROR: No newly dispatched or running jobs!')
                            print('%d remaining jobs' % (remaining_jobs))
                            print('Completed jobs:')
                            for i, job_id in enumerate(sorted(completed_jobs)):
                                print('\t%d) %s' % (i+1, job_id))
                            print('Per worker type queues:')
                            for worker_type in self._per_worker_type_job_queue:
                                print(worker_type + ':')
                                for jqe in self._per_worker_type_job_queue[worker_type]:
                                    allocation = self._allocation[jqe.job_id][worker_type]
                                    print('%s: Allocation: %10f\t'
                                          'Priority: %10f' % (jqe.job_id,
                                                              allocation,
                                                              jqe.priority))
                            sys.exit(-1)
                        else:
                            no_dispatched_or_running_jobs = True

                # Check if any jobs have completed.
                while len(running_jobs) > 0:
                    (finish_time, job_id, worker_ids, all_num_steps) = \
                            running_jobs[0]
                    if self._schedule_in_rounds:
                        finish_time = (-finish_time)
                    if finish_time <= self._current_timestamp:
                        all_execution_times = []
                        for single_job_id in job_id.singletons():
                            if self._schedule_in_rounds:
                                start_time = current_round_start_time
                            else:
                                start_time = \
                                    self._per_job_latest_timestamps[single_job_id]
                            execution_time = finish_time - start_time
                            all_execution_times.append(execution_time)
                            self._per_job_latest_timestamps[single_job_id] = \
                                    finish_time
                        # TODO: decide whether to pass in all worker_ids to
                        # _done_callback.
                        for worker_id in worker_ids:
                            self._done_callback(job_id, worker_id,
                                                all_num_steps,
                                                all_execution_times)
                        for single_job_id in job_id.singletons():
                            if single_job_id not in self._jobs:
                                completed_jobs.add(single_job_id)
                                if from_trace or num_total_jobs is not None:
                                    remaining_jobs -= 1
                        heapq.heappop(running_jobs)
                    else:
                        break

                if self._schedule_in_rounds:
                    # Since we're scheduling in rounds, no jobs should be
                    # running when scheduling the next round of jobs.
                    assert(len(running_jobs) == 0)

            # Dispatch any newly arrived jobs.
            if from_trace:
                while len(queued_jobs) > 0:
                    (arrival_time, job) = queued_jobs[0]
                    if arrival_time <= self._current_timestamp:
                        job_id = self.add_job(job, timestamp=arrival_time)
                        queued_jobs.pop(0)
                    else:
                        break
            else:
                while next_job_arrival_time <= self._current_timestamp:
                    if num_total_jobs is not None:
                        if num_jobs_generated > num_total_jobs:
                            break
                    job = self._generate_job(fixed_job_duration=fixed_job_duration)
                    num_jobs_generated += 1
                    self._all_jobs.append((next_job_arrival_time, job))
                    job_id = self.add_job(job, timestamp=next_job_arrival_time)
                    if (measurement_window is not None and
                        next_job_arrival_time >= measurement_window[0] and
                        next_job_arrival_time <= measurement_window[1]):
                        jobs_to_measure.add(job_id)

                    last_job_arrival_time = next_job_arrival_time
                    if lam == 0.0:
                        arrival_time_delta = 0.0
                    else:
                        arrival_time_delta = \
                                self._sample_arrival_time_delta(1.0 / lam)
                    next_job_arrival_time = \
                            arrival_time_delta + last_job_arrival_time

            if not ideal:
                # Schedule jobs until there are no available workers or no jobs
                # with non-zero allocations on available workers.
                if self._schedule_in_rounds:
                    #TODO: Handle packing and multiple jobs.
                    scheduled_jobs = self._schedule_jobs_on_workers()
                    for (job_id, worker_ids) in scheduled_jobs:
                        worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
                        for worker_id in worker_ids:
                            self._remove_available_worker_id(worker_id)
                        all_num_steps, max_finish_time = \
                                self._get_job_steps_and_finish_times(job_id,
                                                                     worker_type)
                        heapq.heappush(running_jobs, (-max_finish_time, job_id,
                                                      worker_ids,
                                                      all_num_steps))
                else:
                    seen_worker_ids = set()
                    while True:
                        worker_id = self._remove_available_worker_id()
                        if worker_id in seen_worker_ids:
                            self._add_available_worker_id(worker_id)
                            break
                        elif worker_id is None:
                            break
                        else:
                            seen_worker_ids.add(worker_id)

                        job_id = self._schedule_job_on_worker(worker_id)
                        if job_id is None:
                            continue

                        worker_type = self._worker_id_to_worker_type_mapping[worker_id]
                        all_num_steps, max_finish_time = \
                                self._get_job_steps_and_finish_times(job_id,
                                                                     worker_type)
                        heapq.heappush(running_jobs, (max_finish_time, job_id,
                                       (worker_id,), all_num_steps))

        print('Total duration: %.3f seconds' % (self._current_timestamp))
        if measurement_window is not None:
            return jobs_to_measure
        else:
            return None


    def _schedule_without_rounds(self):
        """Schedules jobs on workers without rounds.

        In a loop, schedules the inactive application most in need of an
        available worker (that is, the application with the highest
        fraction_allocated/fraction_run ratio).

        Scheduler holds two internal data structures,
        {(application, worker_type): time_run_on_worker}
        & {(application, worker_type): allocation_fraction}.
        As an algorithmic optimization, the scheduler maintains
        a heap of all currently inactive applications for each
        worker, sorted by fraction_run/fraction_allocated ratio.
        """

        while True:
            worker_id = self._remove_available_worker_id()
            with self._scheduler_lock:
                job_id = self._schedule_job_on_worker(worker_id)
                if job_id is None:
                    continue
                worker_type = self._worker_id_to_worker_type_mapping[worker_id]
                # TODO: Support packing.
                num_steps = self._get_num_steps(job_id, worker_type)
                self._worker_connections[worker_id].run(
                    [(job_id[0], self._jobs[job_id].command,
                      self._jobs[job_id].num_steps_arg,
                      num_steps)])

    def _schedule_with_rounds(self):
        """Schedules jobs on workers using rounds.

        In a loop, schedules in rounds the applications most in need of
        being run (that is, the applications with the highest
        fraction_allocated/fraction_run ratio) using a DP algorithm.
        """

        while True:
            with self._scheduler_lock:
                num_workers = len(self.worker_ids)
                # Reset available_worker_ids to the desired size.
                self._available_worker_ids = queue.Queue(self.num_workers)
                for worker_id in self.worker_ids:
                    self._add_available_worker_id(worker_id)
                scheduled_jobs = self._schedule_jobs_on_workers()
                for (job_id, worker_ids) in scheduled_jobs:
                    worker_type = self._worker_id_to_worker_type_mapping[worker_ids[0]]
                    # TODO: Support packing.
                    num_steps = self._get_num_steps(job_id, worker_type)
                    for worker_id in worker_ids:
                        self._worker_connections[worker_id].run(
                            [(job_id[0], self._jobs[job_id].command,
                              self._jobs[job_id].num_steps_arg,
                              num_steps)])
                        self._remove_available_worker_id(worker_id)
            self._wait_until_all_workers_available(num_workers)

    def schedule(self):
        """Schedules jobs on workers."""
        if self._schedule_in_rounds:
            self._schedule_with_rounds()
        else:
            self._schedule_without_rounds()


    def get_average_jct(self, job_ids=None):
        """Computes the average job completion time.

           Args:
               job_ids: A list of `JobIdPair` objects. If specified, computes
                        the average JCT using only these jobs.

           Returns: The average JCT.
        """
        with self._scheduler_lock:
            if len(self._job_completion_times) == 0:
                return
            if job_ids is None:
                job_ids = sorted([job_id for job_id in self._job_completion_times])
            print('Job completion times:')
            for job_id in job_ids:
                print('Job %s: %.3f' % (job_id,
                                        self._job_completion_times[job_id]))
            average_job_completion_time = \
                np.mean([self._job_completion_times[job_id] for job_id in job_ids])
            print('Average job completion time: '
                  '%.3f seconds' % (average_job_completion_time))
            return average_job_completion_time


    def get_cluster_utilization(self):
        """Computes the utilization of the cluster."""
        with self._scheduler_lock:
            utilizations = []
            current_timestamp = self.get_current_timestamp()
            for worker_id in self._cumulative_worker_time_so_far:
                total_runtime = (current_timestamp -
                                 self._worker_start_times[worker_id])
                worker_time = self._cumulative_worker_time_so_far[worker_id]
                utilization = worker_time / total_runtime
                if utilization > 1.0 and not self._job_packing:
                    print('Error: invalid utilization %.3f' % (utilization))
                    print('Worker ID: %d' % (worker_id))
                    print('Worker time: %.3f' % (worker_time))
                    print('Total time: %.3f.' % (total_runtime))
                    return None
                utilizations.append(utilization)
            cluster_utilization = np.mean(utilizations)
            print('Cluster utilization: %.3f' % (cluster_utilization))
            return cluster_utilization

    def get_micro_tasks(self):
        """Prints all micro-tasks run for each job.

           Debug function used print all micro-tasks run for each job.
        """
        job_ids = sorted(self._micro_tasks_per_job.keys())
        for job_id in job_ids:
            print('Job %s: %d' % (job_id, len(self._micro_tasks_per_job[job_id])))
            for i, (start, end) in enumerate(self._micro_tasks_per_job[job_id]):
                print('\t%d%f - %f' % (i, start, end))
            print('')

    def get_job_start_and_end_times(self):
        """Returns the start and end times of each job.

           Debug function for returning the start and end times of each job.
        """
        with self._scheduler_lock:
            job_ids = sorted(
                [job_id for job_id in self._per_job_latest_timestamps])
            start_times = [
                self._per_job_start_timestamps[job_id]
                for job_id in job_ids]
            end_times = [
                self._per_job_latest_timestamps[job_id]
                for job_id in job_ids]
        return start_times, end_times

    def get_all_emulated_jobs(self, job_range):
        """Returns all the jobs run during emulation.

           Debug function used to print all jobs generated during
           emulation within a specified range.

           Args:
               job_range: A tuple specifying which jobs to be printed.
        """
        print('All emulated jobs:')
        for arrival_time, job in self._all_jobs[job_range[0]:job_range[1]]:
            print('%s\t%s\t%d\t%f' % (job.job_id,
                                      job.job_type,
                                      job.total_steps,
                                      arrival_time))
    """
    ======================================================================
       Helper methods to get and mutate state needed for scheduling.
    ======================================================================
    """

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _print_allocation(self):
        """Prints the allocation.

           Debug method used for printing the allocation of each job on each
           worker type.
        """
        print('')
        print('=' * 80)
        print('Allocation\t(Current_time: %f)' % (self.get_current_timestamp()))
        print('-' * 80)
        for job_id in sorted(list(self._allocation.keys())):
            allocation_str = 'Job ID %s:' % (job_id)
            for worker_type in sorted(list(self._allocation[job_id].keys())):
                allocation = self._allocation[job_id][worker_type]
                allocation_str += ' [%s: %f]' % (worker_type, allocation)
            print(allocation_str)
        print('=' * 80)
        print('')

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _print_deficits(self):
        """Prints the deficit.

           Debug method used for printing the deficit of each job on each
           worker type.
        """
        print('')
        print('=' * 80)
        print('Deficits\t(Current_time: %f)' % (self.get_current_timestamp()))
        print('-' * 80)
        for job_id in sorted(list(self._jobs.keys())):
            deficit_str = 'Job ID %s:' % (job_id)
            for worker_type in sorted(self._worker_types):
                deficit = self._deficits[worker_type][job_id]
                deficit_str += ' [%s: %f]' % (worker_type, deficit)
            print(deficit_str)
        print('=' * 80)
        print('')

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _get_allocation(self):
        """Computes the allocation.

        Uses the specified policy to compute an allocation of jobs to
        compute resources. Requires self._scheduler_lock to be held
        when calling this function.

        Returns:
            A 2-level dict indexed by job_id and then worker_type. For
            example,

            {0: {"v100": 0.25, "p100": 0.95}, 1: {"v100": 0.75, "p100": 0.05}}

            indicates that for 25% of the time, worker type 'v100' should run,
            job 0 and for 95% of the time, worker type 'p100' should run job 0.
        """

        if self._policy.name.startswith("MinTotalDuration"):
            # TODO: Need to fix this for packed policies.
            num_steps_remaining = {
                job_id: self._get_remaining_steps(job_id)
                for job_id in self._jobs}
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, num_steps_remaining, self._cluster_spec)
        else:
            unflattened_allocation = self._policy.get_allocation(
                self._throughputs, self._cluster_spec)
        if unflattened_allocation is None:
            return None
        for job_id in unflattened_allocation:
            for worker_type in unflattened_allocation[job_id]:
                threshold = float(len(self._worker_type_to_worker_id_mapping[worker_type])) / \
                    (len(self._jobs) * 1000.0)
                if unflattened_allocation[job_id][worker_type] < threshold:
                    unflattened_allocation[job_id][worker_type] = 0.0

        return unflattened_allocation

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _populate_job_combination_metadata(self, job_id, worker_type):
        """Populate metadata for job combinations involving passed-in job_id."""

        job = self._jobs[job_id]
        for other_job_id in self._jobs:
            if other_job_id != job_id:
                other_job = self._jobs[other_job_id]
                merged_job_id = \
                        job_id_pair.JobIdPair(job_id[0], other_job_id[0])
                if merged_job_id not in self._throughputs:
                    self._throughputs[merged_job_id] = {}
                    self._steps_run_so_far[merged_job_id] = {}
                    self._job_time_so_far[merged_job_id] = {}
                    self._priorities[worker_type][job_id] = 0.0
                    self._deficits[worker_type][job_id] = 0.0
                # The single-job IDs for job pairs are stored in sorted order,
                # so make sure the co-located throughputs match the order of the
                # single-job IDs.
                if job_id [0] == merged_job_id[0]:
                    self._throughputs[merged_job_id][worker_type] = \
                        self._compute_throughput(
                            [job.job_type, other_job.job_type],
                            worker_type)
                else:
                    self._throughputs[merged_job_id][worker_type] = \
                        self._compute_throughput(
                            [other_job.job_type, job.job_type],
                            worker_type)

                self._steps_run_so_far[merged_job_id][worker_type] = 0

    def _compute_throughput(self, job_types, worker_type):
        if isinstance(job_types, list):
            if self._emulate:
                return self._all_throughputs[worker_type][job_types[0]][job_types[1]]
            else:
                return (DEFAULT_THROUGHPUT / 2.0, DEFAULT_THROUGHPUT / 2.0)
        else:
            if self._emulate:
                return self._all_throughputs[worker_type][job_types]["null"]
            else:
                return DEFAULT_THROUGHPUT

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _reset_time_run_so_far(self):
        """Reset _time_run_so_far so that all jobs receive new fair allocation
        from here on out.

        Requires self._scheduler_lock to be held when calling this function.
        """
        current_time = self.get_current_timestamp()
        elapsed_time_since_last_reset = current_time - self._last_reset_time
        for worker_type in self._worker_types:
            self._worker_time_so_far[worker_type] = 0.0
            for job_id in self._job_time_so_far:
                # _job_time_so_far keeps track of how long job_id has run on
                # worker_type since the last reset event.
                if worker_type not in self._job_time_so_far[job_id]:
                    time_received = 0.0
                else:
                    # Ignore the initial time recorded for the job.
                    time_received = \
                            (self._job_time_so_far[job_id][worker_type] -
                             (self._time_per_iteration / 2.0))

                # Compute the time this job_id should have received since the
                # last reset event.
                if self._allocation is None or job_id not in self._allocation:
                    time_should_have_received = 0
                else:
                    time_should_have_received = \
                            self._allocation[job_id][worker_type] *\
                                elapsed_time_since_last_reset

                # deficit is now just the difference between the time job_id
                # should have received, and how much it actually received.
                deficit = time_should_have_received - time_received
                if job_id not in self._deficits[worker_type]:
                    self._deficits[worker_type][job_id] = 0.0
                self._deficits[worker_type][job_id] += deficit

                self._job_time_so_far[job_id][worker_type] = \
                        (self._time_per_iteration / 2.0)
                self._worker_time_so_far[worker_type] += \
                        self._job_time_so_far[job_id][worker_type]
        # Prints deficits every time allocation is reset.
        # self._print_deficits()
        self._last_reset_time = current_time

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _add_to_queue(self, job_id, worker_type=None):
        """Adds a job_id to each worker's queue.
        NOTE: Used when scheduling is not performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        worker_types = self._worker_types
        if worker_type is not None:
            worker_types = [worker_type]
        for worker_type in worker_types:
            self._per_worker_type_job_queue[worker_type].add_job(
                                                        priority=INFINITY,
                                                        allocation=0.0,
                                                        steps_run=0,
                                                        job_id=job_id)
            for other_job_id in self._throughputs:
                if (other_job_id.is_pair() and
                    job_id.overlaps_with(other_job_id)):
                    add_job = True
                    for single_job_id in other_job_id.singletons():
                        if single_job_id in self._running_jobs:
                            add_job = False
                            break
                    if add_job:
                        self._per_worker_type_job_queue[worker_type].add_job(
                                    priority=INFINITY,
                                    allocation=0.0,
                                    steps_run=0,
                                    job_id=other_job_id)

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _add_to_priorities(self, job_id, worker_type=None):
        """Adds a job_id to each worker type's priority list.
        NOTE: Used when scheduling is performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        worker_types = self._worker_types
        if worker_type is not None:
            worker_types = [worker_type]
        for worker_type in worker_types:
            self._priorities[worker_type][job_id] = 0.0
            self._deficits[worker_type][job_id] = 0.0
            for other_job_id in self._throughputs:
                if (other_job_id.is_pair() and
                    job_id.overlaps_with(other_job_id)):
                    self._priorities[worker_type][other_job_id] = 0.0
                    self._deficits[worker_type][other_job_id] = 0.0

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _remove_from_queue(self, job_id):
        """Removes a job_id from each worker's queue.
        NOTE: Used when scheduling is not performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' queues.
        """
        for worker_type in self._worker_types:
            while True:
                found = False
                for i in range(self._per_worker_type_job_queue[worker_type].size()):
                    queued_job = self._per_worker_type_job_queue[worker_type][i]
                    if job_id.overlaps_with(queued_job.job_id):
                        if self._per_worker_type_job_queue[worker_type].size() > 0:
                            self._per_worker_type_job_queue[worker_type].pop(i)
                            found = True
                        break
                if not found:
                    break

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _remove_from_priorities(self, job_id):
        """Removes a job_id from each worker type's priority list.
        NOTE: Used when scheduling is performed in rounds.

        Requires self._scheduler_lock to be held when calling this function.

        Args:
           job_id: The job_id to remove from the workers' queues.
        """
        for worker_type in self._worker_types:
            while True:
                found = False
                for other_job_id in self._priorities[worker_type]:
                    if job_id.overlaps_with(other_job_id):
                        del self._priorities[worker_type][other_job_id]
                        del self._deficits[worker_type][other_job_id]
                        found = True
                        break
                if not found:
                    break

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _update_queue(self):
        """Updates each per-worker queue.

        Re-sorts the queue of each worker to compute the next job to run.
        For a given worker w_i, the next job to be scheduled will be the job
        that has so far received the smallest fraction of its computed
        fair allocation.
        Requires self._scheduler_lock to be held when calling this function.

        NOTE: Used when scheduling is not performed in rounds.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            for job_id in self._job_time_so_far:
                fraction = self._job_time_so_far[job_id][worker_type] / \
                        self._worker_time_so_far[worker_type]
                fractions[worker_type][job_id] = fraction
            for i in range(self._per_worker_type_job_queue[worker_type].size()):
                queued_job = self._per_worker_type_job_queue[worker_type][i]
                job_id = queued_job.job_id
                # If allocation for this job_id and worker_type is 0, or
                # if the throughput of this job_id pair on this worker_type is
                # 0, give this job_id pair the lowest possible priority.
                if self._allocation[job_id][worker_type] == 0.0 or \
                    self._throughputs[job_id][worker_type] == 0.0 or \
                    self._throughputs[job_id][worker_type] == [0.0, 0.0]:
                    new_priority = INFINITY
                    steps_run = None
                else:
                    new_priority = fractions[worker_type][job_id] /\
                            self._allocation[job_id][worker_type]
                    steps_run = self._steps_run_so_far[job_id][worker_type]
                # Use negative deficit and allocation here to sort in
                # order of highest deficit -> lowest deficit followed by
                # highest allocation -> lowest allocation.
                self._per_worker_type_job_queue[worker_type].update_entry(
                        i, priority=new_priority,
                        deficit=-self._deficits[worker_type][job_id],
                        allocation=-self._allocation[job_id][worker_type],
                        steps_run=steps_run)
            self._per_worker_type_job_queue[worker_type].heapify()

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _update_priorities(self):
        """Updates each per-worker queue.

        Re-sorts the queue of each worker to compute the next job to run.
        For a given worker w_i, the next job to be scheduled will be the job
        that has so far received the smallest fraction of its computed
        fair allocation.
        Requires self._scheduler_lock to be held when calling this function.

        NOTE: Used when scheduling is performed in rounds.

        Args:
            job_id: The job_id to add to the workers' queues.
        """

        if self._need_to_update_allocation:
            self._reset_time_run_so_far()
            self._allocation = self._get_allocation()
            self._need_to_update_allocation = False

        # Stores the fraction of time spent running a job for each worker.
        fractions = {}

        for worker_type in self._worker_types:
            fractions[worker_type] = {}
            for job_id in self._job_time_so_far:
                fraction = self._job_time_so_far[job_id][worker_type] / \
                         self._worker_time_so_far[worker_type]
                fractions[worker_type][job_id] = fraction
            for job_id in self._priorities[worker_type]:
                # Don't use inf so 2*new_priority > new_priority.
                #
                # Scale the default value by the allocation so that newly
                # added jobs run according to their respective allocations.
                new_priority = self._allocation[job_id][worker_type] * 1e9
                if self._allocation[job_id][worker_type] == 0.0:
                    assert(new_priority == 0)
                elif ((job_id.is_pair() and
                       (self._throughputs[job_id][worker_type][0] == 0 or
                        self._throughputs[job_id][worker_type][1] == 0)) or
                      (not job_id.is_pair() and
                       self._throughputs[job_id][worker_type] == 0)):
                    new_priority = 0
                elif fractions[worker_type][job_id] > 0.0:
                    new_priority = self._allocation[job_id][worker_type] /\
                            fractions[worker_type][job_id]
                self._priorities[worker_type][job_id] = new_priority

    def _add_available_worker_id(self, worker_id):
        """Adds a worker_id to the list of available workers."""

        self._available_worker_ids.put(worker_id)

    def _remove_available_worker_id(self, worker_id=None):
        """Returns the worker_id of the next available worker."""

        if self._emulate:
            try:
                return self._available_worker_ids.get_nowait()
            except queue.Empty as e:
                return None
        else:
            return self._available_worker_ids.get()

    # @preconditions(lambda self: self._scheduler_lock.locked())
    def _get_highest_priority(self, job_id):
        priorities = []
        for timestamp, worker_id in self._available_worker_ids.queue:
            if timestamp > self._per_job_latest_timestamps.get(job_id, 0):
                continue
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            for i in range(self._per_worker_type_job_queue[worker_type].size()):
                queued_job = self._per_worker_type_job_queue[worker_type][i]
                if queued_job.job_id == job_id:
                    priorities.append((queued_job.priority, worker_id,
                                       worker_type))
        priorities.sort(key=lambda x: x[0])
        if len(priorities) == 0:
            return INFINITY, None
        priority = priorities[0][0]
        worker_id = priorities[0][1]
        return priority, worker_id

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def _get_remaining_steps(self, job_id):
        steps_run_so_far = self._total_steps_run[job_id]
        return self._jobs[job_id].total_steps - steps_run_so_far

    # @preconditions(lambda self: self._emulate or self._scheduler_lock.locked())
    def get_current_timestamp(self):
        if self._emulate:
            return self._current_timestamp
        else:
            return time.time()

    """
    ======================================================================
       Callback methods called by workers.
    ======================================================================
    """

    def _register_worker_callback(self, worker_type, ip_addr=None, port=None):
        """Registers a worker with the scheduler.

        Initializes state for a new worker and assigns it an id.
        The worker provides an IP address and port for its RPC server
        so that the scheduler can establish an RPC client for
        scheduler-to-worker communication. The worker also
        enumerates its available devices so that the scheduler
        can make fine-grained scheduling decisions.

        Args:
            ip_addr: IP address of the worker's RPC server.
            port: Port number for the worker's RPC server.
            devices: List of available devices on the worker.

        Returns:
            The worker_id of the newly registered worker.
        """

        with self._scheduler_lock:
            worker_id = self._worker_id_counter
            self._worker_ids.append(worker_id)
            self._worker_id_counter += 1
            self._worker_types.add(worker_type)
            self._worker_id_to_worker_type_mapping[worker_id] = worker_type
            self._cumulative_worker_time_so_far[worker_id] = 0.0
            found = True
            if worker_type not in self._worker_type_to_worker_id_mapping:
                found = False
                self._worker_type_to_worker_id_mapping[worker_type] = []
            self._worker_type_to_worker_id_mapping[worker_type].append(worker_id)

            if not found:
                self._per_worker_type_job_queue[worker_type] = \
                        job_queue.JobQueue()
                self._priorities[worker_type] = {}
                self._deficits[worker_type] = {}
                for job_id in self._jobs:
                    self._steps_run_so_far[job_id][worker_type] = 0
                    self._job_time_so_far[job_id][worker_type] = \
                            (self._time_per_iteration / 2.0)
                    self._throughputs[job_id][worker_type] = \
                        self._compute_throughput(self._jobs[job_id],
                                                 worker_type)
                    if self._job_packing:
                        self._populate_job_combination_metadata(job_id,
                                                                worker_type)

                    self._initialize_num_steps_per_iteration(job_id, worker_type)
                    # Add to relevant priority data structure.
                    if self._schedule_in_rounds:
                        self._add_to_priorities(job_id, worker_type=worker_type)
                    else:
                        self._add_to_queue(job_id, worker_type=worker_type)
                if worker_type not in self._worker_time_so_far:
                    self._worker_time_so_far[worker_type] = 0.0

            self._add_available_worker_id(worker_id)

            if worker_type not in self._cluster_spec:
                self._cluster_spec[worker_type] = 0
            self._cluster_spec[worker_type] += 1
            if not self._emulate:
                self._worker_connections[worker_id] = \
                    scheduler_client.SchedulerRpcClient(ip_addr, port)

            self._worker_start_times[worker_id] = self.get_current_timestamp()
            self._need_to_update_allocation = True

        return worker_id

    def _done_callback(self, job_id, worker_id, all_num_steps,
                       all_execution_times):
        """Handles completion of a scheduled job.

        Updates the running total of completed steps and time spent on each
        worker, for every currently active application. Removes the job from
        the scheduler if the job has finished all its requested steps. Adds
        the worker back to the list of available workers.

        Args:
            job_id: The id of the completed job(s).
            worker_id: The id of the worker where the job(s) were completed.
            all_num_steps: List of the number of steps each job ran for.
        """

        self._add_available_worker_id(worker_id)

        to_remove = []
        with self._scheduler_lock:
            worker_type = self._worker_id_to_worker_type_mapping[worker_id]
            current_timestamp = self.get_current_timestamp()

            if np.min(all_execution_times) < 0:
                # Micro-task failed.
                self._num_failures_per_job[job_id] += 1
                print(('%s]\t[Micro-task failed]\t'
                       'Job ID: %s') % (current_timestamp,
                                        job_id))
                if self._num_failures_per_job[job_id] >= MAX_FAILED_ATTEMPTS:
                    print(('%s]\t[Job failed]\t'
                           'Job ID: %s') % (current_timestamp, job_id))
                    to_remove.append(job_id)

            else:
                print(('%s]\t[Micro-task succeeded]\t'
                       'Job ID: %s\tWorker type: %s\t'
                       'Worker ID: %d') % (current_timestamp,
                                           job_id,
                                           worker_type,
                                           worker_id))
                self._num_failures_per_job[job_id] = 0
                for single_job_id, num_steps, execution_time in \
                        zip(job_id.singletons(), all_num_steps,
                            all_execution_times):
                    if single_job_id in self._running_jobs:
                        self._running_jobs.remove(single_job_id)
                        self._steps_run_so_far[single_job_id][worker_type] += \
                                num_steps
                        self._total_steps_run[single_job_id] += num_steps
                        if (self._total_steps_run[single_job_id] <
                             self._jobs[single_job_id].total_steps):
                            if not self._schedule_in_rounds:
                                self._add_to_queue(single_job_id)
                        else:
                            finish_time = \
                                    self._per_job_latest_timestamps[single_job_id]
                            print(('%s]\t[Job succeeded]\t'
                                   'Job ID: %s') % (finish_time,
                                                    single_job_id))
                            to_remove.append(single_job_id)

                    if not self._emulate:
                        # NOTE: We update the timestamp before calling this
                        # function in emulation.
                        self._per_job_latest_timestamps[single_job_id] = \
                                self.get_current_timestamp()

                # TODO: fix this for job pairs.
                if not self._emulate and not job_id.is_pair():
                    old_throughput = self._throughputs[job_id][worker_type]
                    self._update_throughput(job_id, worker_type,
                                            all_num_steps[0],
                                            all_execution_times[0])

                # If we just ran co-located jobs, use the maximum of the
                # individual execution times.
                max_execution_time = np.max(all_execution_times)

                if job_id in self._job_time_so_far:
                    self._job_time_so_far[job_id][worker_type] += \
                            max_execution_time
                self._worker_time_so_far[worker_type] += max_execution_time
                self._cumulative_worker_time_so_far[worker_id] += \
                        max_execution_time

        for single_job_id in to_remove:
            self.remove_job(single_job_id[0])
