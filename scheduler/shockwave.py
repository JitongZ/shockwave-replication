import logging
from collections import OrderedDict
import numpy as np
import math
from job_metadata import ShockwaveJobMetadata

# imports for solver
import cvxpy as cp
import gurobipy


class ShockwaveScheduler(object):
    def __init__(self, shockwave_config: dict):
        self.shockwave_config = shockwave_config
        self.num_gpus = shockwave_config["num_gpus"]
        self.round_duration = shockwave_config["time_per_iteration"]
        self.future_rounds = shockwave_config["future_rounds"]
        self.priority_power = shockwave_config["lambda"]
        self.regularizer = shockwave_config["k"]
        self.round_index = 0
        self.recompute_flag = False
        self.schedules = OrderedDict()
        self.job_metadata = OrderedDict()
        self.finish_time_estimates = []

    @property
    def num_jobs(self):
        return len(self.job_metadata)

    def add_metadata(self, job_id, metadata: ShockwaveJobMetadata):
        self.job_metadata[job_id] = metadata

    def delete_metadata(self, job_id):
        return self.job_metadata.pop(job_id, None)

    def _increment_round(self):
        self.round_index += 1

    def set_recompute_flag(self):
        self.recompute_flag = True

    def unset_recompute_flag(self):
        self.recompute_flag = False

    def create_round_schedule_vars_and_constraints(self):
        """Generate round schedule's variables and constraints
        Args:
        Returns:
        - round_schedule_vars, round_schedule_constrs
        """
        round_schedule_vars = (
            []
        )  # boolean Variables of dimension (num_jobs, future_rounds)
        for _ in range(self.num_jobs):
            round_schedule_vars.append(
                [cp.Variable(boolean=True) for _ in range(self.future_rounds)]
            )
        # make sure number of required workers each round smaller than number of GPUs
        round_schedule_constrs = self.create_round_schedule_constraints()
        return round_schedule_vars, round_schedule_constrs

    def create_round_schedule_constraints(self, round_schedule_vars):
        round_schedule_constrs = []
        jobs_nworkers = [job.nworkers for job in self.job_metadata]
        for iround in range(self.future_rounds):
            round_schedule = [
                round_schedule_vars[ijob][iround] for ijob in range(self.num_jobs)
            ]
            round_required_workers = cp.vdot(
                cp.hstack(round_schedule), cp.hstack(jobs_nworkers)
            )
            round_schedule_constrs.append(round_required_workers <= self.num_gpus)
        return round_schedule_constrs

    def current_round_schedule(self):
        # TODO: only update schedule when recompute flag is true
        schedule_vars = self.dynamic_eisenberg_gale_scheduling()
        self._generate_schedule(schedule_vars)

        # write to self.schedules
        return self.schedules[self.round_index]

    def _job_log_utility(self, round_schedule_vars):
        """Compute nash social welfare first order approximation (EQ 7 in the paper)
        Args:
        Returns:
        - planned_runtimes, log_utilities, job_utility_constrs
        """
        # self.num_jobs
        log_bases = []
        bases_breakpoints = self.shockwave_config["log_approximation_bases"]
        for base in bases_breakpoints:
            if base == 0.0:
                log_bases.append(math.log(1e-6))
            else:
                log_bases.append(math.log(base))
        planned_runtimes = []  # Y * D in EQ(7)
        log_utilities = []  # result of EQ(7)
        job_utility_constrs = []  # constraints for solver
        planned_epochs_constrs = []

        for ijob in range(self.num_jobs):
            job = self.job_metadata[ijob]
            # epoch progress of a job in future_rounds, the right term in EQ(7)
            planned_epochs = cp.Variable(nonneg=True)

            job.recompute_epoch_duration()
            # Epoch duration as in the footnote under EQ(7)
            epoch_duration_interpolated = np.mean(
                job.epoch_duration[: job.completed_epochs + 1]
            )

            planned_runtime = epoch_duration_interpolated * planned_epochs
            planned_runtimes.append(planned_runtime)

            # planned runtime <= number of rounds a job is planned * round duration
            planned_epochs_constrs.append(
                planned_runtime
                <= cp.sum(cp.hstack(round_schedule_vars[ijob])) * self.round_duration
            )

            # job.completed_epochs = F in EQ(7), TODO: do we need to use cp.multiply?
            objective_epochs_normalized = (
                job.completed_epochs + planned_epochs
            ) / job.total_epochs

            # the following is a trick to optimize for non-linear item (e.g. log function)
            # Ref: Inspired by the author's reply.
            # Create binary variables for each segment
            segment_pointer = cp.Variable(len(bases_breakpoints) - 1, boolean=True)
            # There should be only one segment
            breakpoints = np.array(bases_breakpoints)
            log_bases_array = np.array(log_bases)
            approx_constraints = [
                cp.sum(segment_pointer) == 1,
                approx_constraints <= breakpoints[:-1] @ segment_pointer,
                approx_constraints >= breakpoints[1:] @ segment_pointer,
            ]

            log_var_approx = (
                log_bases_array[:-1] @ segment_pointer
                + (log_bases_array[1:] - log_bases_array[:-1])
                / (breakpoints[1:] - breakpoints[:-1])
                * (objective_epochs_normalized - breakpoints[:-1])
                @ segment_pointer
            )
            log_utilities.append(log_var_approx)
            job_utility_constrs += approx_constraints

        job_utility_constrs += planned_epochs_constrs

        return planned_runtimes, log_utilities, job_utility_constrs

    def _compute_interpolated_finish_time(self, alpha=0.9):
        """
        Compute a running average on job finish time.

        Returns:
        - float, interpolated job finish time
        """
        round_ids = [id for id, _ in self.finish_time_estimates]
        windows = np.diff(round_ids)
        weights = [1] if np.sum(windows) == 0 else windows / np.sum(windows)
        finish_times = np.array(
            [ft for _, ft in self.finish_time_estimates[: weights.size]]
        )
        avg_finish_time = np.dot(weights, finish_times)
        interpolated_finish_time = (
            alpha * avg_finish_time + (1 - alpha) * self.finish_time_estimates[-1][1]
        )
        return interpolated_finish_time

    def _compute_finish_times(self, planned_runtimes):
        """
        Estimate the Finish Time Fairness by
        (predicted job completion time / interpolated job finish time).

        Returns:
        - list of float, finish time fairness estimate for each job
        """
        remaining_times = []
        finish_time_fairnesses = []
        for ijob in range(self.num_jobs):
            job = self.job_metadata[ijob]
            contention_factor = self.num_jobs / self.num_gpus
            planned_time = planned_runtimes[ijob]
            round_time = (self.round_index + self.future_rounds) * self.round_duration
            remaining_time = max(job.compute_remaining_runtime() - planned_time, 0)
            remaining_times.append(remaining_time)
            predicted_job_completion_time = (
                round_time + remaining_time * contention_factor
            )
            # TODO: subtract job.submit_time from both denominator and numerator
            predicted_finish_time = (
                job.submit_time
                + sum(job.epoch_duration[: job.completed_epochs])
                + job.compute_remaining_runtime()
            )
            self.finish_time_estimates.append(self.round_index, predicted_finish_time)
            finish_time_fairnesses.append(
                predicted_job_completion_time / self._compute_interpolated_finish_time()
            )
        return remaining_times, finish_time_fairnesses

    def _prioritize_unfair_jobs(self, schedule_vars, priorities):
        """
        Note: this is based on Shockwave paper Appendix G.2. Some logic was not
        specified in the paper and were implemented based on an understanding of
        Shockwave's codebase.
        """
        prioritized_schedule_vars = []  # new schedule to solve for.
        for _ in range(self.num_jobs):
            prioritized_schedule_vars.append(
                [cp.Variable(boolean=True) for _ in range(self.future_rounds)]
            )

        job_planned_rounds = (
            []
        )  # number of planned rounds in the planning window for each job.
        for i in range(self.num_jobs):
            job_planned_rounds.append(cp.sum(cp.hstack(schedule_vars[i])).value)

        # Constraint: new schedule has the same planned rounds for each job
        consts = []
        for i in range(self.num_jobs):
            consts.append(
                job_planned_rounds[i] == cp.sum(cp.hstack(prioritized_schedule_vars[i]))
            )

        # Constraint: in each round, the total nworkers required cannot exceed num_gpus
        consts += self.create_round_schedule_constraints(schedule_vars)

        # Objective: minimize scheduled round index for more unfair jobs
        # Ref: The exact logic of this optimization is not specified in the paper's
        #      Appendix G.2. The following objective is inspired by the Shockwave codebase.
        objective_per_job = []
        for ijob in range(self.num_jobs):
            priority = priorities[ijob]
            if job_planned_rounds[ijob] > 0:
                avg_sched_idx = (
                    cp.vdot(
                        cp.hstack([t for t in range(self.future_rounds)]),
                        cp.hstack(prioritized_schedule_vars[ijob]),
                    )
                    / job_planned_rounds[ijob]
                )
                objective_per_job.append(avg_sched_idx * priority)

        objective = cp.Minimize(cp.sum(cp.hstack(objective_per_job)))
        problem = self._solve_gurobi(objective, consts)

        if problem.status not in cp.settings.SOLUTION_PRESENT:
            return schedule_vars

        return prioritized_schedule_vars

    def _eisenberg_gale_program(self):
        """ """
        constraints = []
        # create a round's job schedule's cp variables, and corresponding constraints
        round_schedule_vars, round_schedule_constrs = (
            self.create_round_schedule_vars_and_constraints()
        )
        constraints += round_schedule_constrs
        # compute utility related variables and constraints
        planned_runtimes, log_utilities, job_utility_constrs = self._job_log_utility(
            round_schedule_vars
        )
        constraints += job_utility_constrs
        # compute remaining run times (R in EQ 10) and FTFs (EQ 9)
        remaining_times, finish_time_fairnesses = self._compute_finish_times(
            planned_runtimes
        )
        # compute makespan (EQ 10)
        makespan = cp.max(cp.hstack(remaining_times))
        prioritized_log_utilities = []
        priorities = []
        # compute left item of EQ 11
        for util, ftf in zip(log_utilities, finish_time_fairnesses):
            priority = ftf**self.priority_power
            prioritized_util = util * priority
            priorities.append(priority)
            prioritized_log_utilities.append(prioritized_util)

        objective = cp.Maximize(
            cp.sum(
                cp.hstack(prioritized_log_utilities)
                / (self.num_jobs * self.future_rounds)
            )
            - self.regularizer * makespan
        )

        problem = self._solve_gurobi(objective, constraints)
        assert problem.status in cp.settings.SOLUTION_PRESENT

        round_schedule_vars = self._prioritize_unfair_jobs(round_schedule_vars)

        return round_schedule_vars

    def _generate_schedule(self, round_schedule_vars):
        # rounds_schedules = OrderedDict()
        for iround in self.future_rounds:
            scheduled_job_ids = []
            future_round_index = self.round_index + iround
            for ijob in range(self.num_jobs):
                if bool(round_schedule_vars[ijob][iround].value.item()):
                    scheduled_job_ids.append(ijob)
            # rounds_schedules[future_round_index] = scheduled_job_ids
            self.schedules[future_round_index] = scheduled_job_ids
        # return rounds_schedules

    def _solve_gurobi(self, objective, constraints):
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(
            solver=cp.GUROBI,
            verbose=True,
            MIPGap=self.shockwave_config["solver_rel_gap"],
            Threads=self.shockwave_config["solver_num_threads"],
            TimeLimit=self.shockwave_config["solver_timeout"],
        )
        if problem.status != "optimal":
            print("WARNING: Allocation returned by policy not optimal!")
        return problem
