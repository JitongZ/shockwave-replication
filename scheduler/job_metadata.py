import copy
import numpy as np
from collections import OrderedDict

INFINITY = 1e9


class ShockwaveJobMetadata:
    def __init__(self, config: dict, round_duration: int, scale_factor=None):
        """
        ShockwaveJobMetadata constructor.

        Parameters:
        - config: dict, a dictionary mapping configuration fields to their values.
            Expected keys are:
            - "num_epochs": int, total number of epochs.
            - "num_samples_per_epoch": int, number of samples processed per epoch.
            - "scale_factor": int, number of workers or scale factor.
            - "duration": float, total duration of the job.
            - "bs_every_epoch": list of int, batch sizes for each epoch.
            - "mem_every_epoch": list of float, memory requirements for each epoch.
            - "util_every_epoch": list of float, GPU utilization for each epoch.
            - "duration_every_epoch": list of float, duration for each epoch.

        Returns:
        - None
        """
        self.total_epochs = config["num_epochs"]
        self.completed_epochs = 0

        self.nsamples_per_epoch = config["num_samples_per_epoch"]
        self.nworkers = config["scale_factor"]
        if scale_factor is not None:
            self.nworkers = scale_factor
        # print(f"self.nworkers:{self.nworkers}")
        self.duration = config["duration"]

        self.epoch_batch_sizes = config["bs_every_epoch"]
        self.epoch_mem_reqs = config["mem_every_epoch"]
        self.epoch_gpu_reqs = config["util_every_epoch"]
        durations = config["duration_every_epoch"]

        self.epoch_durations = [max(1.0, round(duration)) for duration in durations]
        self.estimated_epoch_durations = copy.deepcopy(self.epoch_durations)

        self.regimes = sorted(list(set(self.epoch_batch_sizes)))
        self.dirichlet = {
            bs: self.total_epochs / len(self.regimes) for bs in self.regimes
        }

        self.submit_time = None
        self.throughput_schedule = OrderedDict()
        self.round_duration = round_duration

    def submit(self, time):
        """
        Set job submit time, used for finish time fairness computation.

        Parameters:
        - time: float, time when the job is added.

        Returns:
        - None
        """
        if self.submit_time is None:
            self.submit_time = time

    def complete(self, num_epochs=None):
        """
        Mark the current job as complete by setting the number of completed epochs
        to the total number of epochs.

        Returns:
        - None
        """
        if num_epochs is None:
            self.completed_epochs = self.total_epochs
        else:
            assert (
                num_epochs <= self.total_epochs
            ), f"Incorrect epoch progress {num_epochs}"
            self.completed_epochs = num_epochs

    def update_throughput_schedule(self, round_id, throughput, bs):
        """
        Update the throughput schedule of the current job.

        Parameters:
        - round_id: int, index of current round.
        - throughput: float, throughput in the current round.
        - bs: int, batch size in the current round.

        Returns:
        - None
        """
        self.throughput_schedule[round_id] = (throughput, bs)

    def recompute_epoch_duration(self):
        """
        Compute the estimated epoch duration based on the throughput schedule.
        This adjusts the estimated durations of epochs to better match the actual measured throughput.

        Returns:
        - None
        """
        assert self.throughput_schedule is not None, "Throughput schedule must be set."
        if len(self.throughput_schedule) <= 0:
            return
        assert self.epoch_batch_sizes is not None, "Epoch batch sizes must be set."
        assert self.round_duration is not None, "Round duration must be set."

        # Compute the actual number of samples processed (measured_nsamples)
        measured_rounds = sorted(list(self.throughput_schedule.keys()))
        prev_round = 0
        measured_nsamples = 0  # Actual number of samples processed
        for cur_round in measured_rounds:
            measured_throughput, measured_bs = self.throughput_schedule[cur_round]
            measured_niters = (
                measured_throughput  # Iterations per second
                * self.round_duration  # Seconds per round
                * (cur_round - prev_round)  # Number of rounds
            )
            measured_nsamples += measured_bs * measured_niters
            prev_round = cur_round
        end_round = max(measured_rounds)
        measured_time_range = self.round_duration * end_round

        # Compute the estimated number of samples processed (estimated_nsamples)
        estimated_time_range = 0
        estimated_nsamples = 0  # Estimated number of samples processed
        for iepoch, duration in enumerate(self.estimated_epoch_durations):
            if estimated_time_range + duration > measured_time_range:
                break
            else:
                estimated_time_range += duration
                estimated_nsamples += self.nsamples_per_epoch
        # Account for the partially completed epoch
        time_diff = measured_time_range - estimated_time_range
        if time_diff > 0:
            epoch_duration = self.epoch_durations[iepoch]
            estimated_nsamples += self.nsamples_per_epoch * time_diff / epoch_duration

        # Adjust epoch durations based on the ratio of estimated to measured samples
        if (
            measured_nsamples <= 0
            or estimated_nsamples <= 0
            or abs(measured_nsamples - estimated_nsamples) / estimated_nsamples <= 0.4
        ):
            return
        else:
            factor = estimated_nsamples / measured_nsamples
            for iepoch in range(len(self.epoch_durations)):
                self.epoch_durations[iepoch] = (
                    self.estimated_epoch_durations[iepoch] * factor
                )

    def compute_bs_epoch_duration(self):
        """
        Compute a mapping from each batch size to its average epoch duration.

        Returns:
        - dict: A dictionary where keys are batch sizes (bs) and values are the
          average duration of epochs with that batch size.
        """
        self.recompute_epoch_duration()
        bs_epoch_duration_map = {}
        for iepoch, duration in enumerate(self.epoch_durations):
            bs = self.epoch_batch_sizes[iepoch]
            if bs not in bs_epoch_duration_map:
                bs_epoch_duration_map[bs] = []
            bs_epoch_duration_map[bs].append(duration)
        for bs in bs_epoch_duration_map.keys():
            mean_duration = np.mean(bs_epoch_duration_map[bs])
            assert mean_duration > 0 and mean_duration < INFINITY
            bs_epoch_duration_map[bs] = mean_duration
        return bs_epoch_duration_map

    def compute_remaining_runtime(self):
        """
        Compute the remaining runtime of the job using the Dirichlet posterior distribution
        and the average epoch durations for each batch size.

        Returns:
        - float: The estimated remaining runtime of the job.
        """
        if len(self.dirichlet) <= 0 or self.completed_epochs >= self.total_epochs:
            return 1.0

        # Update Dirichlet distribution with observed batch sizes
        observed_bs_schedule = self.epoch_batch_sizes[: self.completed_epochs + 1]
        dirichlet_posterior = copy.deepcopy(self.dirichlet)
        for bs in observed_bs_schedule:
            dirichlet_posterior[bs] += 1

        # Rebase Dirichlet distribution to match the total number of epochs
        dirichlet_epoch_sum = sum(list(dirichlet_posterior.values()))
        dirichlet_rebased = {
            bs: self.total_epochs * (epochs / dirichlet_epoch_sum)
            for bs, epochs in dirichlet_posterior.items()
        }

        # Adjust for already completed epochs
        for bs in observed_bs_schedule:
            if dirichlet_rebased[bs] >= 1:
                dirichlet_rebased[bs] -= 1

        # Scale down dirichlet_rebased so that total remaining epochs stay the same
        remaining_epochs = self.total_epochs - self.completed_epochs
        normalizer = min(
            1, remaining_epochs / int(sum(list(dirichlet_rebased.values())) + 1)
        )

        # Compute the remaining runtime based on average epoch durations
        bs_epoch_duration_map = self.compute_bs_epoch_duration()
        remaining_runtime = 0.0
        for bs in dirichlet_rebased.keys():
            bs_remaining_epochs = dirichlet_rebased[bs]
            remaining_runtime += bs_remaining_epochs * bs_epoch_duration_map[bs]

        remaining_runtime *= normalizer

        return remaining_runtime
