import logging
from collections import OrderedDict
import numpy as np

# imports for solver
import cvxpy
import gurobipy


class ShockwaveScheduler(object):
    def __init__(self, shockwave_confg: dict):
        self.shockwave_config = shockwave_confg
        self.num_gpus = shockwave_confg["num_gpus"]
        self.round_duration = shockwave_confg["time_per_iteration"]
        self.round_index = 0
        self.recompute_flag = False
        self.schedules = OrderedDict()
        self.job_metadata = OrderedDict()

    def add_metadata(self, job_id, metadata):
        self.job_metadata[job_id] = metadata

    def delete_metadata(self, job_id):
        return self.job_metadata.pop(job_id, None)

    def _increment_round(self):
        self.round_index += 1

    def set_recompute_flag(self):
        self.recompute_flag = True

    def unset_recompute_flag(self):
        self.recompute_flag = False

    def current_round_schedule(self):
        # write to self.schedules
        return self.schedules[self.round_index]

    def _job_log_utility(self):
        pass

    def _finish_time_fairness(self):
        pass

    def _eisenberg_gale_program(self):
        pass

    def _generate_schedule(self):
        pass
