import os, sys
from policy import Policy

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

class ShockwavePolicy(Policy):
    def __init__(self):
        self._name = "shockwave"
