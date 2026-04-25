# go2_rec_policy/terminations.py
"""Terminations for the recovery policy.

The agile policy terminates on any base contact or low base height — we
CANNOT do that here, because the robot is expected to start lying on its
base, and terminating immediately would make the problem unlearnable.

So the only termination left is time_out: give the policy the full
episode window to recover.
"""
from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
