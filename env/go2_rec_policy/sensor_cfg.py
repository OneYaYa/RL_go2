# go2_rec_policy/sensor_cfg.py
"""Sensor configuration for the recovery policy.

The recovery policy intentionally strips out perception sensors (ray2d,
depth camera) because the robot only needs proprioception to right
itself. We keep the config shell so the env mirrors the agile policy
layout and can be extended later if needed.
"""
from __future__ import annotations
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class SensorsCfg:
    # Recovery is proprioception-only — no ray2d / depth cam fields.
    pass


_sensors_cfg = SensorsCfg()
