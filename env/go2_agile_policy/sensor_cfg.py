from __future__ import annotations
import numpy as np
import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns


@configclass
class Ray2dSensorCfg:
    enable:      bool  = True
    log2:        bool  = True
    min_dist:    float = 0.1
    max_dist:    float = 6.0
    theta_start: float = -np.pi / 4
    theta_end:   float =  np.pi / 4 + 0.0001
    theta_step:  float =  np.pi / 20
    x_0:         float = -0.05
    y_0:         float =  0.0
    front_rear:  bool  = False
    illusion:    bool  = True
    raycolor:    tuple = (0, 0.5, 0.5)

    # Derived helpers (computed in __post_init__)
    num_rays:    int   = MISSING
    fov_start_deg: float = MISSING
    fov_end_deg:   float = MISSING
    res_deg:       float = MISSING

    def __post_init__(self):
        self.fov_start_deg = float(np.degrees(self.theta_start))
        self.fov_end_deg   = float(np.degrees(self.theta_end))
        self.res_deg       = float(np.degrees(self.theta_step))
        self.num_rays      = int(
            round((self.theta_end - self.theta_start) / self.theta_step)
        )


@configclass
class DepthCamSensorCfg:
    enable:     bool  = False
    resolution: list  = (1280 // 8, 720 // 8)
    x:          float = 0.0
    y:          float = 0.0
    z:          float = 0.27
    far_plane:  float = 10.0
    hfov:       float = 102.0
    min_:       float = 0.1
    max_:       float = 6.0



def _build_ray2d_sensor(ray2d = Ray2dSensorCfg()) -> RayCasterCfg | None:
    """Builds RayCasterCfg from Ray2dSensorCfg. Returns None if disabled."""
    if not ray2d.enable:
        return None
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/base",
        offset=RayCasterCfg.OffsetCfg(pos=(ray2d.x_0, ray2d.y_0, 0.0)),
        attach_yaw_only=True,            # matches original yaw_quat() usage
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(ray2d.fov_start_deg, ray2d.fov_end_deg),
            horizontal_res=ray2d.res_deg,
        ),
        max_distance=ray2d.max_dist,
        drift_range=(0.0, 0.0),
        # debug_vis=True,
        # ── Required: meshes the rays will intersect against ───────────
        mesh_prim_paths=[
            "/World/ground",                      # terrain / ground plane
            # "/World/envs/env_.*/Obstacle_.*",         # all dynamic obstacles
        ],
    )

@configclass
class SensorsCfg:
    ray2d: Ray2dSensorCfg = Ray2dSensorCfg()
    depth_cam: DepthCamSensorCfg  = DepthCamSensorCfg()


_sensors_cfg = SensorsCfg()   # used at class-definition time below


