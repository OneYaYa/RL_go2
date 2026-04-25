import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def contact_forces_binary(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Returns binary foot contact state scaled to [-1, 1].
    Mirrors original: contact_filt.float() * 2 - 1.0
    Shape: (num_envs, num_bodies)
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # net contact force magnitude per body
    net_forces = contact_sensor.data.net_forces_w        # (N, num_bodies, 3)
    force_mag  = torch.norm(net_forces, dim=-1)           # (N, num_bodies)
    binary     = (force_mag > threshold).float()          # 0.0 or 1.0
    return binary * 2.0 - 1.0                             # scaled to [-1, 1]