import torch
import warp as wp
from isaaclab.sensors import RayCaster as _IsaacLabRayCaster
from isaaclab.sensors import RayCasterCfg as _IsaacLabRayCasterCfg
from isaaclab.utils import configclass


class RayCaster(_IsaacLabRayCaster):
    """`RayCaster` whose BVH and ray casts live on CUDA even when `sim.device` is the CPU.

    The upstream sensor builds its warp mesh (and therefore launches the
    ray-cast kernel) on the *simulation* device. SRB pins ``sim.device`` to the
    CPU unless a scene contains deformable objects, which puts every ray of a
    dense lidar (e.g. the Mid-360's 40 x 360 pattern) through warp's
    single-threaded CPU backend: ~59 ms per tick, measured. Casting on CUDA and
    copying the hit points back costs well under a millisecond.
    """

    cfg: "RayCasterCfg"

    def _initialize_warp_meshes(self):
        super()._initialize_warp_meshes()
        device = self.cfg.raycast_device
        if device is None:
            return
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            mesh = self.meshes[mesh_prim_path]
            if wp.device_to_torch(mesh.device) == device:
                continue
            self.meshes[mesh_prim_path] = wp.Mesh(
                points=mesh.points.to(device), indices=mesh.indices.to(device)
            )


@configclass
class RayCasterCfg(_IsaacLabRayCasterCfg):
    class_type: type = RayCaster

    raycast_device: str | None = "cuda" if torch.cuda.is_available() else None
    """Device the warp mesh (BVH) is built on and ray casts are launched on.

    ``None`` keeps upstream behaviour (the simulation device). Hit points are
    always returned on the simulation device, so this is transparent to users.
    """
