from dataclasses import MISSING

from isaaclab.sensors.ray_caster import patterns as ray_caster_patterns

from srb.core.asset import AssetVariant, Humanoid, MobileRobot
from srb.core.env import BaseEventCfg, BaseSceneCfg, DirectEnv, DirectEnvCfg
from srb.core.marker import RED_ARROW_X_MARKER_CFG
from srb.core.sensor import Imu, ImuCfg, RayCaster, RayCasterCfg
from srb.utils.cfg import configclass

## Livox Mid-360 FOV (datasheet): 360 deg horizontal, -7..+52 deg vertical.
## `RayCaster`'s `LidarPatternCfg` rasterizes a fixed channel/azimuth grid --
## it cannot reproduce the Mid-360's real non-repetitive scan pattern (no
## IsaacLab sensor here does; see docs/execution_plan.md item 2c for the
## planned RTX Lidar follow-up). Channel/resolution values below are chosen
## for a comparable per-scan point density (~14k pts/scan), not a literal
## channel-count reproduction.
_MID360_VERTICAL_FOV_DEG = (-7.0, 52.0)
_MID360_HORIZONTAL_RES_DEG = 1.0
_MID360_CHANNELS = 40
_MID360_MAX_RANGE_M = 40.0


@configclass
class MobileSceneCfg(BaseSceneCfg):
    imu_robot: ImuCfg = ImuCfg(
        prim_path=MISSING,  # type: ignore
        gravity_bias=(0.0, 0.0, 0.0),
        visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(  # type: ignore
            prim_path="/Visuals/imu_robot/lin_acc"
        ),
    )
    ## Optional: only populated in __post_init__ when the robot declares a
    ## `frame_lidar` (mirrors `contacts_end_effector`'s optional-sensor
    ## pattern in `srb/core/env/manipulation/env.py`).
    lidar_robot: RayCasterCfg | None = None


@configclass
class MobileEventCfg(BaseEventCfg):
    pass


@configclass
class MobileEnvCfg(DirectEnvCfg):
    ## Assets
    robot: MobileRobot | Humanoid | AssetVariant = MISSING  # type: ignore
    _robot: MobileRobot | Humanoid = MISSING  # type: ignore

    ## Scene
    scene: MobileSceneCfg = MobileSceneCfg()

    ## Events
    events: MobileEventCfg = MobileEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Sensor: Robot IMU
        if self._robot.frame_imu:
            self.scene.imu_robot.prim_path = (
                f"{self.scene.robot.prim_path}/{self._robot.frame_base.prim_relpath}"
            )
            self.scene.imu_robot.offset.pos = self._robot.frame_imu.offset.pos
            self.scene.imu_robot.offset.rot = self._robot.frame_imu.offset.rot
        else:
            self.scene.imu_robot.prim_path = (
                f"{self.scene.robot.prim_path}/{self._robot.frame_base.prim_relpath}"
                if self._robot.frame_base.prim_relpath
                else self.scene.robot.prim_path
            )
            self.scene.imu_robot.offset.pos = self._robot.frame_base.offset.pos
            self.scene.imu_robot.offset.rot = self._robot.frame_base.offset.rot

        # Sensor: Robot lidar (e.g. LunarBot's Livox Mid-360)
        if self._robot.frame_lidar is not None and self.scene.scenery is not None:
            self.scene.lidar_robot = RayCasterCfg(
                prim_path=f"{self.scene.robot.prim_path}/{self._robot.frame_lidar.prim_relpath}",
                offset=RayCasterCfg.OffsetCfg(
                    pos=self._robot.frame_lidar.offset.pos,
                    rot=self._robot.frame_lidar.offset.rot,
                ),
                ## NOTE: this version of IsaacLab's `RayCaster` supports
                ## exactly one static mesh (hard-enforced, see
                ## `ray_caster.py::_initialize_warp_meshes`) baked into a BVH
                ## once at sensor init -- it does not re-query the live scene,
                ## so it is blind to other prims (e.g. task objects like a
                ## pedestal/panel) and, for `num_envs > 1`, only ever sees
                ## env_0's terrain even though every env's lidar samples
                ## against it. Correct for the actual use case (single-env
                ## real/deployed-robot navigation via Nav2, ground/slope
                ## sensing); near-field object avoidance is expected to come
                ## from the depth camera's point cloud instead. See
                ## docs/execution_plan.md item 2c for the RTX Lidar
                ## follow-up that removes this limitation.
                mesh_prim_paths=[
                    self.scene.scenery.prim_path.replace(
                        "{ENV_REGEX_NS}", "/World/envs/env_0"
                    )
                ],
                pattern_cfg=ray_caster_patterns.LidarPatternCfg(
                    channels=_MID360_CHANNELS,
                    vertical_fov_range=_MID360_VERTICAL_FOV_DEG,
                    horizontal_fov_range=(-180.0, 180.0),
                    horizontal_res=_MID360_HORIZONTAL_RES_DEG,
                ),
                max_distance=_MID360_MAX_RANGE_M,
            )


class MobileEnv(DirectEnv):
    cfg: MobileEnvCfg

    def __init__(self, cfg: MobileEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._imu_robot: Imu = self.scene["imu_robot"]
        self._lidar_robot: RayCaster | None = self.scene.sensors.get(  # type: ignore
            "lidar_robot", None
        )
