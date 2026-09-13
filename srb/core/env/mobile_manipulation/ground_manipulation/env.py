from dataclasses import MISSING

from srb import assets
from srb.core.action import (
    InverseKinematicsActionGroup,
    JointPositionRelativeActionGroup,
    OperationalSpaceControlActionGroup,
)
from srb.core.asset import (
    Articulation,
    AssetVariant,
    GroundManipulator,
    WheeledManipulator,
)
from srb.core.env.manipulation.env import (
    ManipulationEnv,
    ManipulationEventCfg,
    ManipulationSceneCfg,
)
from srb.core.env.mobile.ground.env import (
    GroundEnv,
    GroundEnvCfg,
    GroundEventCfg,
    GroundSceneCfg,
)
from srb.core.manager import SceneEntityCfg
from srb.core.sensor import ContactSensorCfg
from srb.utils.cfg import configclass
from srb.utils.math import combine_frame_transforms_tuple


@configclass
class GroundManipulationSceneCfg(GroundSceneCfg, ManipulationSceneCfg):
    pass


@configclass
class GroundManipulationEventCfg(GroundEventCfg, ManipulationEventCfg):
    pass


@configclass
class GroundManipulationEnvCfg(GroundEnvCfg):
    ## Assets
    robot: GroundManipulator | WheeledManipulator | AssetVariant = (
        assets.GenericGroundManipulator(mobile_base=assets.Spot(), manipulator=assets.Franka())
    )
    _robot: GroundManipulator | WheeledManipulator = MISSING  # type: ignore

    ## Scene
    scene: GroundManipulationSceneCfg = GroundManipulationSceneCfg()

    ## Events
    events: GroundManipulationEventCfg = GroundManipulationEventCfg()

    ## Time
    env_rate: float = 1.0 / 150.0
    agent_rate: float = 1.0 / 50.0

    def __post_init__(self):
        ## Jacobian-based actions are currently not supported for free-floating manipulators
        if isinstance(self.robot, GroundManipulator) and isinstance(
            self.robot.manipulator.actions,
            (InverseKinematicsActionGroup, OperationalSpaceControlActionGroup),
        ):
            self.robot.manipulator.actions = JointPositionRelativeActionGroup()

        super().__post_init__()

        if isinstance(self._robot, WheeledManipulator):
            ## Monolithic wheeled manipulator: base + arm + end effector all
            ## belong to the same articulation exposed as the "robot" scene entity.
            ## NOTE: When overriding a CombinedMobileManipulator-typed default (e.g.
            ## GenericGroundManipulator) with a WheeledManipulator via a Hydra
            ## `env.robot=...` CLI override, `self.scene.manipulator`/`end_effector`
            ## are reconstructed from the *stale pre-override* config snapshot
            ## (they are side effects of the combined-robot wiring in
            ## `env_cfg.py::_add_robot`, which is skipped for non-combined
            ## robots and therefore never re-clears them). Explicitly clear
            ## them here so no leftover Franka/Spot-shaped assets get spawned.
            self.scene.manipulator = None
            self.scene.end_effector = None
            self.joint_assemblies.pop("manipulator", None)
            self.joint_assemblies.pop("end_effector", None)

            # Sensor: End-effector transform
            self.scene.tf_end_effector.prim_path = (
                f"{self.scene.robot.prim_path}/{self._robot.frame_base.prim_relpath}"
            )
            self.scene.tf_end_effector.target_frames[0].prim_path = (
                f"{self.scene.robot.prim_path}/{self._robot.frame_flange.prim_relpath}"
            )
            (
                self.scene.tf_end_effector.target_frames[0].offset.pos,
                self.scene.tf_end_effector.target_frames[0].offset.rot,
            ) = (
                self._robot.frame_flange.offset.pos,
                self._robot.frame_flange.offset.rot,
            )

            # Sensor: Robot contacts (covers the whole fused articulation)
            self.scene.contacts_robot.prim_path = f"{self.scene.robot.prim_path}/.*"

            # No separate end-effector asset -> no dedicated contact sensor
            self.scene.contacts_end_effector = None

            # Event: Randomize robot joints (the "robot" scene entity IS the manipulator)
            self.events.randomize_robot_joints.params["asset_cfg"] = SceneEntityCfg(
                "robot"
            )
            return

        assert self.scene.manipulator is not None

        ## Adapted from ManipulationEnvCfg
        # Sensor: End-effector transform
        self.scene.tf_end_effector.prim_path = f"{self.scene.manipulator.prim_path}/{self._robot.manipulator.frame_base.prim_relpath}"
        self.scene.tf_end_effector.target_frames[
            0
        ].prim_path = f"{self.scene.manipulator.prim_path}/{self._robot.manipulator.frame_flange.prim_relpath}"
        if self._robot.manipulator.end_effector is not None:
            (
                self.scene.tf_end_effector.target_frames[0].offset.pos,
                self.scene.tf_end_effector.target_frames[0].offset.rot,
            ) = combine_frame_transforms_tuple(
                self._robot.manipulator.frame_flange.offset.pos,
                self._robot.manipulator.frame_flange.offset.rot,
                self._robot.manipulator.end_effector.frame_tool_centre_point.offset.pos,
                self._robot.manipulator.end_effector.frame_tool_centre_point.offset.rot,
            )
        else:
            (
                self.scene.tf_end_effector.target_frames[0].offset.pos,
                self.scene.tf_end_effector.target_frames[0].offset.rot,
            ) = (
                self._robot.manipulator.frame_flange.offset.pos,
                self._robot.manipulator.frame_flange.offset.rot,
            )

        # Sensor: Robot contacts
        self.scene.contacts_robot.prim_path = f"{self.scene.manipulator.prim_path}/.*"

        # Sensor: End-effector contacts
        self.scene.contacts_end_effector = (
            ContactSensorCfg(
                prim_path=f"{self._robot.manipulator.end_effector.asset_cfg.prim_path}/.*",
            )
            if self._robot.manipulator.end_effector is not None
            else None
        )

        # Event: Randomize robot joints
        self.events.randomize_robot_joints.params["asset_cfg"] = SceneEntityCfg(
            "manipulator"
        )


class GroundManipulationEnv(GroundEnv, ManipulationEnv):
    cfg: GroundManipulationEnvCfg

    def __init__(self, cfg: GroundManipulationEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._manipulator: Articulation = (
            self._robot
            if isinstance(cfg._robot, WheeledManipulator)
            else self.scene["manipulator"]
        )
