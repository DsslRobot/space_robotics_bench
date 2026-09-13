from typing import ClassVar, Sequence

from srb.core.action import (
    ActionGroup,
    BinaryJointPositionActionCfg,
    DifferentialIKControllerCfg,
    FourWheelSteerActionCfg,
    SwitchableArmActionCfg,
    WheeledManipulatorActionGroup,
)
from srb.core.actuator import ImplicitActuatorCfg
from srb.core.asset import ArticulationCfg, Frame, Transform, WheeledManipulator
from srb.core.domain import Domain
from srb.core.sim import (
    ArticulationRootPropertiesCfg,
    UsdFileCfg,
)
from srb.utils.math import deg_to_rad, rpy_to_quat
from srb.utils.path import SRB_ASSETS_DIR_SRB_ROBOT

## Tool-centre-point offset from the Link7 flange along its own +Z axis
## (the EG2-4C2 gripper is welded directly to Link7 with an identity transform)
_TCP_OFFSET = (0.0, 0.0, 0.140)

## EG2-4C2 gripper: single actuated joint (eg2_joint1) + 5 joints driven in
## lock-step (mimic behavior emulated at the action-group level, since the
## PhysX mimic constraint was dropped from the imported gripper for
## simulation stability -- see assets/srb_assets robot/mobile_manipulation/lunarbot)
_GRIPPER_STROKE_RAD = 0.82
_GRIPPER_JOINT_NAMES = [
    "eg2_joint1",
    "eg2_joint2",
    "eg2_joint3",
    "eg2_joint4",
    "eg2_joint5",
    "eg2_joint6",
]
_GRIPPER_SIGN = {
    "eg2_joint1": 1.0,
    "eg2_joint2": 1.0,
    "eg2_joint5": 1.0,
    "eg2_joint6": 1.0,
    "eg2_joint3": -1.0,
    "eg2_joint4": -1.0,
}


class LunarBot(WheeledManipulator):
    """
    LunarBot: a four-wheel independent-drive/independent-steering (4WIS/4WID)
    rover chassis with a 7-DoF RealMan RM-75 manipulator and an Inspire-Robots
    EG2-4C2 parallel electric gripper, fused into a **single PhysX
    articulation** (base + arm + gripper share one USD/articulation root).
    """

    ## Scenario
    DOMAINS: ClassVar[Sequence[Domain]] = (Domain.MOON,)

    ## Model
    asset_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/lunarbot",
        spawn=UsdFileCfg(
            usd_path=SRB_ASSETS_DIR_SRB_ROBOT.joinpath("mobile_manipulation")
            .joinpath("lunarbot")
            .joinpath("lunarbot.usdc")
            .as_posix(),
            activate_contact_sensors=True,
            ## NOTE: gravity is disabled on the arm/gripper links directly in
            ## the USD asset (matching Franka's own convention in SRB). This
            ## is required, not cosmetic: the arm's relative-mode differential
            ## IK action re-syncs its target to the *current* joint state
            ## every step, so it provides zero position-holding stiffness
            ## against persistent external disturbances like gravity -- a
            ## real RM-75 controller performs active gravity compensation
            ## internally, which this approximates.
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05),
            joint_pos={
                ## Steering neutral, wheels free
                "chassis_to_.*_steering_joint": 0.0,
                ## Arm: folded-but-visible stow pose
                "joint1": 0.0,
                "joint2": -0.6,
                "joint3": 0.0,
                "joint4": 1.2,
                "joint5": 0.0,
                "joint6": 0.6,
                "joint7": 0.0,
                ## Gripper: open
                "eg2_joint1": _GRIPPER_STROKE_RAD,
                "eg2_joint2": _GRIPPER_STROKE_RAD,
                "eg2_joint5": _GRIPPER_STROKE_RAD,
                "eg2_joint6": _GRIPPER_STROKE_RAD,
                "eg2_joint3": -_GRIPPER_STROKE_RAD,
                "eg2_joint4": -_GRIPPER_STROKE_RAD,
            },
        ),
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=["chassis_to_.*_steering_joint"],
                effort_limit=100.0,
                velocity_limit=1.9635,
                stiffness=280.0,
                damping=90.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_steering_to_wheel_joint"],
                effort_limit=48.512,
                velocity_limit=41.4458,
                stiffness=0.0,
                damping=40.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                effort_limit={
                    "joint1": 60.0,
                    "joint2": 60.0,
                    "joint3": 30.0,
                    "joint4": 30.0,
                    "joint5": 10.0,
                    "joint6": 10.0,
                    "joint7": 10.0,
                },
                velocity_limit={
                    "joint1": 3.14,
                    "joint2": 3.14,
                    "joint3": 3.92,
                    "joint4": 3.92,
                    "joint5": 3.92,
                    "joint6": 3.92,
                    "joint7": 3.92,
                },
                stiffness=600.0,
                damping=140.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=_GRIPPER_JOINT_NAMES,
                effort_limit=0.9,
                velocity_limit=_GRIPPER_STROKE_RAD / 1.3,
                stiffness=200.0,
                damping=20.0,
                armature=1.0e-4,
            ),
        },
    )

    ## Actions
    actions: ActionGroup = WheeledManipulatorActionGroup(
        cmd_vel=FourWheelSteerActionCfg(
            asset_name="robot",
            steering_joint_names=[
                "chassis_to_front_left_steering_joint",
                "chassis_to_front_right_steering_joint",
                "chassis_to_rear_left_steering_joint",
                "chassis_to_rear_right_steering_joint",
            ],
            drive_joint_names=[
                "front_left_steering_to_wheel_joint",
                "front_right_steering_to_wheel_joint",
                "rear_left_steering_to_wheel_joint",
                "rear_right_steering_to_wheel_joint",
            ],
            ## (x, y) of each wheel in the chassis body frame [m], same order as above
            ## NOTE: the rear pair is offset an extra -0.03 m in x relative to
            ## the URDF's (symmetric) value -- the rear axle's USD rest pose
            ## was shifted rearward by that amount to correct a mismatch
            ## against the (asymmetric) chassis hull/rail mesh; keep this in
            ## sync with the rear steering/wheel links' authored positions.
            wheel_positions=[
                (0.4925, 0.42705),
                (0.4925, -0.42705),
                (-0.5225, 0.42705),
                (-0.5225, -0.42705),
            ],
            wheel_radius=0.1453,
            max_steering_angle=deg_to_rad(120.0),
            ## NOTE: both negated relative to the raw teleop/RL command
            ## convention -- verified empirically: (1) the chassis' local +X
            ## (as derived from the URDF/USD) drives opposite to the
            ## visually-intended "forward" direction; (2) positive omega (as
            ## produced by the "A" key) was observed to yield an angular
            ## velocity along -Z under a right-handed, Z-up convention,
            ## i.e. a right turn, whereas "A" should yield a left turn (+Z).
            scale_linear=-1.0,
            scale_angular=-1.0,
        ),
        arm=SwitchableArmActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            base_name="base_link",
            body_name="Link7",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            ik_scale=0.1,
            body_offset=SwitchableArmActionCfg.OffsetCfg(pos=_TCP_OFFSET),
            # Direct radians pass-through: raw joint-position sub-command IS
            # the physical target angle, matching OpenRAL's JOINT_POSITION
            # control mode's own convention (physical units end to end, no
            # HAL-side unit conversion needed -- unlike the IK sub-mode's
            # raw<->m/s calibration, which SRB's differential-IK action term
            # genuinely needs since it treats its raw input as a scaled
            # per-step position delta, not a physical unit).
            joint_pos_scale=1.0,
        ),
        hand=BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=_GRIPPER_JOINT_NAMES,
            close_command_expr={jn: 0.0 for jn in _GRIPPER_JOINT_NAMES},
            open_command_expr={
                jn: sign * _GRIPPER_STROKE_RAD for jn, sign in _GRIPPER_SIGN.items()
            },
        ),
    )

    ## Frames
    frame_base: Frame = Frame(prim_relpath="chassis_base_link")
    frame_flange: Frame = Frame(prim_relpath="Link7", offset=Transform(pos=_TCP_OFFSET))
    frame_front_camera: Frame = Frame(
        prim_relpath="rgbd_camera_frame/camera_front",
        offset=Transform(rot=rpy_to_quat(0.0, 15.0, 0.0)),
    )
    frame_wrist_camera: Frame = Frame(
        prim_relpath="Link7/camera_wrist",
        offset=Transform(
            pos=(0.0, -0.048, -0.018),
            rot=rpy_to_quat(0.0, -73.10, 90.0),
        ),
    )
