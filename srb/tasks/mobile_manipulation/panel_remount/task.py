from typing import Sequence

import torch

from srb import assets
from srb._typing import StepReturn
from srb.core.asset import (
    Articulation,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from srb.core.env import (
    GroundManipulationEnv,
    GroundManipulationEnvCfg,
    GroundManipulationEventCfg,
    GroundManipulationSceneCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sensor import ContactSensor
from srb.core.sim import PreviewSurfaceCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    matrix_from_quat,
    rotmat_to_rot6d,
    scale_transform,
    subtract_frame_transforms,
)

##############
### Config ###
##############

## Where the panel spawns relative to the env origin -- knocked off its
## mount, lying nearby. Randomized per-reset within a small radius so the
## exact starting pose is not memorizable across episodes.
##
## Positions here MUST stay within the procedural moon terrain's flat
## landing pad (env_cfg.py's `flat_area_size`, ~4m across at this task's
## `env_spacing=32.0`, centred on the env origin) -- outside it the terrain
## is genuinely uneven, and a rigid object spawned there can end up
## embedded in a bump; PhysX's depenetration solver then ejects it with a
## large velocity spike, which (combined with the Moon's weak gravity)
## sends it falling/flying indefinitely instead of settling. Confirmed
## live 2026-09-13: an earlier attempt at x=2.4-3.0 did exactly this (panel
## fell to z=-2637 and counting, never hitting anything). Keep everything
## within ~1.5 m of the origin, well inside the pad.
_PANEL_SPAWN_POS = (1.1, 0.3, 0.3)
_PANEL_SPAWN_POS_RANGE_XY = 0.2

## Pedestal position (fixed, not randomized -- the "infrastructure site" the
## mission instruction points the robot at) and the panel's target pose once
## correctly remounted on it.
_PEDESTAL_POS = (1.5, 0.0, 0.0)
_PANEL_TARGET_POS = (1.5, 0.0, 0.5)
_PANEL_TARGET_QUAT = (1.0, 0.0, 0.0, 0.0)

## Success tolerances live as local variables inside _compute_step_return
## (deliberately loose for a first working version -- LunarBot's arm is
## driven through CARTESIAN_TWIST, an EE-velocity surface, not a precision
## position controller yet) -- torch.jit.script cannot close over a plain
## Python float defined at module level, only over locals/arguments.


@configclass
class SceneCfg(GroundManipulationSceneCfg):
    pedestal: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/pedestal",
        spawn=None,  # set in TaskCfg.__post_init__
        init_state=AssetBaseCfg.InitialStateCfg(pos=_PEDESTAL_POS),
    )
    panel: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/panel",
        spawn=None,  # set in TaskCfg.__post_init__
        init_state=RigidObjectCfg.InitialStateCfg(pos=_PANEL_SPAWN_POS),
    )


@configclass
class EventCfg(GroundManipulationEventCfg):
    randomize_panel_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("panel"),
            "pose_range": {
                "x": (
                    _PANEL_SPAWN_POS[0] - _PANEL_SPAWN_POS_RANGE_XY,
                    _PANEL_SPAWN_POS[0] + _PANEL_SPAWN_POS_RANGE_XY,
                ),
                "y": (
                    _PANEL_SPAWN_POS[1] - _PANEL_SPAWN_POS_RANGE_XY,
                    _PANEL_SPAWN_POS[1] + _PANEL_SPAWN_POS_RANGE_XY,
                ),
                "z": (_PANEL_SPAWN_POS[2], _PANEL_SPAWN_POS[2]),
                "roll": (torch.pi / 2, torch.pi / 2),  # lying flat, knocked over
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {},
        },
    )


@configclass
class TaskCfg(GroundManipulationEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 60.0
    is_finite_horizon: bool = True

    ## Target -- Panel remounted on the pedestal
    panel_tf_pos_target = _PANEL_TARGET_POS
    panel_tf_quat_target = _PANEL_TARGET_QUAT
    panel_target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/panel_target",
        markers={
            "target": assets.SolarPanel().asset_cfg.spawn.replace(  # type: ignore
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.8, 0.2)),
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Scene: Pedestal -- the fixed infrastructure fixture the panel mounts on.
        self.scene.pedestal.spawn = assets.IndustrialPedestal50().asset_cfg.spawn

        # Scene: Panel -- the object that has come loose and must be
        # relocated back onto the pedestal.
        self.scene.panel = assets.SolarPanel().asset_cfg.replace(
            init_state=RigidObjectCfg.InitialStateCfg(pos=_PANEL_SPAWN_POS),
        )


############
### Task ###
############


class Task(GroundManipulationEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._panel: RigidObject = self.scene["panel"]
        self._panel_target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.panel_target_marker_cfg
        )

        ## Target pose is a fixed offset from the env origin (the pedestal
        ## itself is static scenery, not a queryable Asset -- same pattern
        ## solar_panel_assembly uses for its own target).
        self._tf_pos_panel_target = self.scene.env_origins + torch.tensor(
            self.cfg.panel_tf_pos_target, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._tf_quat_panel_target = torch.tensor(
            self.cfg.panel_tf_quat_target, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)

        ## Visualize target
        self._panel_target_marker.visualize(
            self._tf_pos_panel_target, self._tf_quat_panel_target
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

    def extract_step_return(self) -> StepReturn:
        return _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            # Root
            tf_pos_robot=self._robot.data.root_pos_w,
            tf_quat_robot=self._robot.data.root_quat_w,
            vel_lin_robot=self._robot.data.root_lin_vel_b,
            vel_ang_robot=self._robot.data.root_ang_vel_b,
            # IMU
            imu_lin_acc=self._imu_robot.data.lin_acc_b,
            imu_ang_vel=self._imu_robot.data.ang_vel_b,
            # Joints (manipulator)
            joint_pos_robot=self._manipulator.data.joint_pos,
            joint_pos_limits_robot=(
                self._manipulator.data.soft_joint_pos_limits
                if torch.all(
                    torch.isfinite(self._manipulator.data.soft_joint_pos_limits)
                )
                else None
            ),
            joint_pos_end_effector=self._end_effector.data.joint_pos
            if isinstance(self._end_effector, Articulation)
            else None,
            joint_pos_limits_end_effector=(
                self._end_effector.data.soft_joint_pos_limits
                if isinstance(self._end_effector, Articulation)
                and torch.all(
                    torch.isfinite(self._end_effector.data.soft_joint_pos_limits)
                )
                else None
            ),
            # Kinematics
            fk_pos_end_effector=self._tf_end_effector.data.target_pos_source[:, 0, :],
            fk_quat_end_effector=self._tf_end_effector.data.target_quat_source[:, 0, :],
            # Transforms (world frame)
            tf_pos_end_effector=self._tf_end_effector.data.target_pos_w[:, 0, :],
            tf_quat_end_effector=self._tf_end_effector.data.target_quat_w[:, 0, :],
            tf_pos_panel=self._panel.data.root_pos_w,
            tf_quat_panel=self._panel.data.root_quat_w,
            tf_pos_panel_target=self._tf_pos_panel_target,
            tf_quat_panel_target=self._tf_quat_panel_target,
            # Contacts
            contact_forces_robot=self._contacts_robot.data.net_forces_w,  # type: ignore
            contact_forces_end_effector=self._contacts_end_effector.data.net_forces_w
            if isinstance(self._contacts_end_effector, ContactSensor)
            else None,
        )


@torch.jit.script
def _compute_step_return(
    *,
    ## Time
    episode_length: torch.Tensor,
    max_episode_length: int,
    truncate_episodes: bool,
    ## Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    ## States
    # Root
    tf_pos_robot: torch.Tensor,
    tf_quat_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    # IMU
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    # Joints
    joint_pos_robot: torch.Tensor,
    joint_pos_limits_robot: torch.Tensor | None,
    joint_pos_end_effector: torch.Tensor | None,
    joint_pos_limits_end_effector: torch.Tensor | None,
    # Kinematics
    fk_pos_end_effector: torch.Tensor,
    fk_quat_end_effector: torch.Tensor,
    # Transforms (world frame)
    tf_pos_end_effector: torch.Tensor,
    tf_quat_end_effector: torch.Tensor,
    tf_pos_panel: torch.Tensor,
    tf_quat_panel: torch.Tensor,
    tf_pos_panel_target: torch.Tensor,
    tf_quat_panel_target: torch.Tensor,
    # Contacts
    contact_forces_robot: torch.Tensor,
    contact_forces_end_effector: torch.Tensor | None,
) -> StepReturn:
    num_envs = episode_length.size(0)
    dtype = episode_length.dtype
    device = episode_length.device

    ############
    ## States ##
    ############
    ## Root
    tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
    tf_rot6d_robot = rotmat_to_rot6d(tf_rotmat_robot)

    ## Joints
    joint_pos_robot_normalized = (
        scale_transform(
            joint_pos_robot,
            joint_pos_limits_robot[:, :, 0],
            joint_pos_limits_robot[:, :, 1],
        )
        if joint_pos_limits_robot is not None
        else joint_pos_robot
    )
    joint_pos_end_effector_normalized = (
        scale_transform(
            joint_pos_end_effector,
            joint_pos_limits_end_effector[:, :, 0],
            joint_pos_limits_end_effector[:, :, 1],
        )
        if joint_pos_end_effector is not None
        and joint_pos_limits_end_effector is not None
        else (
            joint_pos_end_effector
            if joint_pos_end_effector is not None
            else torch.empty((num_envs, 0), dtype=dtype, device=device)
        )
    )

    ## Kinematics
    fk_rotmat_end_effector = matrix_from_quat(fk_quat_end_effector)
    fk_rot6d_end_effector = rotmat_to_rot6d(fk_rotmat_end_effector)

    ## Transforms (world frame)
    # End-effector -> Panel
    tf_pos_end_effector_to_panel, tf_quat_end_effector_to_panel = (
        subtract_frame_transforms(
            t01=tf_pos_end_effector,
            q01=tf_quat_end_effector,
            t02=tf_pos_panel,
            q02=tf_quat_panel,
        )
    )
    tf_rotmat_end_effector_to_panel = matrix_from_quat(tf_quat_end_effector_to_panel)
    tf_rot6d_end_effector_to_panel = rotmat_to_rot6d(tf_rotmat_end_effector_to_panel)

    # Panel -> Panel target
    tf_pos_panel_to_target, tf_quat_panel_to_target = subtract_frame_transforms(
        t01=tf_pos_panel,
        q01=tf_quat_panel,
        t02=tf_pos_panel_target,
        q02=tf_quat_panel_target,
    )
    tf_rotmat_panel_to_target = matrix_from_quat(tf_quat_panel_to_target)
    tf_rot6d_panel_to_target = rotmat_to_rot6d(tf_rotmat_panel_to_target)

    ## Contacts
    contact_forces_mean_robot = contact_forces_robot.mean(dim=1)
    contact_forces_mean_end_effector = (
        contact_forces_end_effector.mean(dim=1)
        if contact_forces_end_effector is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )
    contact_forces_end_effector = (
        contact_forces_end_effector
        if contact_forces_end_effector is not None
        else torch.empty((num_envs, 0), dtype=dtype, device=device)
    )

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.5
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.mean(
        torch.square(act_current - act_previous), dim=1
    )

    # Penalty: Undesired robot contacts
    WEIGHT_UNDESIRED_ROBOT_CONTACTS = -1.0
    THRESHOLD_UNDESIRED_ROBOT_CONTACTS = 10.0
    penalty_undesired_robot_contacts = WEIGHT_UNDESIRED_ROBOT_CONTACTS * (
        torch.max(torch.norm(contact_forces_robot, dim=-1), dim=1)[0]
        > THRESHOLD_UNDESIRED_ROBOT_CONTACTS
    )

    # Reward: Distance | End-effector <--> Panel
    WEIGHT_DISTANCE_EE_TO_PANEL = 4.0
    TANH_STD_DISTANCE_EE_TO_PANEL = 0.25
    dist_ee_to_panel = torch.norm(tf_pos_end_effector_to_panel, dim=-1)
    reward_distance_ee_to_panel = WEIGHT_DISTANCE_EE_TO_PANEL * (
        1.0 - torch.tanh(dist_ee_to_panel / TANH_STD_DISTANCE_EE_TO_PANEL)
    )

    # Reward: Distance | Panel <--> Target (gradual, guides the whole approach)
    WEIGHT_DISTANCE_PANEL_TO_TARGET_GRADUAL = 8.0
    TANH_STD_DISTANCE_PANEL_TO_TARGET_GRADUAL = 0.5
    dist_panel_to_target = torch.norm(tf_pos_panel_to_target, dim=-1)
    reward_distance_panel_to_target_gradual = WEIGHT_DISTANCE_PANEL_TO_TARGET_GRADUAL * (
        1.0 - torch.tanh(dist_panel_to_target / TANH_STD_DISTANCE_PANEL_TO_TARGET_GRADUAL)
    )

    # Reward: Distance | Panel <--> Target (precise, near the goal)
    WEIGHT_DISTANCE_PANEL_TO_TARGET = 32.0
    TANH_STD_DISTANCE_PANEL_TO_TARGET = 0.05
    reward_distance_panel_to_target = WEIGHT_DISTANCE_PANEL_TO_TARGET * (
        1.0 - torch.tanh(dist_panel_to_target / TANH_STD_DISTANCE_PANEL_TO_TARGET)
    )

    ##################
    ## Terminations ##
    ##################
    # Success: the panel is back on the pedestal, within position and
    # orientation tolerance -- a genuine binary completion signal (most SRB
    # tasks only ever shape a dense reward; this task needs a real one for
    # the harness's verification layer to check against). Tolerances are
    # local (not module-level globals) because torch.jit.script cannot close
    # over a plain Python float defined outside the scripted function.
    SUCCESS_POS_TOLERANCE_M = 0.1
    SUCCESS_ROT_TOLERANCE_COS_HALF_ANGLE = 0.9
    panel_position_restored = dist_panel_to_target < SUCCESS_POS_TOLERANCE_M
    # tf_quat_panel_to_target.w == cos(angle / 2) for the relative rotation;
    # abs() so the sign ambiguity of a quaternion (q and -q are the same
    # rotation) never falsely fails an otherwise-aligned panel.
    panel_orientation_restored = (
        torch.abs(tf_quat_panel_to_target[:, 0]) > SUCCESS_ROT_TOLERANCE_COS_HALF_ANGLE
    )
    termination_success = panel_position_restored & panel_orientation_restored

    # Reward: Success bonus
    WEIGHT_SUCCESS = 64.0
    reward_success = WEIGHT_SUCCESS * termination_success.float()

    # Termination
    termination = termination_success
    # Truncation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "tf_pos_robot": tf_rot6d_robot,
                "tf_rot6d_robot": tf_rot6d_robot,
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "contact_forces_mean_robot": contact_forces_mean_robot,
                "contact_forces_mean_end_effector": contact_forces_mean_end_effector,
                "tf_pos_end_effector_to_panel": tf_pos_end_effector_to_panel,
                "tf_rot6d_end_effector_to_panel": tf_rot6d_end_effector_to_panel,
                "tf_pos_panel_to_target": tf_pos_panel_to_target,
                "tf_rot6d_panel_to_target": tf_rot6d_panel_to_target,
            },
            "state_dyn": {
                "contact_forces_robot": contact_forces_robot,
                "contact_forces_end_effector": contact_forces_end_effector,
            },
            "proprio": {
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
                "fk_pos_end_effector": fk_pos_end_effector,
                "fk_rot6d_end_effector": fk_rot6d_end_effector,
            },
            "proprio_dyn": {
                "joint_pos_robot_normalized": joint_pos_robot_normalized,
                "joint_pos_end_effector_normalized": joint_pos_end_effector_normalized,
            },
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "penalty_undesired_robot_contacts": penalty_undesired_robot_contacts,
            "reward_distance_ee_to_panel": reward_distance_ee_to_panel,
            "reward_distance_panel_to_target_gradual": reward_distance_panel_to_target_gradual,
            "reward_distance_panel_to_target": reward_distance_panel_to_target,
            "reward_success": reward_success,
        },
        termination,
        truncation,
    )
