from dataclasses import MISSING
from typing import TYPE_CHECKING, Type

import isaaclab.utils.math as math_utils
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from srb._typing import AnyEnv
    from srb.core.asset import Articulation


class SwitchableArmAction(ActionTerm):
    """Single-slot arm action term that switches, per env and per step,
    between differential-IK Cartesian-twist control and direct
    joint-position control.

    The mode-switch lives *inside this controller* -- both sub-controllers
    always compute their result every step, and a per-env mode flag (the
    first element of the raw action vector) selects which one is actually
    written to the articulation via ``set_joint_position_target``. This
    mirrors how a real manipulator's own controller supports switching
    control modes on demand, and is required here because Isaac Lab's
    ActionManager applies every *declared* action term unconditionally each
    step -- two independent terms both claiming the same joints (an IK term
    and a joint-position term) would silently fight over which one's
    target survives, with no framework-level "pick one" mechanism.

    Raw action layout (``action_dim`` = ``1 + 6 + len(joint_names)``):
        ``[0]``      mode flag -- > 0.5 selects joint-position mode for
                     that env this step; otherwise IK/twist mode.
        ``[1:7]``    IK sub-command: (vx, vy, vz, wx, wy, wz), scaled by
                     ``cfg.ik_scale`` and applied as a relative-mode
                     differential-IK pose command (mirrors
                     ``DifferentialInverseKinematicsAction``'s
                     ``use_relative_mode=True`` semantics exactly).
        ``[7:]``     Joint-position sub-command: one absolute target per
                     joint in ``cfg.joint_names`` order, scaled by
                     ``cfg.joint_pos_scale`` (radians per unit -- 1.0 means
                     the raw value *is* the physical joint angle).

    Switching mode never causes a jump: the IK sub-controller's target is
    always (re)computed from the end-effector's *current measured* pose
    every step (the same "recompute fresh" property
    ``DifferentialInverseKinematicsAction`` already has), so resuming IK
    mode after a stretch of joint-position mode picks up from wherever the
    arm actually is, not a stale target.
    """

    cfg: "SwitchableArmActionCfg"
    _env: "AnyEnv"
    _asset: "Articulation"

    def __init__(self, cfg: "SwitchableArmActionCfg", env: "AnyEnv"):
        super().__init__(cfg, env)  # type: ignore[arg-type]

        # preserve_order=True: the joint-position sub-command's raw indices
        # must map to cfg.joint_names in the order that list is declared
        # (find_joints defaults to the articulation's own internal joint
        # index order, which need not match).
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)  # type: ignore[assignment]

        body_ids, _ = self._asset.find_bodies(self.cfg.body_name)
        assert len(body_ids) == 1, (
            f"Expected exactly one match for body_name={self.cfg.body_name!r}, got {len(body_ids)}."
        )
        self._body_idx = body_ids[0]

        if self.cfg.base_name:
            base_ids, base_names = self._asset.find_bodies(self.cfg.base_name)
            assert len(base_ids) == 1, (
                f"Expected exactly one match for base_name={self.cfg.base_name!r}, "
                f"got {len(base_ids)}: {base_names}."
            )
            self._base_idx: int | None = base_ids[0]
        else:
            self._base_idx = None

        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]  # type: ignore[union-attr]

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(
                self.cfg.body_offset.pos, device=self.device
            ).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(
                self.cfg.body_offset.rot, device=self.device
            ).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._mode = torch.zeros(self.num_envs, device=self.device)
        self._joint_pos_target_cmd = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 1 + 6 + self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        if self._base_idx is not None:
            base_quat_w = self._asset.data.body_quat_w[:, self._base_idx]
        else:
            base_quat_w = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._mode = self._raw_actions[:, 0]

        ik_raw = self._raw_actions[:, 1:7] * self.cfg.ik_scale
        self._processed_actions[:, 1:7] = ik_raw
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(ik_raw, ee_pos_curr, ee_quat_curr)

        joint_pos_raw = self._raw_actions[:, 7:] * self.cfg.joint_pos_scale
        self._processed_actions[:, 7:] = joint_pos_raw
        self._joint_pos_target_cmd = joint_pos_raw

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            ik_joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            ik_joint_pos_des = joint_pos.clone()

        use_joint_pos_mode = (self._mode > 0.5).unsqueeze(-1)
        joint_pos_des = torch.where(use_joint_pos_mode, self._joint_pos_target_cmd, ik_joint_pos_des)
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids if env_ids is not None else slice(None)] = 0.0

    """
    Helper functions -- identical structure to
    DifferentialInverseKinematicsAction (srb/core/action/term/manipulation/differential_ik.py).
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]

        if self._base_idx is not None:
            base_pos_w = self._asset.data.body_pos_w[:, self._base_idx]
            base_quat_w = self._asset.data.body_quat_w[:, self._base_idx]
        else:
            base_pos_w = self._asset.data.root_pos_w
            base_quat_w = self._asset.data.root_quat_w

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, ee_pos_w, ee_quat_w
        )

        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.jacobian_b
        if self.cfg.body_offset is not None:
            jacobian[:, :3, :] += torch.bmm(
                math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :]
            )
            jacobian[:, 3:, :] = torch.bmm(
                math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :]
            )
        return jacobian


@configclass
class SwitchableArmActionCfg(ActionTermCfg):
    """Configuration for :class:`SwitchableArmAction`."""

    @configclass
    class OffsetCfg:
        """Offset pose from the parent frame (``body_name``) to the child (tool) frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    class_type: Type[ActionTerm] = SwitchableArmAction

    joint_names: list[str] = MISSING  # type: ignore
    """Arm joint names/regex -- also the exact order joint-position commands use."""

    body_name: str = MISSING  # type: ignore
    """Name of the body/frame IK is performed against (the flange)."""

    base_name: str = ""
    """Name of the base body IK poses are expressed relative to. Empty uses the articulation root."""

    body_offset: OffsetCfg | None = None
    """Fixed offset from ``body_name`` to the actual tool-centre-point frame."""

    ik_scale: float = 1.0
    """Scale applied to the IK sub-command before it becomes a relative-mode pose delta."""

    controller: DifferentialIKControllerCfg = MISSING  # type: ignore
    """Differential IK controller configuration for the IK sub-mode."""

    joint_pos_scale: float = 1.0
    """Scale applied to the joint-position sub-command (radians per unit)."""
