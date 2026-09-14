import math
from typing import TYPE_CHECKING, List, Sequence, Tuple, Type

import torch

from srb.core.manager import ActionTerm, ActionTermCfg
from srb.utils.cfg import configclass

if TYPE_CHECKING:
    from srb._typing import AnyEnv
    from srb.core.asset import Articulation


class FourWheelSteerAction(ActionTerm):
    """
    True 4-wheel independent-steering / independent-drive (4WIS/4WID) base
    controller.

    Unlike :class:`~srb.core.action.term.mobile.wheeled_drive.WheeledDriveAction`
    (which targets rocker-bogie-style rovers with a limited steering range and
    deliberately damps the steering angle to ~30% during simultaneous
    translation+rotation), this term computes the **exact** rigid-body
    instantaneous velocity of each corner wheel and steers/drives it toward
    that velocity directly. This supports combined forward+turn commands
    (e.g. simultaneous "W"+"A") at full authority, using the wheel's entire
    available steering range.

    Command: ``(v, omega)`` -- forward linear velocity [m/s] and yaw rate
    [rad/s] of the chassis, expressed in the chassis' own body frame
    (+X forward, +Y left).
    """

    cfg: "FourWheelSteerActionCfg"
    _env: "AnyEnv"
    _asset: "Articulation"

    def __init__(self, cfg: "FourWheelSteerActionCfg", env: "AnyEnv"):
        super().__init__(cfg, env)

        self._steering_joint_indices = self._asset.find_joints(
            self.cfg.steering_joint_names, preserve_order=True
        )[0]
        self._drive_joint_indices = self._asset.find_joints(
            self.cfg.drive_joint_names, preserve_order=True
        )[0]
        assert len(self._steering_joint_indices) == len(self.cfg.wheel_positions)
        assert len(self._drive_joint_indices) == len(self.cfg.wheel_positions)

        self._wheel_pos = torch.tensor(
            self.cfg.wheel_positions, device=self.device, dtype=torch.float32
        )  # (N, 2): (x, y) of each wheel in the chassis body frame

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 2  # (linear_velocity, angular_velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        self._processed_actions[:, 0] *= self.cfg.scale_linear
        self._processed_actions[:, 1] *= self.cfg.scale_angular
        if self.cfg.max_linear_velocity is not None:
            self._processed_actions[:, 0].clamp_(
                -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity
            )
        if self.cfg.max_angular_velocity is not None:
            self._processed_actions[:, 1].clamp_(
                -self.cfg.max_angular_velocity, self.cfg.max_angular_velocity
            )

    def apply_actions(self):
        v = self.processed_actions[:, 0].unsqueeze(-1)  # (num_envs, 1)
        w = self.processed_actions[:, 1].unsqueeze(-1)  # (num_envs, 1)
        x = self._wheel_pos[:, 0].unsqueeze(0)  # (1, N)
        y = self._wheel_pos[:, 1].unsqueeze(0)  # (1, N)

        ## Exact rigid-body velocity of each wheel: v_wheel = v_com + omega x r
        vx = v - w * y  # (num_envs, N)
        vy = w * x  # (num_envs, N)

        steer_angle = torch.atan2(vy, vx)
        speed = torch.sqrt(vx * vx + vy * vy)

        ## Keep the steering angle within (-90, 90] deg by allowing the drive
        ## wheel to spin in reverse instead of physically over-rotating past
        ## a right angle (standard car-steering convention)
        too_pos = steer_angle > (math.pi / 2)
        too_neg = steer_angle < (-math.pi / 2)
        steer_angle = torch.where(too_pos, steer_angle - math.pi, steer_angle)
        steer_angle = torch.where(too_neg, steer_angle + math.pi, steer_angle)
        speed = torch.where(too_pos | too_neg, -speed, speed)

        steer_angle = torch.clamp(
            steer_angle, -self.cfg.max_steering_angle, self.cfg.max_steering_angle
        )

        ## Steer-before-drive: scale each wheel's drive speed by how closely
        ## its CURRENT measured steering angle already matches the
        ## newly-commanded target, instead of driving at full speed the
        ## instant a new target is set. The steering joints are position-
        ## controlled (finite velocity limit, ~0.8s to swing up to 90 deg on
        ## this robot) while the drive joints are velocity-controlled and
        ## would otherwise receive their target in the same step -- so a
        ## large in-place turn drove every wheel at full speed while it was
        ## still mid-swing, pushing in inconsistent directions until
        ## steering caught up (F19,
        ## docs/lunar_bot_capability_set_plan.md §3.4). Standard 4WIS
        ## practice: zero drive authority while more than 90 deg off the
        ## commanded steering angle, full authority once aligned.
        current_steer_angle = self._asset.data.joint_pos[:, self._steering_joint_indices]
        steer_alignment = torch.cos(steer_angle - current_steer_angle).clamp(min=0.0)
        drive_velocity = (speed / self.cfg.wheel_radius) * steer_alignment

        self._asset.set_joint_position_target(
            steer_angle, joint_ids=self._steering_joint_indices
        )
        self._asset.set_joint_velocity_target(
            drive_velocity, joint_ids=self._drive_joint_indices
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)


@configclass
class FourWheelSteerActionCfg(ActionTermCfg):
    class_type: Type = FourWheelSteerAction

    scale_linear: float = 1.0
    scale_angular: float = 1.0

    steering_joint_names: List[str] = []
    drive_joint_names: List[str] = []

    ## (x, y) position of each wheel in the chassis body frame [m],
    ## in the same order as steering_joint_names/drive_joint_names
    wheel_positions: List[Tuple[float, float]] = []

    wheel_radius: float = 0.1
    max_steering_angle: float = math.pi / 2

    max_linear_velocity: float | None = None
    max_angular_velocity: float | None = None
