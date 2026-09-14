from dataclasses import MISSING
from typing import Literal

import torch

from srb.core.action import BinaryJointPositionActionCfg
from srb.core.action.action_group import ActionGroup
from srb.core.action.term import FourWheelSteerActionCfg, SwitchableArmActionCfg
from srb.utils import logging
from srb.utils.cfg import configclass


@configclass
class WheeledManipulatorActionGroup(ActionGroup):
    """
    Composite action group for a :class:`~srb.core.asset.robot.mobile_manipulation.wheeled_manipulator.WheeledManipulator`
    (single-articulation wheeled base + arm + end effector).

    The raw action vector is the concatenation of the three constituent
    action terms, in declaration order: base drive velocity, the arm's
    switchable task-space/joint-position command (see
    :class:`~srb.core.action.term.manipulation.switchable_arm.SwitchableArmAction`
    -- a single slot, not two, since two independent terms both claiming
    the arm's joints would silently fight over the applied target every
    step), and the binary end-effector command.

    Because both the base and the arm are driven from the *same* 6-DoF
    teleoperation twist (WASDQE + ZXCVTG), direct keyboard/spacemouse
    teleoperation supports a **mode toggle** (see :meth:`toggle_mode`, wired
    to a dedicated key by the ``teleop`` CLI command) that routes the twist
    to only one of the two mechanisms at a time, freezing the other. This
    does not affect RL/scripted control, which always drives all raw
    action dimensions directly -- for the arm slot specifically, the first
    element of its sub-vector is the mode flag described on
    ``SwitchableArmAction``, not a frozen/live toggle the way teleop uses.
    """

    cmd_vel: FourWheelSteerActionCfg = MISSING  # type: ignore
    arm: SwitchableArmActionCfg = MISSING  # type: ignore
    hand: BinaryJointPositionActionCfg = MISSING  # type: ignore

    ## Teleoperation-only state: which mechanism the shared twist currently drives
    teleop_mode: Literal["base", "arm"] = "base"

    def toggle_mode(self) -> str:
        self.teleop_mode = "arm" if self.teleop_mode == "base" else "base"
        logging.info(
            f'Teleop mode switched to "{self.teleop_mode}"'
            + (
                " (WASDQE/ZXCVTG now drives the base; the arm is frozen)"
                if self.teleop_mode == "base"
                else " (WASDQE/ZXCVTG now drives the arm end effector; the base is frozen)"
            )
        )
        return self.teleop_mode

    def map_cmd_to_action(self, twist: torch.Tensor, event: bool) -> torch.Tensor:
        assert self.arm.controller.command_type == "pose", (
            "Only pose-based IK control is supported for teleoperation"
        )
        assert self.arm.controller.use_relative_mode, "Only relative mode is supported"

        # Teleop always drives the arm (when driving it at all) in IK/twist
        # mode -- mode flag 0.0 -- never joint-position mode, so the 7
        # joint-position sub-slots are always an ignored placeholder here.
        arm_mode_flag = torch.zeros(1, device=twist.device, dtype=twist.dtype)
        joint_pos_placeholder = torch.zeros(7, device=twist.device, dtype=twist.dtype)

        if self.teleop_mode == "arm":
            base_cmd = torch.zeros(2, device=twist.device, dtype=twist.dtype)
            ik_cmd = twist.clone()
            ## NOTE: negated relative to the raw twist -- verified empirically
            ## that "base_link"'s local +X/+Y (as derived from the URDF/USD)
            ## is opposite to the visually-intended forward/left direction of
            ## the end effector, same as the base's own cmd_vel convention.
            ik_cmd[0] *= -1.0
            ik_cmd[1] *= -1.0
        else:
            # FourWheelSteerAction's 2-DoF command is (v, omega) -- forward
            # linear speed and yaw rate (its own docstring) -- NOT the first
            # two slots of the 6-DoF twist [vx,vy,vz,wx,wy,wz]. `twist[:2]`
            # silently took (vx, vy), so every angular.z command (a pure
            # in-place turn included) was dropped before it ever reached
            # FourWheelSteerAction.process_actions() -- confirmed live via
            # FourWheelSteerAction.apply_actions() reading v=w=0.0 while a
            # real angular.z=0.4 cmd_vel was arriving on the ROS topic.
            # Found while live-verifying F19 (steer-before-drive) in
            # four_wheel_steer.py -- a separate, pre-existing bug, not
            # introduced by that fix, but a direct blocker to observing it
            # through the real BODY_TWIST dispatch path.
            base_cmd = twist[[0, 5]]
            ik_cmd = torch.zeros(6, device=twist.device, dtype=twist.dtype)

        return torch.cat(
            (
                base_cmd,
                arm_mode_flag,
                ik_cmd,
                joint_pos_placeholder,
                torch.Tensor((-1.0 if event else 1.0,)).to(device=twist.device),
            )
        )
