from dataclasses import MISSING
from typing import Literal

import torch

from srb.core.action import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from srb.core.action.action_group import ActionGroup
from srb.core.action.term import FourWheelSteerActionCfg
from srb.utils import logging
from srb.utils.cfg import configclass


@configclass
class WheeledManipulatorActionGroup(ActionGroup):
    """
    Composite action group for a :class:`~srb.core.asset.robot.mobile_manipulation.wheeled_manipulator.WheeledManipulator`
    (single-articulation wheeled base + arm + end effector).

    The raw action vector is the concatenation of the three constituent
    action terms, in declaration order: base drive velocity, arm
    task-space (relative pose) delta, and the binary end-effector command.

    Because both the base and the arm are driven from the *same* 6-DoF
    teleoperation twist (WASDQE + ZXCVTG), direct keyboard/spacemouse
    teleoperation supports a **mode toggle** (see :meth:`toggle_mode`, wired
    to a dedicated key by the ``teleop`` CLI command) that routes the twist
    to only one of the two mechanisms at a time, freezing the other. This
    does not affect RL/scripted control, which always drives all 9 raw
    action dimensions directly.
    """

    cmd_vel: FourWheelSteerActionCfg = MISSING  # type: ignore
    arm_ik: DifferentialInverseKinematicsActionCfg = MISSING  # type: ignore
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
        assert self.arm_ik.controller.command_type == "pose", (
            "Only pose-based IK control is supported for teleoperation"
        )
        assert self.arm_ik.controller.use_relative_mode, "Only relative mode is supported"

        if self.teleop_mode == "arm":
            base_cmd = torch.zeros(2, device=twist.device, dtype=twist.dtype)
            arm_cmd = twist.clone()
            ## NOTE: negated relative to the raw twist -- verified empirically
            ## that "base_link"'s local +X/+Y (as derived from the URDF/USD)
            ## is opposite to the visually-intended forward/left direction of
            ## the end effector, same as the base's own cmd_vel convention.
            arm_cmd[0] *= -1.0
            arm_cmd[1] *= -1.0
        else:
            base_cmd = twist[:2]
            arm_cmd = torch.zeros(6, device=twist.device, dtype=twist.dtype)

        return torch.cat(
            (
                base_cmd,
                arm_cmd,
                torch.Tensor((-1.0 if event else 1.0,)).to(device=twist.device),
            )
        )
