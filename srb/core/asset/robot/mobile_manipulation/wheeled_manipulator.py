from __future__ import annotations

from typing import Sequence, Type

from srb.core.asset.common import Frame
from srb.core.asset.robot.mobile_manipulation.mobile_manipulator import (
    MobileManipulator,
    MobileManipulatorRegistry,
)
from srb.core.asset.robot.mobile_manipulation.mobile_manipulator_type import (
    MobileManipulatorType,
)


class WheeledManipulator(
    MobileManipulator, mobile_manipulator_entrypoint=MobileManipulatorType.WHEELED
):
    """
    A wheeled mobile manipulator modeled as a **single fused articulation**
    (wheeled base + arm + end effector all belong to the same USD/PhysX
    articulation), as opposed to :class:`~srb.core.asset.robot.mobile_manipulation.combined.ground_manipulator.GroundManipulator`,
    which assembles an independent mobile base and manipulator via a runtime
    fixed-joint assembly.

    Because the entire kinematic chain lives in one articulation, this class
    does not declare a separate ``manipulator`` sub-asset; the whole action
    space (base drive + arm + end effector) is expressed through a single
    :attr:`actions` group on the robot itself.
    """

    ## Frames
    frame_imu: Frame | None = None
    frame_lidar: Frame | None = None
    frame_flange: Frame
    frame_front_camera: Frame
    frame_wrist_camera: Frame

    @classmethod
    def mobile_manipulator_registry(cls) -> Sequence[Type[WheeledManipulator]]:
        return MobileManipulatorRegistry.registry.get(
            MobileManipulatorType.WHEELED, []
        )  # type: ignore
