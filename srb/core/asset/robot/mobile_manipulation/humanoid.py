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


class Humanoid(
    MobileManipulator, mobile_manipulator_entrypoint=MobileManipulatorType.HUMANOID
):
    ## Frames
    frame_imu: Frame | None = None
    frame_lidar: Frame | None = None
    frame_front_camera: Frame

    ## Links
    regex_feet_links: str

    @classmethod
    def mobile_manipulator_registry(cls) -> Sequence[Type[Humanoid]]:
        return MobileManipulatorRegistry.registry.get(
            MobileManipulatorType.HUMANOID, []
        )  # type: ignore
