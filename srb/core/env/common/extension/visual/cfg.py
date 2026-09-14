from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal, Sequence, Tuple

from srb.core.sensor import CameraCfg
from srb.utils.cfg import configclass
from srb.utils.path import SRB_LOGS_DIR

if TYPE_CHECKING:
    from srb._typing import AnyEnvCfg


@configclass
class VisualExtCfg:
    ## Extension overrides
    # Never stack visual environments
    stack: bool = False
    # Use a skydome for the environment
    skydome: Literal["low_res", "high_res"] | bool | None = True
    # Re-render frames on reset for visual observations
    rerender_on_reset: bool = True
    # Always disable debug visualization
    debug_vis: bool = False

    ## Camera sensors
    cameras_cfg: Dict[str, CameraCfg] = MISSING  # type: ignore
    camera_resolution: Tuple[int, int] | None = (64, 64)
    camera_update_period: float = -1.0
    camera_data_types: (
        Sequence[
            Literal[
                "rgb",  # same as "rgba"
                "depth",  # same as "distance_to_image_plane"
                "distance_to_camera",
                "normals",
                "motion_vectors",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
                # "instance_segmentation",
                # "instance_id_segmentation",
                # "bounding_box_2d_tight",
                # "bounding_box_2d_tight_fast",
                # "bounding_box_2d_loose",
                # "bounding_box_2d_loose_fast",
                # "bounding_box_3d",
                # "bounding_box_3d_fast",
            ]
        ]
        | None
    ) = ("rgb", "depth")

    ## Recording
    skip_observations: bool = False
    """Skip constructing the image observations in `_get_observations()`.

    The observations are the RL-agent-facing view of the cameras (per-camera
    channel merge + normalisation, on the sim device). Workflows that consume
    the raw camera buffers elsewhere — `srb agent ros` publishes them over
    ROS 2 straight from `sensor.data.output` — pay ~8 ms per 640x480 camera
    per tick for nothing when the sim device is the CPU; they set this.
    Video recording (`camera_record`) is unaffected.
    """

    camera_record: bool = False
    camera_record_dir: Path | str = SRB_LOGS_DIR.joinpath("camera_videos")

    def wrap(self, env_cfg: "AnyEnvCfg"):
        ## Add camera sensors to the scene
        for camera_key, camera_cfg in self.cameras_cfg.items():
            if self.camera_resolution is not None:
                if (not isinstance(camera_cfg.width, int) or camera_cfg.width <= 0) or (
                    not isinstance(camera_cfg.height, int) or camera_cfg.height <= 0
                ):
                    camera_cfg.width = self.camera_resolution[0]
                    camera_cfg.height = self.camera_resolution[1]
            if self.camera_update_period is not None:
                if camera_cfg.update_period == 0.0:
                    if self.camera_update_period < 0.0:
                        camera_cfg.update_period = env_cfg.agent_rate
                    else:
                        camera_cfg.update_period = self.camera_update_period
            if self.camera_data_types is not None:
                if (
                    len(camera_cfg.data_types) == 1
                    and camera_cfg.data_types[0] == "rgb"
                ):
                    camera_cfg.data_types = self.camera_data_types  # type: ignore
            setattr(env_cfg.scene, camera_key, camera_cfg)
