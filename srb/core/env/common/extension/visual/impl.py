from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from srb.core.env import InteractiveScene
from srb.utils import logging
from srb.utils.cfg import DEFAULT_DATETIME_FORMAT
from srb.utils.str import sanitize_cam_name

from .cfg import VisualExtCfg


class VisualExt:
    ## Subclass requirements
    scene: InteractiveScene
    cfg: VisualExtCfg

    def __init__(self, cfg: VisualExtCfg, **kwargs):
        self.__cameras = [
            (
                self.scene.sensors[camera_key],
                f"image_{sanitize_cam_name(camera_key)}",
                cfg.cameras_cfg[camera_key].data_types,
                cfg.cameras_cfg[camera_key].spawn.clipping_range,  # type: ignore
            )
            for camera_key in cfg.cameras_cfg.keys()
        ]

        self.__camera_record = cfg.camera_record
        if self.__camera_record:
            logging.info(f"Video recording enabled: {self.__camera_record}")
            self.__camera_record_dir = Path(cfg.camera_record_dir).joinpath(
                f"{datetime.now().strftime(DEFAULT_DATETIME_FORMAT)}"
            )
            self.__camera_record_dir.mkdir(parents=True, exist_ok=True)
            self.__camera_record_ep_frames = [
                defaultdict(lambda: deque()) for _ in range(self.scene.num_envs)
            ]
            self.__camera_record_ep_counter = [0] * self.scene.num_envs
            self.__camera_record_fps = {}
            for camera_sensor, basename, _, _ in self.__cameras:
                update_period = camera_sensor.cfg.update_period
                if update_period > 0:
                    self.__camera_record_fps[basename] = round(1.0 / update_period)
                else:
                    raise ValueError(
                        f"Cannot record video for camera with non-positive update period: {update_period}"
                    )

    def _reset_idx(self, env_ids: Sequence[int]):
        if self.__camera_record:
            self.__save_episode_videos(env_ids)
            for env_id in env_ids:
                self.__camera_record_ep_frames[env_id].clear()
                self.__camera_record_ep_counter[env_id] += 1

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        ## Record videos if enabled
        if self.__camera_record:
            import cv2

            # Get the unmerged images needed for creating video frames.
            unmerged_images = {
                image_key: image
                for camera, image_basename, data_types, clipping_range in self.__cameras
                for image_key, image in construct_observation(
                    image_basename=image_basename,
                    data_types=data_types,
                    clipping_range=clipping_range,  # type: ignore
                    merge_channels=False,
                    **camera.data.output,
                ).items()
            }

            for image_key, image in unmerged_images.items():
                for i in range(image.shape[0]):
                    if image.dtype == torch.uint8:
                        frame = image[i].cpu().numpy()
                    else:
                        frame = (image[i].cpu().numpy() * 255).astype(np.uint8)

                    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[-1] == 1):
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[-1] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    self.__camera_record_ep_frames[i][image_key].append(frame)

        if self.cfg.skip_observations:
            return {}

        return {
            image_key: image
            for camera, image_basename, data_types, clipping_range in self.__cameras
            for image_key, image in construct_observation(
                image_basename=image_basename,
                data_types=data_types,
                clipping_range=clipping_range,  # type: ignore
                **camera.data.output,
            ).items()
        }

    def __save_episode_videos(self, env_ids: Sequence[int]):
        import cv2

        for env_id in env_ids:
            if not self.__camera_record_ep_frames[env_id]:
                return

            for image_key, frames in self.__camera_record_ep_frames[env_id].items():
                if len(frames) < 2:
                    continue

                video_fps = None
                for basename, fps in self.__camera_record_fps.items():
                    if image_key.startswith(basename):
                        video_fps = fps
                        break

                if video_fps is None:
                    logging.warning(
                        f"Could not find FPS for '{image_key}'. Skipping video save."
                    )
                    continue

                video_path = (
                    self.__camera_record_dir
                    / f"{image_key}_env{env_id}_ep{self.__camera_record_ep_counter[env_id]}.mp4"
                )
                logging.info(f"Saving video to {video_path} ({len(frames)} frames)")

                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, video_fps, (width, height)
                )

                if not video_writer.isOpened():
                    logging.warning(f"Could not open video writer for {video_path}")
                    continue

                for frame in frames:
                    video_writer.write(frame)

                video_writer.release()
                logging.info(f"Video saved: {video_path}")


def construct_observation(
    *,
    image_basename: str,
    data_types: Sequence[str],
    clipping_range: Tuple[float, float],
    merge_channels: bool = True,
    as_u8: bool = True,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    processors = _PROCESSORS_U8 if as_u8 else _PROCESSORS_F32
    images = {
        f"{image_basename}_{data_type}": processors[data_type](
            kwargs[data_type], clipping_range
        )
        if data_type == "depth"
        else processors[data_type](kwargs[data_type])
        for data_type in data_types
    }
    if merge_channels:
        return {
            image_basename: torch.cat(
                [images[image_key] for image_key in images.keys()],
                dim=-1,
            )
        }
    else:
        return images


@torch.jit.script
def process_img_u8_as_u8(image: torch.Tensor) -> torch.Tensor:
    return image


@torch.jit.script
def process_img_u8_as_f32(image: torch.Tensor) -> torch.Tensor:
    return image.to(torch.float32) / 255.0


@torch.jit.script
def process_img_f32_as_u8(image: torch.Tensor) -> torch.Tensor:
    return (255.0 * image).to(torch.uint8)


@torch.jit.script
def process_img_f32_as_f32(image: torch.Tensor) -> torch.Tensor:
    return image


@torch.jit.script
def process_rgb_u8(image: torch.Tensor) -> torch.Tensor:
    return image[..., :3]


@torch.jit.script
def process_rgb_f32(image: torch.Tensor) -> torch.Tensor:
    return process_rgb_u8(image).to(torch.float32) / 255.0


@torch.jit.script
def process_depth_f32(
    image: torch.Tensor,
    clipping_range: Tuple[float, float],
) -> torch.Tensor:
    return (
        image.nan_to_num(
            nan=clipping_range[1], posinf=clipping_range[1], neginf=clipping_range[1]
        ).clamp(clipping_range[0], clipping_range[1])
        - clipping_range[0]
    ) / (clipping_range[1] - clipping_range[0])


@torch.jit.script
def process_depth_u8(
    image: torch.Tensor,
    clipping_range: Tuple[float, float],
) -> torch.Tensor:
    return (255.0 * process_depth_f32(image, clipping_range)).to(torch.uint8)


_PROCESSORS_U8 = {
    "rgb": process_rgb_u8,
    "depth": process_depth_u8,
    "distance_to_camera": process_depth_u8,
    "normals": process_rgb_u8,
    # "motion_vectors": ...,
    "semantic_segmentation": process_img_u8_as_u8,
    "instance_segmentation_fast": process_img_u8_as_u8,
    "instance_id_segmentation_fast": process_img_u8_as_u8,
}

_PROCESSORS_F32 = {
    "rgb": process_rgb_f32,
    "depth": process_depth_f32,
    "distance_to_camera": process_depth_f32,
    "normals": process_rgb_f32,
    "motion_vectors": process_img_f32_as_f32,
    "semantic_segmentation": process_img_u8_as_f32,
    "instance_segmentation_fast": process_img_u8_as_f32,
    "instance_id_segmentation_fast": process_img_u8_as_f32,
}
