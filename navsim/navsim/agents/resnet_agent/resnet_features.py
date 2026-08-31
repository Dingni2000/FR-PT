from typing import Dict

import cv2
import numpy as np
import torch

from navsim.agents.resnet_agent.resnet_config import ResNetConfig
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class ResNetFeatureBuilder(AbstractFeatureBuilder):
    """Builds the stitched current-frame camera image and 8-D ego status."""

    def __init__(self, config: ResNetConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Return a cache name that does not collide with ego-status-only cache."""
        return self._config.feature_cache_name

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Build camera and ego-status tensors from NAVSIM AgentInput."""
        frame_index = self._config.camera_history_index

        if frame_index >= len(agent_input.cameras):
            raise IndexError(
                f"camera_history_index={frame_index}, but AgentInput only contains "
                f"{len(agent_input.cameras)} camera frames."
            )
        if frame_index >= len(agent_input.ego_statuses):
            raise IndexError(
                f"camera_history_index={frame_index}, but AgentInput only contains "
                f"{len(agent_input.ego_statuses)} ego-status frames."
            )

        camera_feature = self._build_camera_feature(agent_input, frame_index)
        status_feature = self._build_status_feature(agent_input, frame_index)

        return {
            "camera_feature": camera_feature,
            "status_feature": status_feature,
        }

    def _build_camera_feature(self, agent_input: AgentInput, frame_index: int) -> torch.Tensor:
        """Crop, horizontally stitch and resize cam_l0 / cam_f0 / cam_r0."""
        cameras = agent_input.cameras[frame_index]
        camera_images = {
            "cam_l0": cameras.cam_l0.image,
            "cam_f0": cameras.cam_f0.image,
            "cam_r0": cameras.cam_r0.image,
        }

        for camera_name, image in camera_images.items():
            if image is None:
                raise ValueError(
                    f"{camera_name}.image is None. Check get_sensor_config() and "
                    "the requested camera_history_index."
                )
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(
                    f"{camera_name} must have shape [H, W, C>=3], got {image.shape}."
                )

        # Keep RGB only in case an input unexpectedly contains an alpha channel.
        l0_image = camera_images["cam_l0"][..., :3]
        f0_image = camera_images["cam_f0"][..., :3]
        r0_image = camera_images["cam_r0"][..., :3]

        side_h = self._config.side_crop_height
        side_w = self._config.side_crop_width
        front_h = self._config.front_crop_height

        self._validate_crop(l0_image, side_h, side_w, "cam_l0")
        self._validate_crop(r0_image, side_h, side_w, "cam_r0")
        self._validate_crop(f0_image, front_h, 0, "cam_f0")

        # Same crop-and-stitch convention as the NAVSIM v1.1 Transfuser feature builder.
        l0_crop = l0_image[side_h:-side_h, side_w:-side_w]
        f0_crop = f0_image[front_h:-front_h]
        r0_crop = r0_image[side_h:-side_h, side_w:-side_w]

        if not (l0_crop.shape[0] == f0_crop.shape[0] == r0_crop.shape[0]):
            raise ValueError(
                "Cropped camera heights do not match: "
                f"l0={l0_crop.shape}, f0={f0_crop.shape}, r0={r0_crop.shape}."
            )

        stitched_image = np.concatenate([l0_crop, f0_crop, r0_crop], axis=1)
        resized_image = cv2.resize(
            stitched_image,
            (self._config.image_width, self._config.image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        resized_image = np.ascontiguousarray(resized_image, dtype=np.float32)
        if float(resized_image.max()) > 1.0:
            resized_image /= 255.0

        camera_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).contiguous()

        expected_shape = (3, self._config.image_height, self._config.image_width)
        if tuple(camera_tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected camera tensor shape {tuple(camera_tensor.shape)}, "
                f"expected {expected_shape}."
            )

        return camera_tensor.to(dtype=torch.float32)

    @staticmethod
    def _validate_crop(image: np.ndarray, crop_h: int, crop_w: int, camera_name: str) -> None:
        if image.shape[0] <= 2 * crop_h:
            raise ValueError(
                f"{camera_name} height {image.shape[0]} is too small for crop_h={crop_h}."
            )
        if crop_w > 0 and image.shape[1] <= 2 * crop_w:
            raise ValueError(
                f"{camera_name} width {image.shape[1]} is too small for crop_w={crop_w}."
            )

    @staticmethod
    def _build_status_feature(agent_input: AgentInput, frame_index: int) -> torch.Tensor:
        """Preserve the original EgoStatusMLP feature order: velocity, acceleration, command."""
        ego_status = agent_input.ego_statuses[frame_index]

        velocity = torch.as_tensor(ego_status.ego_velocity, dtype=torch.float32).flatten()
        acceleration = torch.as_tensor(ego_status.ego_acceleration, dtype=torch.float32).flatten()
        driving_command = torch.as_tensor(ego_status.driving_command, dtype=torch.float32).flatten()

        status_feature = torch.cat([velocity, acceleration, driving_command], dim=0)
        if status_feature.numel() != 8:
            raise ValueError(
                "Expected 8 ego-status values from velocity + acceleration + driving_command, "
                f"but received {status_feature.numel()} values with shapes "
                f"velocity={tuple(velocity.shape)}, acceleration={tuple(acceleration.shape)}, "
                f"driving_command={tuple(driving_command.shape)}."
            )

        return status_feature


class ResNetTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Builds the future ego trajectory target."""

    def __init__(self, config: ResNetConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "camera_status_trajectory_target_v1"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._config.trajectory_sampling.num_poses
        )
        trajectory = torch.as_tensor(future_trajectory.poses, dtype=torch.float32)
        return {"trajectory": trajectory}
