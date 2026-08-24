"""Run with: python -m navsim.agents.camera_status.smoke_test"""
import torch

from navsim.agents.camera_status.camera_status_config import CameraStatusConfig
from navsim.agents.camera_status.camera_status_model import CameraStatusTrajectoryModel


def main() -> None:
    config = CameraStatusConfig(
        image_pretrained=False,
        return_intermediate_features=True,
    )
    model = CameraStatusTrajectoryModel(config).eval()

    batch_size = 1
    features = {
        "camera_feature": torch.rand(
            batch_size,
            3,
            config.image_height,
            config.image_width,
        ),
        "status_feature": torch.randn(batch_size, config.status_input_dim),
    }

    with torch.no_grad():
        predictions = model(features)

    expected_trajectory_shape = (
        batch_size,
        config.trajectory_sampling.num_poses,
        3,
    )
    assert tuple(predictions["trajectory"].shape) == expected_trajectory_shape
    assert tuple(predictions["image_embedding"].shape) == (
        batch_size,
        config.image_embedding_dim,
    )
    assert tuple(predictions["fusion_z"].shape) == (
        batch_size,
        config.image_embedding_dim,
    )
    assert tuple(predictions["planning_z2"].shape) == (
        batch_size,
        config.planning_hidden_dim,
    )

    print("Smoke test passed.")
    for name, tensor in predictions.items():
        print(f"{name:>20s}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


if __name__ == "__main__":
    main()
