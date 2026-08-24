from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

try:
    from ..rvs_cpt import route_rvs_act, linear_reverseCom
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from rvs_cpt import route_rvs_act, linear_reverseCom


class EgoStatusFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self):
        """Initializes the feature builder."""
        pass

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "ego_status_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        ego_status = agent_input.ego_statuses[-1]
        velocity = torch.tensor(ego_status.ego_velocity)
        acceleration = torch.tensor(ego_status.ego_acceleration)
        driving_command = torch.tensor(ego_status.driving_command)
        ego_status_feature = torch.cat([velocity, acceleration, driving_command], dim=-1)
        return {"ego_status": ego_status_feature}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class EgoStatusMLPAgent(AbstractAgent):
    """EgoStatMLP agent interface."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        hidden_layer_dim: int,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes the agent interface for EgoStatusMLP.
        :param trajectory_sampling: trajectory sampling specification.
        :param hidden_layer_dim: dimensionality of hidden layer.
        :param lr: learning rate during training.
        :param checkpoint_path: optional checkpoint path as string, defaults to None
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = checkpoint_path

        self._lr = lr

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, self._trajectory_sampling.num_poses * 3),
        )
        self.recons_map = {'recons_z3':5, 'recons_z2':3, 'recons_z1':1}  

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path is None:
            return

        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        checkpoint = torch.load(self._checkpoint_path, map_location=map_location)
        state_dict: Dict[str, Any] = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [EgoStatusFeatureBuilder()]

    # def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # original
    #     """Inherited, see superclass."""
    #     poses: torch.Tensor = self._mlp(features["ego_status"])
    #     return {"trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]: 
        """Inherited, see superclass."""
        assert len(self._mlp) == 7
        z1 = self._mlp[0](features["ego_status"])
        a1 = self._mlp[1](z1)
        z2 = self._mlp[2](a1)
        a2 = self._mlp[3](z2)
        z3 = self._mlp[4](a2)
        a3 = self._mlp[5](z3)
        poses = self._mlp[6](a3)
        return {'z1':z1,'z2':z2, 'z3':z3, "trajectory": poses.reshape(-1, self._trajectory_sampling.num_poses, 3)}
    
    def get_recons_fea(self, input_data: Dict[str, torch.Tensor], label_data:torch.Tensor, recons_key=None):
        res = self(input_data)
        assert label_data.shape == res['trajectory'].shape, f"label size={label_data.shape}, out traj size={res['trajectory'].shape}"
        label_flat = label_data.reshape(label_data.shape[0], -1)
        
        a3 = self._mlp[5](res['z3'])
        recons_a3 = linear_reverseCom(a3, label_flat, self._mlp[6])
        recons_z3 = route_rvs_act(self._mlp[5], res['z3'], recons_a3)
        if recons_key == 'recons_z3': return recons_z3

        a2 = self._mlp[3](res['z2'])
        recons_a2 = linear_reverseCom(a2, recons_z3, self._mlp[4])
        recons_z2 = route_rvs_act(self._mlp[3], res['z2'], recons_a2)
        if recons_key == 'recons_z2': return recons_z2

        a1 = self._mlp[1](res['z1'])
        recons_a1 = linear_reverseCom(a1, recons_z2, self._mlp[2])
        recons_z1 = route_rvs_act(self._mlp[1], res['z1'], recons_a1)
        if recons_key == 'recons_z1': return recons_z1
        return {'gt_trajectory':label_data, 'recons_z3':recons_z3, 'recons_z2':recons_z2, 'recons_z1':recons_z1 }

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)
    
    def set_recons_param(self, recons_key):
        for p in self.parameters():
            p.requires_grad = True
        freeze_from = self.recons_map.get(recons_key, None)
        if freeze_from is None:
            raise ValueError(f"Unknown recons_key={recons_key}; available keys={self.get_fea_name()}")
        for _, layer in list(self._mlp.named_children())[freeze_from:]: 
            for p in layer.parameters():
                p.requires_grad = False
    
    def get_fea_name(self):
        return list(self.recons_map.keys())
    
if __name__ == '__main__':
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    model = EgoStatusMLPAgent(ts, 16, 0.001)
    named_children = list(model._mlp.named_children())
    for idx, (n, layer) in enumerate(named_children):
        print(idx, n, layer)
