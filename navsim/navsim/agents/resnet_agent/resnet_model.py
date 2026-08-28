import math
from typing import Dict, Tuple

import torch
from torch import nn

from navsim.agents.resnet_agent.resnet_config import ResNetConfig
from navsim.agents.resnet_agent.resnet34_feature_backbone import ResNet34FeatureBackbone

try:
    from ..rvs_cpt import composite_reverseCom, linear_reverseCom, route_rvs_act, layer_norm_reverseCom
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from rvs_cpt import composite_reverseCom, linear_reverseCom, route_rvs_act, layer_norm_reverseCom


class ResNetTrajectoryModel(nn.Module):
    """ResNet34 image encoder with ego-status-conditioned vector modulation."""
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self._config = config

        if config.image_architecture != "resnet34":
            raise ValueError(
                "This self-contained feature backbone currently supports only resnet34, "
                f"but received '{config.image_architecture}'."
            )

        self.image_backbone = ResNet34FeatureBackbone(
            checkpoint_path=config.image_checkpoint_path,
            load_imagenet_checkpoint=config.load_imagenet_checkpoint,
            freeze=config.freeze_image_encoder,
        )
        backbone_channels = self.image_backbone.output_channels

        self.image_pool = nn.AdaptiveAvgPool2d(
            (config.spatial_pool_height, config.spatial_pool_width)
        )
        pooled_image_dim = (
            backbone_channels
            * config.spatial_pool_height
            * config.spatial_pool_width
        )
        self.image_projection = nn.Sequential(
            nn.Linear(pooled_image_dim, config.image_embedding_dim),
            nn.LayerNorm(config.image_embedding_dim),
            nn.GELU(),
        )

        self.status_encoder = nn.Sequential(
            nn.Linear(config.status_input_dim, config.status_intermediate_dim),
            nn.LayerNorm(config.status_intermediate_dim),
            nn.GELU(),
            nn.Linear(config.status_intermediate_dim, config.status_context_dim),
            nn.LayerNorm(config.status_context_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(
            config.status_context_dim,
            config.image_embedding_dim,
        )
        self.beta_head = nn.Linear(
            config.status_context_dim,
            config.image_embedding_dim,
        )

        self.planning_layer1 = nn.Sequential(
            nn.Linear(config.image_embedding_dim, config.planning_hidden_dim),
            nn.LayerNorm(config.planning_hidden_dim),
            nn.LeakyReLU(config.leaky_relu_slope, inplace=False),
        )
        self.planning_layer2 = nn.Sequential(
            nn.Linear(config.planning_hidden_dim, config.planning_hidden_dim),
            nn.LayerNorm(config.planning_hidden_dim),
            nn.LeakyReLU(config.leaky_relu_slope, inplace=False),
        )
        self.trajectory_head = nn.Linear(
            config.planning_hidden_dim,
            config.trajectory_sampling.num_poses * 3,
        )

        image_mean = torch.tensor(
            config.image_mean,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        image_std = torch.tensor(
            config.image_std,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean, persistent=False)
        self.register_buffer("image_std", image_std, persistent=False)
        # FRPT strips the "recons_" prefix to find the matching forward key.
        # ResNet stage features are returned as "resnet.z*", so keep that
        # namespace here instead of the MLP-style "recons_z*" names.
        self.recons_map = {#需要重建特征对应的编号
            "recons_resnet.z4": 1,
            "recons_resnet.z3": 1,
            # "recons_resnet.z2": 1,
            # "recons_resnet.z1": 1,
            # "recons_resnet.z0": 1,
            "recons_fusion_z": 6,
            "recons_planning_z1": 7,
            "recons_planning_z2": 8,
        }
        self._initialize_modulation_heads()

    def _initialize_modulation_heads(self) -> None:
        """Start close to image-only behavior while retaining status gradients."""
        nn.init.normal_(self.gamma_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.normal_(self.beta_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "camera_feature" not in features or "status_feature" not in features:
            raise KeyError(
                "features must contain both 'camera_feature' and 'status_feature'."
            )

        camera_feature = features["camera_feature"].to(dtype=torch.float32)
        status_feature = features["status_feature"].to(dtype=torch.float32)
        if camera_feature.ndim != 4 or camera_feature.shape[1] != 3:
            raise ValueError(
                "camera_feature must have shape [B, 3, H, W], "
                f"got {tuple(camera_feature.shape)}."
            )
        if (
            status_feature.ndim != 2
            or status_feature.shape[-1] != self._config.status_input_dim
        ):
            raise ValueError(
                f"status_feature must have shape [B, {self._config.status_input_dim}], "
                f"got {tuple(status_feature.shape)}."
            )

        if self._config.normalize_image:
            camera_feature = (camera_feature - self.image_mean) / self.image_std

        image_feature_map, backbone_features = self.image_backbone(#layer输出的特征，并且返回设定的每层feature
            camera_feature,
            capture_mode=self._config.backbone_feature_capture,
            retain_gradients=self._config.retain_backbone_feature_gradients,
        )

        expected_channels = self.image_backbone.output_channels
        if image_feature_map.ndim != 4 or image_feature_map.shape[1] != expected_channels:
            raise RuntimeError(
                "Unexpected final ResNet feature shape: "
                f"{tuple(image_feature_map.shape)}; expected [B, {expected_channels}, H, W]."
            )

        pooled_image_feature = self.image_pool(image_feature_map)
        pooled_image_feature = pooled_image_feature.flatten(start_dim=1)
        image_embedding = self.image_projection(pooled_image_feature)

        status_context = self.status_encoder(status_feature)
        gamma = self.gamma_head(status_context)
        beta = self.beta_head(status_context)

        scale = 1.0 + self._config.modulation_scale * torch.tanh(gamma)#进行特征的调制和融合
        fusion_z = scale * image_embedding + beta

        planning_z1 = self.planning_layer1(fusion_z)
        planning_z2 = self.planning_layer2(planning_z1)
        trajectory_flat = self.trajectory_head(planning_z2)
        trajectory = trajectory_flat.reshape(
            -1,
            self._config.trajectory_sampling.num_poses,
            3,
        )

        predictions: Dict[str, torch.Tensor] = {"trajectory": trajectory} # 预测轨迹, 必需的key

        if self._config.return_intermediate_features:
            predictions.update(
                {
                    "image_feature_map": image_feature_map,
                    "image_embedding": image_embedding,
                    "status_context": status_context,
                    "gamma": gamma,
                    "beta": beta,
                    "fusion_z": fusion_z,
                    "planning_z1": planning_z1,
                    "planning_z2": planning_z2,
                }
            )

        # Keep the dictionary flat so existing HDF5 feature-saving code can iterate
        # over predictions without special handling for nested dictionaries.
        for feature_name, feature_tensor in backbone_features.items():
            predictions[f"resnet.{feature_name}"] = feature_tensor
        return predictions

    @staticmethod
    def _reverse_linear_norm_activation(
        front_fea: torch.Tensor,
        back_fea: torch.Tensor,
        layer: nn.Sequential,
    ) -> torch.Tensor:
        assert len(layer) == 3, f"Expected Sequential(Linear, LayerNorm, Activation), got {layer}."
        linear, norm, activation = layer
        if not isinstance(linear, nn.Linear) or not isinstance(norm, nn.LayerNorm):
            raise TypeError(
                "Reverse helper requires Linear -> LayerNorm -> Activation.")
        linear_z = linear(front_fea)
        norm_z = norm(linear_z)
        recons_norm_z = route_rvs_act(activation, norm_z, back_fea)
        recons_linear_z = layer_norm_reverseCom(recons_norm_z, norm, front_fea=linear_z)
        return linear_reverseCom(front_fea, recons_linear_z, linear)
        # return composite_reverseCom(
        #     front_fea,
        #     back_fea,
        #     layer,
        #     iter=20,
        #     damping=0.8,
        #     max_step_norm=10.0,
        #     tol=1e-6,
        #     method="auto",
        #     jacobian_elem_limit=5e7,
        # )

    @staticmethod
    def _adaptive_avgpool_reverse(
        pooled: torch.Tensor,
        ref_feature: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        _, _, input_h, input_w = ref_feature.shape
        output_h, output_w = output_size
        if pooled.shape[-2:] != output_size:
            raise ValueError(
                f"Expected pooled spatial shape {output_size}, got {tuple(pooled.shape[-2:])}."
            )

        reconstructed = torch.zeros_like(ref_feature)
        counts = torch.zeros(
            (1, 1, input_h, input_w),
            dtype=ref_feature.dtype,
            device=ref_feature.device,
        )
        for out_y in range(output_h):
            start_y = math.floor(out_y * input_h / output_h)
            end_y = math.ceil((out_y + 1) * input_h / output_h)
            for out_x in range(output_w):
                start_x = math.floor(out_x * input_w / output_w)
                end_x = math.ceil((out_x + 1) * input_w / output_w)
                reconstructed[:, :, start_y:end_y, start_x:end_x] += pooled[
                    :, :, out_y : out_y + 1, out_x : out_x + 1
                ]
                counts[:, :, start_y:end_y, start_x:end_x] += 1

        return reconstructed / counts.clamp_min(1)

    def _forward_backbone_for_recons(
        self,
        camera_feature: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
        resnet = self.image_backbone.resnet
        z_stem = resnet.relu(resnet.bn1(resnet.conv1(camera_feature)))
        z0 = resnet.maxpool(z_stem)

        out = z0
        block_inputs: Dict[Tuple[int, int], torch.Tensor] = {}
        stage_features: Dict[str, torch.Tensor] = {"z0": z0}
        for layer_idx, layer in enumerate(
            (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4),
            start=1,
        ):
            for block_idx, block in enumerate(layer):
                block_inputs[(layer_idx, block_idx)] = out
                out = block(out)
            stage_features[f"z{layer_idx}"] = out

        return stage_features, block_inputs

    @staticmethod
    def _reverse_resnet_layer(
        layer: nn.Sequential,
        layer_idx: int,
        block_inputs: Dict[Tuple[int, int], torch.Tensor],
        back_fea: torch.Tensor,
        residual_iter: int=20,
    ) -> torch.Tensor:
        current = back_fea
        for block_idx in reversed(range(len(layer))):
            block = layer[block_idx]
            front_ref = block_inputs[(layer_idx, block_idx)]
            current = composite_reverseCom(
                front_ref,
                current,
                block,
                iter=residual_iter,
                damping=0.9,
                max_step_norm=10.0,
                tol=1e-6,
            )
        return current

    def get_recons_fea(self, input_data, gt_data,recons_key=None,residual_iter=20,):
        device = next(self.parameters()).device
        camera_feature = input_data["camera_feature"].to(
            device=device,
            dtype=torch.float32,
        )
        status_feature = input_data["status_feature"].to(
            device=device,
            dtype=torch.float32,
        )

        if self._config.normalize_image:
            camera_feature = (camera_feature - self.image_mean) / self.image_std

        stage_features, block_inputs = self._forward_backbone_for_recons(camera_feature)
        image_feature_map = stage_features["z4"]

        pooled_image_map = self.image_pool(image_feature_map)
        pooled_image_feature = pooled_image_map.flatten(start_dim=1)

        # image_embedding = self.image_projection(pooled_image_feature)
        image_linear_z = self.image_projection[0](pooled_image_feature)
        image_norm_z = self.image_projection[1](image_linear_z)
        image_embedding = self.image_projection[2](image_norm_z)

        status_context = self.status_encoder(status_feature)
        gamma = self.gamma_head(status_context)
        beta = self.beta_head(status_context)
        scale = 1.0 + self._config.modulation_scale * torch.tanh(gamma)
        fusion_z = scale * image_embedding + beta

        planning_z1 = self.planning_layer1(fusion_z)
        planning_z2 = self.planning_layer2(planning_z1)

        gt_flat = gt_data.to(device=device, dtype=planning_z2.dtype).reshape(gt_data.shape[0], -1)
        recons_planning_z2 = linear_reverseCom(planning_z2, gt_flat, self.trajectory_head)
        if recons_key == 'recons_planning_z2': return recons_planning_z2

        recons_planning_z1 = self._reverse_linear_norm_activation(
            planning_z1,
            recons_planning_z2,
            self.planning_layer2,
        )
        if recons_key == 'recons_planning_z1': return recons_planning_z1

        recons_fusion_z = self._reverse_linear_norm_activation(
            fusion_z,
            recons_planning_z1,
            self.planning_layer1,
        )
        if recons_key == 'recons_fusion_z': return recons_fusion_z

        recons_image_embedding = (recons_fusion_z - beta) / scale.clamp_min(1e-6)
        # recons_pooled_image_feature = composite_reverseCom(
        #     pooled_image_feature,
        #     recons_image_embedding,
        #     self.image_projection,
        #     iter=residual_iter,
        #     damping=0.8,
        #     max_step_norm=10.0,
        #     tol=1e-6,
        #     method="auto",
        #     jacobian_elem_limit=5e7,
        # )
        recons_image_norm_z = route_rvs_act(
            self.image_projection[2],
            image_norm_z,
            recons_image_embedding,
        )
        recons_image_linear_z = layer_norm_reverseCom(
            recons_image_norm_z,
            self.image_projection[1],
            front_fea=image_linear_z,
        )
        recons_pooled_image_feature = linear_reverseCom(
            pooled_image_feature,
            recons_image_linear_z,
            self.image_projection[0],
        )
        recons_pooled_image_map = recons_pooled_image_feature.view_as(pooled_image_map)
        recons_resnet_z4 = self._adaptive_avgpool_reverse(
            recons_pooled_image_map,
            image_feature_map,
            self.image_pool.output_size,
        )
        if recons_key == 'recons_resnet.z4': return recons_resnet_z4

        resnet = self.image_backbone.resnet
        recons_resnet_z3 = self._reverse_resnet_layer(
            resnet.layer4,
            4,
            block_inputs,
            recons_resnet_z4,
            residual_iter,
        )
        if recons_key == 'recons_resnet.z3': return recons_resnet_z3

        # recons_resnet_z2 = self._reverse_resnet_layer(
        #     resnet.layer3,
        #     3,
        #     block_inputs,
        #     recons_resnet_z3,
        #     residual_iter,
        # )
        # recons_resnet_z1 = self._reverse_resnet_layer(
        #     resnet.layer2,
        #     2,
        #     block_inputs,
        #     recons_resnet_z2,
        #     residual_iter,
        # )
        # recons_resnet_z0 = self._reverse_resnet_layer(
        #     resnet.layer1,
        #     1,
        #     block_inputs,
        #     recons_resnet_z1,
        #     residual_iter,
        # )

        return {
            "gt_trajectory": gt_data,
            "recons_resnet.z4": recons_resnet_z4,
            "recons_resnet.z3": recons_resnet_z3,
            # "recons_resnet.z2": recons_resnet_z2,
            # "recons_resnet.z1": recons_resnet_z1,
            # "recons_resnet.z0": recons_resnet_z0,
            "recons_fusion_z": recons_fusion_z,
            "recons_planning_z1": recons_planning_z1,
            "recons_planning_z2": recons_planning_z2,
        }
    
    def set_recons_param(self, recons_key):
        """only set params before recons_key (in other words, the params needed to compute the recons_key) 
            as requires_grad = True. All other params as requires_grad = False
            recons_key can be any in self.get_fea_name()"""
        if recons_key not in self.recons_map:
            raise ValueError(
                f"Unknown recons_key={recons_key}; available keys={self.get_fea_name()}")

        for param in self.parameters():
            param.requires_grad_(False)

        def _enable(module: nn.Module) -> None:
            for param in module.parameters():
                param.requires_grad_(True)

        resnet = self.image_backbone.resnet
        stem_modules = (resnet.conv1, resnet.bn1)
        resnet_stage_modules = {
            "recons_resnet.z0": stem_modules,
            "recons_resnet.z1": (*stem_modules, resnet.layer1),
            "recons_resnet.z2": (*stem_modules, resnet.layer1, resnet.layer2),
            "recons_resnet.z3": (*stem_modules, resnet.layer1, resnet.layer2, resnet.layer3,),
            "recons_resnet.z4": (*stem_modules, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,),
        }
        resnet_forward_modules = resnet_stage_modules["recons_resnet.z4"]

        if recons_key in resnet_stage_modules:
            for module in resnet_stage_modules[recons_key]:
                _enable(module)
            return

        for module in (
            *resnet_forward_modules,
            self.image_pool,
            self.image_projection,
            self.status_encoder,
            self.gamma_head,
            self.beta_head,
        ):
            _enable(module)

        if recons_key == "recons_fusion_z":
            return

        _enable(self.planning_layer1)
        if recons_key == "recons_planning_z1":
            return

        _enable(self.planning_layer2)
        if recons_key == "recons_planning_z2":
            return

        raise ValueError(
            f"set_recons_param has no parameter policy for recons_key={recons_key}"
        )
    
    def get_fea_name(self):
        return list(self.recons_map.keys())
    

    
