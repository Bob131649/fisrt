from collections import OrderedDict

import torch
import torch.nn as nn
from robomimic.models.obs_nets import ObservationGroupEncoder
import robomimic.utils.obs_utils as ObsUtils


def _resolve_class(class_name, import_candidates):
    if class_name is None or not isinstance(class_name, str):
        return class_name
    for module_name in import_candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Could not find robomimic class '{class_name}'.")


class RobomimicObsEncoder(nn.Module):
    """Diffusion-Policy-style robomimic observation encoder wrapper.

    The public forward API accepts a flat obs dict:
        {"image": BxCxHxW, "proprio": BxD}

    Internally robomimic's ObservationGroupEncoder receives:
        {"obs": {"image": ..., "proprio": ...}}
    """

    def __init__(
        self,
        image_shape,
        proprio_dim,
        image_key="image",
        proprio_key="proprio",
        feature_dim=256,
        crop_shape=None,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
    ):
        super().__init__()
        visual_core_class = _resolve_class(
            "VisualCore",
            ["robomimic.models.obs_core", "robomimic.models.base_nets"],
        )
        ObsUtils.OBS_ENCODER_CORES.setdefault("VisualCore", visual_core_class)

        self.image_key = image_key
        self.proprio_key = proprio_key
        self.feature_dim = feature_dim

        ObsUtils.initialize_obs_utils_with_obs_specs(
            {
                "obs": {
                    "rgb": [image_key],
                    "low_dim": [proprio_key],
                    "depth": [],
                    "scan": [],
                }
            }
        )

        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict()
        observation_group_shapes["obs"][image_key] = tuple(image_shape)
        observation_group_shapes["obs"][proprio_key] = (proprio_dim,)

        image_encoder_kwargs = {
            "core_class": "VisualCore",
            "core_kwargs": {
                "feature_dimension": feature_dim,
                "backbone_class": backbone_class,
                "backbone_kwargs": {
                    "pretrained": False,
                    "input_coord_conv": False,
                },
                "pool_class": pool_class,
                "pool_kwargs": {
                    "num_kp": 32,
                    "learnable_temperature": False,
                    "temperature": 1.0,
                    "noise_std": 0.0,
                },
                "flatten": True,
            },
            "obs_randomizer_class": None,
            "obs_randomizer_kwargs": {},
        }
        if crop_shape is not None:
            image_encoder_kwargs["obs_randomizer_class"] = _resolve_class(
                "CropRandomizer",
                ["robomimic.models.obs_core", "robomimic.models.obs_nets"],
            )
            image_encoder_kwargs["obs_randomizer_kwargs"] = {
                "crop_height": crop_shape[0],
                "crop_width": crop_shape[1],
                "num_crops": 1,
                "pos_enc": False,
            }

        encoder_kwargs = {
            "rgb": image_encoder_kwargs,
            "low_dim": {
                "core_class": None,
                "core_kwargs": {},
                "obs_randomizer_class": None,
                "obs_randomizer_kwargs": {},
            },
        }

        self.encoder = ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=nn.ReLU,
        )

    def output_shape(self):
        if hasattr(self.encoder, "output_shape"):
            shape = self.encoder.output_shape()
            if isinstance(shape, dict):
                shape = shape["obs"]
            return tuple(shape)
        return (self.feature_dim,)

    @property
    def output_dim(self):
        shape = self.output_shape()
        dim = 1
        for value in shape:
            dim *= value
        return dim

    def forward(self, obs):
        image = obs[self.image_key]
        proprio = obs[self.proprio_key]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return self.encoder(
            obs={
                self.image_key: image.float(),
                self.proprio_key: proprio.float(),
            }
        )
