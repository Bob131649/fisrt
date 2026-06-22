from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models
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


class RobomimicImageEncoder(nn.Module):
    """Robomimic visual encoder that only consumes RGB observations."""

    def __init__(
        self,
        image_shape,
        image_key="image",
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
        self.feature_dim = feature_dim

        ObsUtils.initialize_obs_utils_with_obs_specs(
            {
                "obs": {
                    "rgb": [image_key],
                    "low_dim": [],
                    "depth": [],
                    "scan": [],
                }
            }
        )

        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict()
        observation_group_shapes["obs"][image_key] = tuple(image_shape)

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

    def forward(self, image):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        feature = self.encoder(obs={self.image_key: image.float()})
        if isinstance(feature, dict):
            if "obs" in feature:
                feature = feature["obs"]
            elif "feature" in feature:
                feature = feature["feature"]
            else:
                raise KeyError("image encoder returned a dict without 'obs' or 'feature'.")
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        return feature


class ZipperImageEncoder(nn.Module):
    """Franka-generative-style ResNet image encoder with global average pooling."""

    def __init__(
        self,
        backbone="resnet18",
        normalize_image=True,
    ):
        super().__init__()
        self.normalize_image = normalize_image
        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=False)
            self.output_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=False)
            self.output_dim = 2048
        else:
            raise ValueError("zipper image encoder supports 'resnet18' and 'resnet50'.")

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()
        if self.normalize_image:
            image = (image - self.image_mean) / self.image_std
        feature = self.feature_extractor(image)
        feature = self.pool(feature)
        return feature.flatten(start_dim=1)


class ZipperObsEncoder(nn.Module):
    """Use a torchvision ResNet + GAP image feature concatenated with proprioception."""

    def __init__(
        self,
        proprio_dim,
        image_key="image",
        proprio_key="proprio",
        backbone="resnet18",
        normalize_image=True,
    ):
        super().__init__()
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.image_encoder = ZipperImageEncoder(
            backbone=backbone,
            normalize_image=normalize_image,
        )
        self.proprio_dim = proprio_dim
        self.feature_dim = self.image_encoder.output_dim + proprio_dim

    def output_shape(self):
        return (self.feature_dim,)

    @property
    def output_dim(self):
        return self.feature_dim

    def forward(self, obs):
        image_feature = self.image_encoder(obs[self.image_key])
        proprio = obs[self.proprio_key].float()
        return torch.cat([image_feature, proprio], dim=1)


def build_obs_encoder(
    encoder_mode,
    image_shape,
    proprio_dim,
    image_key="image",
    proprio_key="proprio",
    robomimic_feature_dim=256,
    robomimic_crop_shape=None,
    robomimic_backbone_class="ResNet18Conv",
    robomimic_pool_class="SpatialSoftmax",
    zipper_backbone="resnet18",
    zipper_normalize_image=True,
):
    if encoder_mode in ("default", "robomimic"):
        # print("Using RobomimicObsEncoder with learnable spatial softmax pooling:")
        return RobomimicObsEncoder(
            image_shape=image_shape,
            proprio_dim=proprio_dim,
            image_key=image_key,
            proprio_key=proprio_key,
            feature_dim=robomimic_feature_dim,
            crop_shape=robomimic_crop_shape,
            backbone_class=robomimic_backbone_class,
            pool_class=robomimic_pool_class,
        )
    if encoder_mode == "zipper":
        # print("Using ZipperObsEncoder with average pooling:")
        return ZipperObsEncoder(
            proprio_dim=proprio_dim,
            image_key=image_key,
            proprio_key=proprio_key,
            backbone=zipper_backbone,
            normalize_image=zipper_normalize_image,
        )
    raise ValueError("encoder_mode must be 'default', 'robomimic', or 'zipper'.")


class FiLMObsEncoder(nn.Module):
    """Use proprioception to FiLM-modulate visual features."""

    def __init__(
        self,
        image_shape,
        proprio_dim,
        encoder_mode="default",
        image_key="image",
        proprio_key="proprio",
        feature_dim=256,
        film_hidden_dim=None,
        crop_shape=None,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
        zipper_backbone="resnet18",
        zipper_normalize_image=True,
    ):
        super().__init__()
        self.image_key = image_key
        self.proprio_key = proprio_key
        if encoder_mode == "zipper":
            self.image_encoder = ZipperImageEncoder(
                backbone=zipper_backbone,
                normalize_image=zipper_normalize_image,
            )
        elif encoder_mode in ("default", "robomimic"):
            self.image_encoder = RobomimicImageEncoder(
                image_shape=image_shape,
                image_key=image_key,
                feature_dim=feature_dim,
                crop_shape=crop_shape,
                backbone_class=backbone_class,
                pool_class=pool_class,
            )
        else:
            raise ValueError("encoder_mode must be 'default', 'robomimic', or 'zipper'.")
        self.feature_dim = self.image_encoder.output_dim
        self.proprio_dim = proprio_dim
        film_hidden_dim = film_hidden_dim or self.feature_dim

        self.gamma = nn.Sequential(
            nn.Linear(proprio_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, self.feature_dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(proprio_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, self.feature_dim),
        )

    def output_shape(self):
        return (self.feature_dim,)
        # return (self.feature_dim + self.proprio_dim,)

    @property
    def output_dim(self):
        return self.feature_dim
        # return self.feature_dim + self.proprio_dim

    def forward(self, obs):
        image = obs[self.image_key]
        proprio = obs[self.proprio_key].float()
        image_feature = self.image_encoder(image)
        gamma = self.gamma(proprio) + 1.0
        beta = self.beta(proprio)
        return gamma * image_feature + beta
        # film_feature = gamma * image_feature + beta
        # return torch.cat([film_feature, proprio],dim=1)
