import torch
import torch.nn as nn
from torchvision import models

'''
这里vision的逻辑是入口是build vision encoder 然后会根据encoder_mode选择film还是concat 在film和concat类里面再去实例化vision encoder以及和proprio的方式
'''

class Resnet18_GAP(nn.Module):
    """Torchvision ResNet18 trunk followed by global average pooling."""

    def __init__(
        self,
        normalize_image=True,
        pretrained=False,
        input_channels=3,
    ):
        super().__init__()
        self.normalize_image = normalize_image
        self.output_dim = 512

        resnet = models.resnet18(pretrained=pretrained)
        if input_channels != 3:
            resnet.conv1 = nn.Conv2d(
                input_channels,
                resnet.conv1.out_channels,
                kernel_size=resnet.conv1.kernel_size,
                stride=resnet.conv1.stride,
                padding=resnet.conv1.padding,
                bias=False,
            )

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

    def _prepare_image(self, image):
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()

        if self.normalize_image:
            image = (image - self.image_mean) / self.image_std
        return image

    def forward(self, image):
        image = self._prepare_image(image)
        feature = self.feature_extractor(image)
        feature = self.pool(feature)
        return feature.flatten(start_dim=1)


class ImageEncoder(nn.Module):
    """Small factory wrapper for selecting vision encoders by name."""

    def __init__(
        self,
        encoder_name="resnet18_gap",
        normalize_image=True,
        pretrained=False,
        input_channels=3,
    ):
        super().__init__()
        self.encoder_name = encoder_name.lower()

        if self.encoder_name == "resnet18_gap":
            self.encoder = Resnet18_GAP(
                normalize_image=normalize_image,
                pretrained=pretrained,
                input_channels=input_channels,
            )
        else:
            raise ValueError(
                "Unsupported image encoder. Choose one of: "
                "'resnet18_gap', ."
            )

        self.output_dim = self.encoder.output_dim

    def forward(self, image):
        return self.encoder(image)


class ConcatObsEncoder(nn.Module):
    """Encode image and concatenate the visual feature with proprioception."""

    def __init__(
        self,
        proprio_dim,
        image_key="image",
        proprio_key="proprio",
        encoder_name="resnet18_gap",
        normalize_image=True,
        pretrained=False,
        input_channels=3,
    ):
        super().__init__()
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.proprio_dim = proprio_dim
        self.image_encoder = ImageEncoder(
            encoder_name=encoder_name,
            normalize_image=normalize_image,
            pretrained=pretrained,
            input_channels=input_channels,
        )
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


class FiLMObsEncoder(nn.Module):
    """Use proprioception to FiLM-modulate ResNet18_GAP image features."""

    def __init__(
        self,
        proprio_dim,
        image_key="image",
        proprio_key="proprio",
        encoder_name="resnet18_gap",
        normalize_image=True,
        pretrained=False,
        input_channels=3,
        film_hidden_dim=None,
        concat_proprio=False,
    ):
        super().__init__()
        self.image_key = image_key
        self.proprio_key = proprio_key
        self.proprio_dim = proprio_dim
        self.concat_proprio = concat_proprio
        self.image_encoder = ImageEncoder(
            encoder_name=encoder_name,
            normalize_image=normalize_image,
            pretrained=pretrained,
            input_channels=input_channels,
        )
        self.image_feature_dim = self.image_encoder.output_dim
        film_hidden_dim = film_hidden_dim or self.image_feature_dim

        self.gamma = nn.Sequential(
            nn.Linear(proprio_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, self.image_feature_dim),
        )
        self.beta = nn.Sequential(
            nn.Linear(proprio_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, self.image_feature_dim),
        )
        self.feature_dim = self.image_feature_dim + proprio_dim if concat_proprio else self.image_feature_dim

    def output_shape(self):
        return (self.feature_dim,)

    @property
    def output_dim(self):
        return self.feature_dim

    def forward(self, obs):
        proprio = obs[self.proprio_key].float()
        image_feature = self.image_encoder(obs[self.image_key])
        film_feature = (self.gamma(proprio) + 1.0) * image_feature + self.beta(proprio)
        if self.concat_proprio:
            return torch.cat([film_feature, proprio], dim=1)
        return film_feature


def build_vision_encoder(
    encoder_mode="film",
    proprio_dim=None,
    image_key="image",
    proprio_key="proprio",
    encoder_name="resnet18_gap",
    normalize_image=True,
    pretrained=False,
    input_channels=3,
    film_hidden_dim=None,
    concat_proprio=False,
):
    if encoder_mode in ("image", "image_only"):
        return ImageEncoder(
            encoder_name=encoder_name,
            normalize_image=normalize_image,
            pretrained=pretrained,
            input_channels=input_channels,
        )

    if proprio_dim is None:
        raise ValueError("proprio_dim is required for concat and film vision encoders.")

    if encoder_mode == "concat":
        return ConcatObsEncoder(
            proprio_dim=proprio_dim,
            image_key=image_key,
            proprio_key=proprio_key,
            encoder_name=encoder_name,
            normalize_image=normalize_image,
            pretrained=pretrained,
            input_channels=input_channels,
        )

    if encoder_mode == "film":
        return FiLMObsEncoder(
            proprio_dim=proprio_dim,
            image_key=image_key,
            proprio_key=proprio_key,
            encoder_name=encoder_name,
            normalize_image=normalize_image,
            pretrained=pretrained,
            input_channels=input_channels,
            film_hidden_dim=film_hidden_dim,
            concat_proprio=concat_proprio,
        )

    raise ValueError("encoder_mode must be 'image', 'concat', or 'film'.")
