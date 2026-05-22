from collections import OrderedDict
from typing import Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models
from torchvision.models import (
    EfficientNet_B2_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Small_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    VGG16_Weights,
)

from utils.singleton import singleton


def freeze_feature_extractor(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


class FrozenEncoder(nn.Module):
    def train(self, mode: bool = True):
        super().train(False)
        if hasattr(self, 'encoder'):
            self.encoder.eval()
        return self

    def encode(self, x):
        return self.forward(x)


def get_mae_encoder(name, in_channels=3, ckpt_path=None):
    model = get_encoder_by_name(name)
    print(f'Use MAE pretrained encoder {name}, in_channels={in_channels}')
    if ckpt_path is not None:
        state_dict = OrderedDict()
        pl_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        for key, value in pl_state_dict.items():
            if key.startswith('model.encoder'):
                state_dict[key.replace('model.encoder.', '')] = value
        model.load_state_dict(state_dict, strict=False)
    return model


def get_encoder(config: Union[str, DictConfig]):
    if isinstance(config, DictConfig):
        name = config.models.encoder_name
        if config.models.pretrain_ckpt is not None:
            in_channels = 1 if config.datasets.name == 'fashionmnist' else 3
            return get_mae_encoder(name, in_channels, config.models.pretrain_ckpt)
        return get_encoder_by_name(name)
    if isinstance(config, str):
        return get_encoder_by_name(config)
    raise TypeError(f'Unsupported encoder config type: {type(config)}')


def get_encoder_by_name(name: str):
    if name == 'identity':
        model = IdentityEncoder()
    elif name == 'resnet50':
        model = ResNet50Encoder()
    elif name == 'resnet34':
        model = ResNet34Encoder()
    elif name == 'vgg16':
        model = Vgg16Encoder()
    elif name == 'mobilenetv3s':
        model = Mobile3SmallEncoder()
    elif name == 'efficientv2s':
        model = Efficient2SmallEncoder()
    elif name == 'efficientb2':
        model = Efficientb2Encoder()
    elif name == 'vitb16':
        model = ViTb16Encoder()
    elif name == 'vitb32':
        model = ViTb32Encoder()
    else:
        raise ValueError(f'no such model: {name}')
    return model


@singleton
class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def encode(self, x):
        return self.forward(x)


@singleton
class ResNet50Encoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class ResNet34Encoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class Vgg16Encoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Sequential(
            self.encoder.classifier[0],
            self.encoder.classifier[1],
            self.encoder.classifier[3],
            self.encoder.classifier[4],
        )
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class Mobile3SmallEncoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Sequential(
            self.encoder.classifier[0],
            self.encoder.classifier[1],
        )
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class Efficient2SmallEncoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Identity()
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class Efficientb2Encoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.encoder.classifier = nn.Identity()
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class ViTb16Encoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.encoder.heads = nn.Identity()
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)


@singleton
class ViTb32Encoder(FrozenEncoder):
    def __init__(self):
        super().__init__()
        self.encoder = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        self.encoder.heads = nn.Identity()
        freeze_feature_extractor(self.encoder)

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)
