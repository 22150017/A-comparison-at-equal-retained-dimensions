





from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class SEAttention(nn.Module):
    




    def __init__(self, channels: int) -> None:
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        weight = self.fc(self.pool(x))
        return x * weight


class EMAResNet34(nn.Module):
    
    def __init__(self, num_classes: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        
        self.se = SEAttention(128)
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


def build_backbone(name: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    
    if name == "resnet34_ema":
        return EMAResNet34(num_classes=num_classes, pretrained=pretrained)

    if name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )
        return model

    if name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {name}")
