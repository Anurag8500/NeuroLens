import torch
import torch.nn as nn
import timm


class SimpleCNN(nn.Module):
    """4-layer convolutional neural network baseline trained from scratch."""

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def create_model(model_name: str, num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Model factory supporting SimpleCNN, ResNet-50, and EfficientNet-B0."""
    name_lower = model_name.lower()
    
    if name_lower == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif name_lower in ["resnet50", "resnet-50"]:
        return timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
    elif name_lower in ["efficientnet_b0", "efficientnet-b0"]:
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    else:
        # Fallback to general timm models
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
