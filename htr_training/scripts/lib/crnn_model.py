from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    width_reduction_factor = 4

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=(4, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.projection = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)

        if features.shape[2] != 1:
            raise RuntimeError(
                "CNN output height must be 1, but received "
                f"{features.shape[2]}. Expected input height is 64."
            )

        features = features.squeeze(2)
        features = features.permute(0, 2, 1)

        recurrent_features, _ = self.lstm(features)
        logits = self.projection(recurrent_features)

        return logits.permute(1, 0, 2)

    def calculate_input_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        return torch.div(
            image_widths,
            self.width_reduction_factor,
            rounding_mode="floor",
        ).clamp(min=1)
