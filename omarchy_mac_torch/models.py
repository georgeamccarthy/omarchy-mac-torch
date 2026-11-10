"""Two small reference CNNs used by the benchmarks and checks (random data is enough: timing doesn't depend on pixels).

PlainCNN      [conv-BN-ReLU x2, max pool] x3, then two fully connected layers: 253,914 parameters.
SeparableCNN  depthwise-separable convolutions and global average pooling: 8,730 parameters.
Both take [N, 1, 28, 28] float32 images and return 10 class scores.
"""

import torch
from torch import nn


def _conv_bn_relu(ci, co, kernel=3, groups=1):
    return [nn.Conv2d(ci, co, kernel, padding=kernel // 2, groups=groups, bias=False), nn.BatchNorm2d(co), nn.ReLU()]


class PlainCNN(nn.Module):
    def __init__(self, widths=(16, 32, 64), hidden=(256, 128), num_classes=10):
        super().__init__()
        layers, ci, size = [], 1, 28
        for w in widths:
            layers += _conv_bn_relu(ci, w) + _conv_bn_relu(w, w) + [nn.MaxPool2d(2, 2)]
            ci, size = w, size // 2
        self.features = nn.Sequential(*layers)
        fc, n = [], ci * size * size
        for h in hidden:
            fc += [nn.Linear(n, h), nn.ReLU()]
            n = h
        self.classifier = nn.Sequential(*fc, nn.Linear(n, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(start_dim=1))


class SeparableCNN(nn.Module):
    def __init__(self, stem=16, widths=(32, 32, 48, 64), pool_after=(0, 2), num_classes=10):
        super().__init__()
        layers, c = _conv_bn_relu(1, stem), stem
        for i, w in enumerate(widths):
            layers += _conv_bn_relu(c, c, 3, groups=c) + _conv_bn_relu(c, w, 1)
            if i in pool_after:
                layers.append(nn.MaxPool2d(2, 2))
            c = w
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(start_dim=1))


MODELS = {"small": SeparableCNN, "large": PlainCNN}


def fake_data(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g)
