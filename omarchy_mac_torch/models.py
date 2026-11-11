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


# ---- acceptance models (tools/acceptance.py): standard architectures written out here so there are no extra
# dependencies. Small widths so a CPU reference run stays quick. ----

class _Basic(nn.Module):
    def __init__(self, ci, co, stride):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(co)
        self.down = (nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))
                     if stride != 1 or ci != co else nn.Identity())

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(self.b2(self.c2(y)) + self.down(x))


class ResNet18(nn.Module):
    """ResNet-18 (He et al.): residual adds and stride-2 convs mid-network (their input gradients run on the GPU)."""

    def __init__(self, num_classes=10, width=64):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 7, 2, 3, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        layers, ci = [], w[0]
        for i, co in enumerate(w):
            layers += [_Basic(ci, co, 1 if i == 0 else 2), _Basic(co, co, 1)]
            ci = co
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(ci, num_classes)

    def forward(self, x):
        return self.head(torch.flatten(nn.functional.adaptive_avg_pool2d(self.layers(self.stem(x)), 1), 1))


class _InvRes(nn.Module):
    def __init__(self, ci, co, expand, stride, k=3):
        super().__init__()
        h = ci * expand
        self.use_res = stride == 1 and ci == co
        self.block = nn.Sequential(
            nn.Conv2d(ci, h, 1, bias=False), nn.BatchNorm2d(h), nn.Hardswish(),
            nn.Conv2d(h, h, k, stride, k // 2, groups=h, bias=False), nn.BatchNorm2d(h), nn.Hardswish(),
            nn.Conv2d(h, co, 1, bias=False), nn.BatchNorm2d(co))

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


class MobileNetV3Small(nn.Module):
    """A MobileNetV3-Small-style network: inverted residuals, depthwise convs (some strided), hardswish."""

    def __init__(self, num_classes=10):
        super().__init__()
        cfg = [(16, 16, 1, 2), (16, 24, 4, 2), (24, 24, 3, 1), (24, 40, 4, 2), (40, 40, 6, 1), (40, 48, 3, 1), (48, 96, 6, 2)]
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.Hardswish())
        self.blocks = nn.Sequential(*[_InvRes(ci, co, e, s) for ci, co, e, s in cfg])
        self.head = nn.Sequential(nn.Conv2d(96, 576, 1, bias=False), nn.BatchNorm2d(576), nn.Hardswish(),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(576, 256), nn.Hardswish(),
                                  nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class ViT(nn.Module):
    """A small Vision Transformer: patch embedding conv, pre-norm attention blocks, GELU MLPs, class token."""

    def __init__(self, image=32, patch=4, dim=128, depth=4, heads=4, num_classes=10):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, patch)
        n = (image // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.head(self.norm(self.encoder(x))[:, 0])


class GPT(nn.Module):
    """A small GPT: token + position embeddings, causal self-attention blocks, weight-tied LM head."""

    def __init__(self, vocab=512, ctx=64, dim=128, depth=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1), persistent=False)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        x = self.blocks(x, mask=self.mask[:t, :t], is_causal=True)
        return self.norm(x) @ self.tok.weight.t()
