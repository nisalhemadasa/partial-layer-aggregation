from typing import List

import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for NCHW tensors.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================
# 2. ConvNeXt blocks
# ============================================================

class ConvNeXtBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)

        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma[:, None, None] * x

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtTinyImageNet(nn.Module):
    """
    ConvNeXt-like model adapted for Tiny ImageNet (64x64).

    This is a moderate-size version, much better than the tiny one you had.
    """
    def __init__(
            self,
            num_classes: int = 200,
            dims: List[int] = [64, 128, 256, 512],
            depths: List[int] = [3, 3, 9, 3],
            drop_path_rate: float = 0.1,
            layer_scale_init_value: float = 1e-6,
            _model_id: int = None,
    ):
        super().__init__()
        self.model_id = _model_id

        # stem: 64x64 -> 16x16
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # stochastic depth schedule
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        block_idx = 0
        for i in range(len(depths)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                block_idx += 1
            self.stages.append(nn.Sequential(*blocks))

            if i < len(depths) - 1:
                self.downsample_layers.append(
                    nn.Sequential(
                        LayerNorm2d(dims[i], eps=1e-6),
                        nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    )
                )

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
                # LayerNorm2d contains LayerNorm internally, so this mainly affects plain LayerNorm
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)

        # global average pool
        x = x.mean(dim=[2, 3])   # [B, C]
        x = self.norm(x)
        x = self.head(x)
        return x

    def get_model_type(self) -> str:
        return "ConvNeXtTinyImageNet"
