"""Compact native-PyTorch networks for learned seed-instance segmentation."""

from __future__ import annotations

from dataclasses import dataclass


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _torch_modules():
    import torch
    import torch.nn as nn

    return torch, nn


class ResidualBlock:
    """Factory wrapper kept import-safe for launcher diagnostics."""

    @staticmethod
    def create(input_channels: int, output_channels: int):
        _torch, nn = _torch_modules()

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.main = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(output_channels), output_channels),
                )
                self.skip = (
                    nn.Identity()
                    if input_channels == output_channels
                    else nn.Conv2d(input_channels, output_channels, 1, bias=False)
                )
                self.activation = nn.SiLU(inplace=True)

            def forward(self, values):
                return self.activation(self.main(values) + self.skip(values))

        return _Block()


class UNetBackbone:
    """Small residual U-Net whose output retains the input raster dimensions."""

    @staticmethod
    def create(
        input_channels: int,
        *,
        base_channels: int = 24,
        depth: int = 4,
    ):
        torch, nn = _torch_modules()
        if not 1 <= depth <= 5:
            raise ValueError("U-Net depth must be between 1 and 5.")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8.")

        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                widths = [base_channels * (2**index) for index in range(depth + 1)]
                encoders = [ResidualBlock.create(input_channels, widths[0])]
                downsamples = []
                for index in range(depth):
                    downsamples.append(
                        nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1)
                    )
                    encoders.append(
                        ResidualBlock.create(widths[index + 1], widths[index + 1])
                    )
                self.encoders = nn.ModuleList(encoders)
                self.downsamples = nn.ModuleList(downsamples)
                self.decoders = nn.ModuleList(
                    ResidualBlock.create(widths[index + 1] + widths[index], widths[index])
                    for index in range(depth - 1, -1, -1)
                )
                self.widths = tuple(widths)
                self.output_channels = widths[0]

            def forward(self, values):
                import torch.nn.functional as functional

                skips = []
                current = values
                for index, encoder in enumerate(self.encoders):
                    current = encoder(current)
                    if index < len(self.downsamples):
                        skips.append(current)
                        current = self.downsamples[index](current)
                for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
                    current = functional.interpolate(
                        current, size=skip.shape[-2:], mode="bilinear", align_corners=False
                    )
                    current = decoder(torch.cat((current, skip), dim=1))
                return current

        return _Backbone()


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    input_channels: int
    base_channels: int = 24
    depth: int = 4
    ray_count: int = 32


def _head(input_channels: int, output_channels: int):
    _torch, nn = _torch_modules()
    hidden = max(8, input_channels // 2)
    return nn.Sequential(
        nn.Conv2d(input_channels, hidden, 3, padding=1),
        nn.SiLU(inplace=True),
        nn.Conv2d(hidden, output_channels, 1),
    )


def MultiHeadSeedUNet(
    input_channels: int,
    *,
    base_channels: int = 24,
    depth: int = 4,
):
    """Build the five-logical-head U-Net requested by the project."""

    _torch, nn = _torch_modules()

    class _Model(nn.Module):
        family = "unet_watershed"

        def __init__(self) -> None:
            super().__init__()
            self.configuration = ModelConfiguration(
                input_channels=input_channels,
                base_channels=base_channels,
                depth=depth,
            )
            self.backbone = UNetBackbone.create(
                input_channels, base_channels=base_channels, depth=depth
            )
            channels = self.backbone.output_channels
            self.interior_head = _head(channels, 1)
            self.physical_boundary_head = _head(channels, 1)
            self.pattern_boundary_head = _head(channels, 1)
            self.centre_distance_head = _head(channels, 2)
            self.uncertainty_head = _head(channels, 5)

        def forward(self, values):
            features = self.backbone(values)
            centre_distance = self.centre_distance_head(features)
            return {
                "interior_logits": self.interior_head(features),
                "physical_boundary_logits": self.physical_boundary_head(features),
                "pattern_boundary_logits": self.pattern_boundary_head(features),
                "centre_logits": centre_distance[:, 0:1],
                "distance": centre_distance[:, 1:2].sigmoid(),
                "uncertainty_logits": self.uncertainty_head(features),
            }

    return _Model()


def SeedStarDist2D(
    input_channels: int,
    *,
    ray_count: int = 32,
    base_channels: int = 24,
    depth: int = 4,
):
    """Build a dense 2-D StarDist network using the same PyTorch backbone."""

    if not 8 <= int(ray_count) <= 256:
        raise ValueError("ray_count must be between 8 and 256.")
    _torch, nn = _torch_modules()

    class _Model(nn.Module):
        family = "stardist"

        def __init__(self) -> None:
            super().__init__()
            self.configuration = ModelConfiguration(
                input_channels=input_channels,
                base_channels=base_channels,
                depth=depth,
                ray_count=int(ray_count),
            )
            self.backbone = UNetBackbone.create(
                input_channels, base_channels=base_channels, depth=depth
            )
            channels = self.backbone.output_channels
            self.object_head = _head(channels, 1)
            self.radial_head = _head(channels, int(ray_count))
            self.uncertainty_head = _head(channels, 1)

        def forward(self, values):
            import torch.nn.functional as functional

            features = self.backbone(values)
            return {
                "object_logits": self.object_head(features),
                "radial_distances": functional.softplus(self.radial_head(features)) + 0.25,
                "radial_uncertainty_logits": self.uncertainty_head(features),
            }

    return _Model()
