"""Standalone neural network used by TIDY inference.

The module intentionally depends only on PyTorch. Its learned module names match
the released TIDY checkpoint so that existing weights remain loadable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChannelLayerNorm2d(nn.Module):
    """Layer normalization over channels at every spatial location."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, image: Tensor) -> Tensor:
        if torch.onnx.is_in_onnx_export():
            # NHWC makes channels the normalized trailing dimension. Exporting
            # the standard operation lets TensorRT retain FP32 accumulation.
            channels_last = image.permute(0, 2, 3, 1)
            normalized = F.layer_norm(
                channels_last,
                (self.weight.numel(),),
                self.weight,
                self.bias,
                self.eps,
            )
            return normalized.permute(0, 3, 1, 2)

        # Preserve the arithmetic and output of the original PyTorch inference
        # implementation exactly outside ONNX export.
        mean = image.mean(dim=1, keepdim=True)
        variance = (image - mean).square().mean(dim=1, keepdim=True)
        normalized = (image - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half and multiply the two halves."""

    def forward(self, features: Tensor) -> Tensor:
        first, second = features.chunk(2, dim=1)
        return first * second


class TIDYBlock(nn.Module):
    """Efficient gated residual block used throughout TIDYNet."""

    def __init__(
        self,
        channels: int,
        depthwise_expansion: int = 2,
        feedforward_expansion: int = 2,
    ) -> None:
        super().__init__()
        depthwise_channels = channels * depthwise_expansion
        feedforward_channels = channels * feedforward_expansion

        # Attribute names are checkpoint compatibility keys.
        self.conv1 = nn.Conv2d(channels, depthwise_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            kernel_size=3,
            padding=1,
            groups=depthwise_channels,
        )
        self.conv3 = nn.Conv2d(depthwise_channels // 2, channels, kernel_size=1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                depthwise_channels // 2,
                depthwise_channels // 2,
                kernel_size=1,
            ),
        )
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(channels, feedforward_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(feedforward_channels // 2, channels, kernel_size=1)
        self.norm1 = ChannelLayerNorm2d(channels)
        self.norm2 = ChannelLayerNorm2d(channels)
        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features: Tensor) -> Tensor:
        residual = self.norm1(features)
        residual = self.conv1(residual)
        residual = self.conv2(residual)
        residual = self.sg(residual)
        residual = residual * self.sca(residual)
        residual = self.dropout1(self.conv3(residual))
        features = features + residual * self.beta

        residual = self.conv4(self.norm2(features))
        residual = self.sg(residual)
        residual = self.dropout2(self.conv5(residual))
        return features + residual * self.gamma


class TIDYNet(nn.Module):
    """Wavelet-domain thermal image denoising network.

    The Haar transform is implemented directly with tensor operations. This
    removes the legacy ``pytorch_wavelets`` and SciPy dependency while retaining
    its coefficient ordering: low-low, low-high, high-low, high-high.
    """

    def __init__(
        self,
        image_channels: int = 3,
        width: int = 64,
        encoder_blocks: Sequence[int] = (2, 2, 4, 8),
        middle_blocks: int = 12,
        decoder_blocks: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(encoder_blocks) != len(decoder_blocks):
            raise ValueError("Encoder and decoder must have the same number of stages.")
        if image_channels <= 0 or width <= 0:
            raise ValueError("image_channels and width must be positive.")

        self.image_channels = image_channels
        self.cond_proj = nn.Sequential(
            nn.Conv2d(image_channels, width, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width * 2),
        )
        self.intro = nn.Conv2d(image_channels * 4, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, image_channels * 4, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width
        for block_count in encoder_blocks:
            self.encoders.append(
                nn.Sequential(*(TIDYBlock(channels) for _ in range(block_count)))
            )
            self.downs.append(
                nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)
            )
            channels *= 2

        self.middle_blks = nn.Sequential(
            *(TIDYBlock(channels) for _ in range(middle_blocks))
        )

        for block_count in decoder_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*(TIDYBlock(channels) for _ in range(block_count)))
            )

        # One extra factor of two accounts for the initial Haar decomposition.
        self.required_multiple = 2 ** (len(self.encoders) + 1)

    @staticmethod
    def _analysis_filter_bank(image: Tensor, vertical: bool) -> Tensor:
        """Apply interleaved low/high Haar filters with stride two."""
        channels = image.shape[1]
        scale = image.new_tensor(2.0**-0.5)
        shape = (1, 1, 2, 1) if vertical else (1, 1, 1, 2)
        low_filter = torch.stack((scale, scale)).reshape(shape)
        high_filter = torch.stack((scale, -scale)).reshape(shape)
        filters = torch.cat([low_filter, high_filter] * channels, dim=0)
        stride = (2, 1) if vertical else (1, 2)
        return F.conv2d(image, filters, stride=stride, groups=channels)

    @classmethod
    def _haar_decompose(cls, image: Tensor) -> tuple[Tensor, Tensor]:
        """Return low-pass and three high-pass Haar subbands."""
        coefficients = cls._analysis_filter_bank(image, vertical=False)
        coefficients = cls._analysis_filter_bank(coefficients, vertical=True)
        batch, _, height, width = coefficients.shape
        coefficients = coefficients.reshape(batch, -1, 4, height, width)
        return coefficients[:, :, 0].contiguous(), coefficients[:, :, 1:].contiguous()

    @staticmethod
    def _haar_reconstruct(
        low: Tensor, high: Tensor, channels: int | None = None
    ) -> Tensor:
        """Invert :meth:`_haar_decompose`."""
        low_high, high_low, high_high = high.unbind(dim=2)
        if channels is None:
            channels = low.shape[1]
        scale = low.new_tensor(2.0**-0.5)
        vertical_shape = (1, 1, 2, 1)
        low_vertical = (
            torch.stack((scale, scale))
            .reshape(vertical_shape)
            .repeat(channels, 1, 1, 1)
        )
        high_vertical = (
            torch.stack((scale, -scale))
            .reshape(vertical_shape)
            .repeat(channels, 1, 1, 1)
        )
        low_rows = F.conv_transpose2d(
            low, low_vertical, stride=(2, 1), groups=channels
        ) + F.conv_transpose2d(low_high, high_vertical, stride=(2, 1), groups=channels)
        high_rows = F.conv_transpose2d(
            high_low, low_vertical, stride=(2, 1), groups=channels
        ) + F.conv_transpose2d(high_high, high_vertical, stride=(2, 1), groups=channels)

        horizontal_shape = (1, 1, 1, 2)
        low_horizontal = (
            torch.stack((scale, scale))
            .reshape(horizontal_shape)
            .repeat(channels, 1, 1, 1)
        )
        high_horizontal = (
            torch.stack((scale, -scale))
            .reshape(horizontal_shape)
            .repeat(channels, 1, 1, 1)
        )
        return F.conv_transpose2d(
            low_rows, low_horizontal, stride=(1, 2), groups=channels
        ) + F.conv_transpose2d(
            high_rows, high_horizontal, stride=(1, 2), groups=channels
        )

    def _pad(self, image: Tensor) -> Tensor:
        height, width = image.shape[-2:]
        pad_height = (-height) % self.required_multiple
        pad_width = (-width) % self.required_multiple
        return F.pad(image, (0, pad_width, 0, pad_height))

    def forward(self, image: Tensor) -> Tensor:
        if not torch.onnx.is_in_onnx_export() and (
            image.ndim != 4 or image.shape[1] != self.image_channels
        ):
            raise ValueError(
                f"Expected Bx{self.image_channels}xHxW input, got {tuple(image.shape)}."
            )

        batch, _, original_height, original_width = image.shape
        channels = self.image_channels
        condition = self.cond_proj(image)
        film_scale, film_bias = condition.chunk(2, dim=1)
        film_scale = film_scale.view(batch, -1, 1, 1)
        film_bias = film_bias.view(batch, -1, 1, 1)

        padded = self._pad(image)
        low, high = self._haar_decompose(padded)
        features = torch.cat(
            (low, high.reshape(batch, channels * 3, low.shape[2], low.shape[3])),
            dim=1,
        )
        features = self.intro(features)
        features = film_scale * features + film_bias

        skip_features: list[Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downs):
            features = encoder(features)
            skip_features.append(features)
            features = downsample(features)

        features = self.middle_blks(features)

        for decoder, upsample, skip in zip(
            self.decoders, self.ups, reversed(skip_features)
        ):
            features = upsample(features) + skip
            features = decoder(features)

        coefficients = self.ending(features)
        low = coefficients[:, :channels]
        high = coefficients[:, channels:].reshape(
            batch, channels, 3, low.shape[2], low.shape[3]
        )
        output = self._haar_reconstruct(low, high, channels=self.image_channels)
        return output[:, :, :original_height, :original_width]
