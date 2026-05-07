"""Fake symmetric quantization for low-rank residual tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from compression.low_rank import LowRankKVSnapshot
from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import QuantizationConfig

ScaleMode = Literal["per_tensor", "per_channel", "group"]


@dataclass(frozen=True)
class QuantizationMetrics:
    mse: float
    max_abs_error: float
    logical_nbytes: int
    stored_nbytes: int


@dataclass(frozen=True)
class QuantizedTensor:
    """Integer q-values plus scales for fake dequantization."""

    qvalues: torch.Tensor
    scales: torch.Tensor
    num_bits: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    scale_mode: ScaleMode
    group_size: int | None = None
    padded_last_dim: int | None = None

    @property
    def qmax(self) -> int:
        return qmax_for_bits(self.num_bits)

    @property
    def stored_nbytes(self) -> int:
        return _tensor_nbytes(self.qvalues) + _tensor_nbytes(self.scales)

    @property
    def logical_nbytes(self) -> int:
        q_bytes = math.ceil(self.qvalues.numel() * self.num_bits / 8)
        return q_bytes + _tensor_nbytes(self.scales)

    def dequantize(self) -> torch.Tensor:
        qvalues = self.qvalues.to(torch.float32)
        if self.scale_mode == "group":
            if self.group_size is None or self.padded_last_dim is None:
                raise ValueError("Grouped dequantization requires group_size and padded_last_dim")
            leading_shape = tuple(self.qvalues.shape[:-1])
            grouped = qvalues.reshape(*leading_shape, self.padded_last_dim // self.group_size, self.group_size)
            dequantized = (grouped * self.scales).reshape(*leading_shape, self.padded_last_dim)
            dequantized = dequantized[..., : self.original_shape[-1]]
        else:
            dequantized = qvalues * self.scales
        return dequantized.reshape(self.original_shape).to(self.original_dtype)

    def metrics(self, target: torch.Tensor) -> QuantizationMetrics:
        dequantized = self.dequantize()
        error = target - dequantized
        return QuantizationMetrics(
            mse=float(torch.mean(error.square()).item()),
            max_abs_error=float(error.abs().max().item()),
            logical_nbytes=self.logical_nbytes,
            stored_nbytes=self.stored_nbytes,
        )


@dataclass(frozen=True)
class QuantizedKVLayerResidual:
    layer_idx: int
    key_residual: QuantizedTensor
    value_residual: QuantizedTensor


@dataclass(frozen=True)
class QuantizedKVResidualSnapshot:
    layers: tuple[QuantizedKVLayerResidual, ...]

    def dequantize(self) -> KVCacheSnapshot:
        return KVCacheSnapshot(
            layers=tuple(
                KVLayerCache(
                    layer_idx=layer.layer_idx,
                    key=layer.key_residual.dequantize(),
                    value=layer.value_residual.dequantize(),
                )
                for layer in self.layers
            ),
            metadata={"source": "quantized_residual_dequantization"},
        )


def qmax_for_bits(num_bits: int) -> int:
    if num_bits not in {2, 4, 8}:
        raise ValueError("num_bits must be one of {2, 4, 8}")
    return (2 ** (num_bits - 1)) - 1


def fake_quantize(tensor: torch.Tensor, config: QuantizationConfig) -> QuantizedTensor:
    """Symmetrically quantize and store q-values plus scales."""

    if not config.symmetric:
        raise NotImplementedError("Only symmetric quantization is implemented initially")
    qmax = qmax_for_bits(config.quant_bits)
    tensor_fp32 = tensor.detach().to(torch.float32)

    if config.per_channel:
        qvalues, scales = _per_channel_quantize(tensor_fp32, qmax)
        return QuantizedTensor(
            qvalues=qvalues,
            scales=scales,
            num_bits=config.quant_bits,
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            scale_mode="per_channel",
        )

    if 0 < config.group_size < tensor.shape[-1]:
        qvalues, scales, padded_last_dim = _group_quantize(tensor_fp32, qmax, config.group_size)
        return QuantizedTensor(
            qvalues=qvalues,
            scales=scales,
            num_bits=config.quant_bits,
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            scale_mode="group",
            group_size=config.group_size,
            padded_last_dim=padded_last_dim,
        )

    qvalues, scales = _per_tensor_quantize(tensor_fp32, qmax)
    return QuantizedTensor(
        qvalues=qvalues,
        scales=scales,
        num_bits=config.quant_bits,
        original_shape=tuple(tensor.shape),
        original_dtype=tensor.dtype,
        scale_mode="per_tensor",
    )


def quantize_residual(
    original: torch.Tensor,
    low_rank_reconstruction: torch.Tensor,
    config: QuantizationConfig,
) -> QuantizedTensor:
    """Quantize residual ``R = original - low_rank_reconstruction``."""

    return fake_quantize(original - low_rank_reconstruction, config)


def quantize_kv_residuals(
    original: KVCacheSnapshot,
    low_rank: LowRankKVSnapshot,
    config: QuantizationConfig,
) -> QuantizedKVResidualSnapshot:
    """Quantize K and V residuals for a full KV snapshot."""

    low_rank_reconstruction = low_rank.reconstruct()
    if original.num_layers != low_rank_reconstruction.num_layers:
        raise ValueError("original and low_rank snapshots must have the same number of layers")

    layers: list[QuantizedKVLayerResidual] = []
    for original_layer, low_rank_layer in zip(original.layers, low_rank_reconstruction.layers, strict=True):
        layers.append(
            QuantizedKVLayerResidual(
                layer_idx=original_layer.layer_idx,
                key_residual=quantize_residual(original_layer.key, low_rank_layer.key, config),
                value_residual=quantize_residual(original_layer.value, low_rank_layer.value, config),
            )
        )
    return QuantizedKVResidualSnapshot(layers=tuple(layers))


def _per_tensor_quantize(tensor: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = tensor.abs().max()
    scale = _safe_scale(max_abs, qmax)
    qvalues = _round_clamp(tensor / scale, qmax)
    return qvalues, scale.reshape(())


def _per_channel_quantize(tensor: torch.Tensor, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(tensor.ndim - 1))
    max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=True)
    scales = _safe_scale(max_abs, qmax)
    qvalues = _round_clamp(tensor / scales, qmax)
    return qvalues, scales


def _group_quantize(
    tensor: torch.Tensor,
    qmax: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    last_dim = tensor.shape[-1]
    padded_last_dim = int(math.ceil(last_dim / group_size) * group_size)
    if padded_last_dim != last_dim:
        tensor = torch.nn.functional.pad(tensor, (0, padded_last_dim - last_dim))
    leading_shape = tuple(tensor.shape[:-1])
    grouped = tensor.reshape(*leading_shape, padded_last_dim // group_size, group_size)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scales = _safe_scale(max_abs, qmax)
    qvalues = _round_clamp(grouped / scales, qmax).reshape(*leading_shape, padded_last_dim)
    return qvalues, scales, padded_last_dim


def _safe_scale(max_abs: torch.Tensor, qmax: int) -> torch.Tensor:
    scale = max_abs / float(qmax)
    return torch.where(max_abs > 0, scale, torch.ones_like(scale))


def _round_clamp(values: torch.Tensor, qmax: int) -> torch.Tensor:
    return torch.clamp(torch.round(values), min=-qmax, max=qmax).to(torch.int8)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
