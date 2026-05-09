"""Low-rank decomposition primitives for KV cache tensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from models.kv_cache import KVCacheSnapshot, KVLayerCache
from utils.config import CompressionConfig

DecompositionType = Literal["truncated_svd", "randomized_svd", "pca"]


@dataclass(frozen=True)
class LowRankMetrics:
    """Reconstruction and spectral statistics for one decomposition."""

    rank: int
    relative_error: float
    mse: float
    spectral_energy: float
    original_bytes: int
    factor_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / self.factor_bytes if self.factor_bytes > 0 else math.inf


@dataclass(frozen=True)
class LowRankMatrixApproximation:
    """SVD factors for a single matrix-like tensor."""

    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    original_shape: tuple[int, ...]
    singular_values: torch.Tensor
    spectral_energy: float

    @property
    def rank(self) -> int:
        return int(self.s.numel())

    @property
    def factor_bytes(self) -> int:
        return _tensor_nbytes(self.u) + _tensor_nbytes(self.s) + _tensor_nbytes(self.vh)

    def reconstruct(self) -> torch.Tensor:
        matrix = (self.u * self.s.unsqueeze(0)) @ self.vh
        return matrix.reshape(self.original_shape)

    def metrics(self, target: torch.Tensor) -> LowRankMetrics:
        reconstruction = self.reconstruct()
        return _low_rank_metrics(target, reconstruction, self.rank, self.spectral_energy, self.factor_bytes)


@dataclass(frozen=True)
class LowRankTensorApproximation:
    """Low-rank approximation of a KV tensor.

    For ``per_head`` granularity, each factor corresponds to one attention head
    and approximates ``[batch, tokens, head_dim]`` flattened as ``[B*T, D]``.
    Otherwise a single factor approximates the full tensor flattened over all
    leading dimensions.
    """

    factors: tuple[LowRankMatrixApproximation, ...]
    original_shape: tuple[int, ...]
    granularity: str

    @property
    def max_rank(self) -> int:
        return max(factor.rank for factor in self.factors)

    @property
    def factor_bytes(self) -> int:
        return sum(factor.factor_bytes for factor in self.factors)

    def reconstruct(self) -> torch.Tensor:
        if self.granularity == "per_head":
            if len(self.original_shape) != 4:
                raise ValueError("per_head reconstruction requires original shape [B, H, T, D]")
            batch_size, num_heads, seq_len, head_dim = self.original_shape
            if len(self.factors) != num_heads:
                raise ValueError(f"Expected {num_heads} head factors, found {len(self.factors)}")
            heads = [
                factor.reconstruct().reshape(batch_size, seq_len, head_dim)
                for factor in self.factors
            ]
            return torch.stack(heads, dim=1)

        if len(self.factors) != 1:
            raise ValueError(f"{self.granularity} reconstruction expects one factor")
        return self.factors[0].reconstruct().reshape(self.original_shape)

    def relative_error(self, target: torch.Tensor) -> float:
        return float(_relative_fro_error(target, self.reconstruct()).item())

    def metrics(self, target: torch.Tensor) -> LowRankMetrics:
        reconstruction = self.reconstruct()
        average_energy = sum(factor.spectral_energy for factor in self.factors) / len(self.factors)
        return _low_rank_metrics(target, reconstruction, self.max_rank, average_energy, self.factor_bytes)


@dataclass(frozen=True)
class LowRankKVLayer:
    """Low-rank K and V approximations for one transformer layer."""

    layer_idx: int
    key: LowRankTensorApproximation
    value: LowRankTensorApproximation


@dataclass(frozen=True)
class LowRankKVSnapshot:
    """Low-rank approximations for all KV cache layers."""

    layers: tuple[LowRankKVLayer, ...]

    def reconstruct(self) -> KVCacheSnapshot:
        return KVCacheSnapshot(
            layers=tuple(
                KVLayerCache(
                    layer_idx=layer.layer_idx,
                    key=layer.key.reconstruct(),
                    value=layer.value.reconstruct(),
                )
                for layer in self.layers
            ),
            metadata={"source": "low_rank_reconstruction"},
        )


def select_rank_by_energy(singular_values: torch.Tensor, threshold: float) -> int:
    """Select the smallest rank whose squared singular-value energy reaches a threshold."""

    if singular_values.ndim != 1:
        raise ValueError("singular_values must be a vector")
    if singular_values.numel() == 0:
        raise ValueError("singular_values must be non-empty")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    energy = singular_values.square()
    total = energy.sum()
    if total <= 0:
        return 1
    cumulative = torch.cumsum(energy, dim=0) / total
    selected = int(torch.searchsorted(cumulative, torch.tensor(threshold, device=cumulative.device)).item() + 1)
    return min(selected, int(singular_values.numel()))


def compress_matrix(
    matrix: torch.Tensor,
    rank: int,
    decomposition_type: DecompositionType = "truncated_svd",
    adaptive_rank: bool = False,
    energy_threshold: float = 0.99,
    randomized_oversamples: int = 8,
    randomized_n_iter: int = 2,
    generator: torch.Generator | None = None,
) -> LowRankMatrixApproximation:
    """Compress a 2D matrix or tensor whose last dimension is the feature dimension."""

    if rank <= 0:
        raise ValueError("rank must be positive")
    original_shape = tuple(matrix.shape)
    if matrix.ndim < 2:
        raise ValueError("matrix must have at least two dimensions")
    flattened = matrix.reshape(-1, matrix.shape[-1])
    decomposition_matrix, output_device = _prepare_matrix_for_decomposition(flattened)
    max_rank = min(flattened.shape)
    target_rank = min(rank, max_rank)

    if decomposition_type == "pca":
        decomposition_type = "truncated_svd"
    if decomposition_type == "truncated_svd":
        u_full, s_full, vh_full = torch.linalg.svd(decomposition_matrix, full_matrices=False)
    elif decomposition_type == "randomized_svd":
        u_full, s_full, vh_full = randomized_svd(
            decomposition_matrix,
            rank=target_rank,
            oversamples=randomized_oversamples,
            n_iter=randomized_n_iter,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported decomposition_type: {decomposition_type}")

    selected_rank = select_rank_by_energy(s_full, energy_threshold) if adaptive_rank else target_rank
    selected_rank = min(selected_rank, target_rank, s_full.numel())
    u = u_full[:, :selected_rank].to(output_device).contiguous()
    s = s_full[:selected_rank].to(output_device).contiguous()
    vh = vh_full[:selected_rank, :].to(output_device).contiguous()
    spectral_energy = _spectral_energy(s_full, selected_rank)
    return LowRankMatrixApproximation(
        u=u,
        s=s,
        vh=vh,
        original_shape=original_shape,
        singular_values=s_full.detach(),
        spectral_energy=spectral_energy,
    )


def randomized_svd(
    matrix: torch.Tensor,
    rank: int,
    oversamples: int = 8,
    n_iter: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a randomized SVD basis for a matrix."""

    if matrix.ndim != 2:
        raise ValueError("randomized_svd expects a 2D matrix")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if oversamples < 0:
        raise ValueError("oversamples must be non-negative")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")

    rows, cols = matrix.shape
    sample_rank = min(rank + oversamples, rows, cols)
    omega = torch.randn(cols, sample_rank, device=matrix.device, dtype=matrix.dtype, generator=generator)
    sample = matrix @ omega
    for _ in range(n_iter):
        sample = matrix @ (matrix.transpose(-2, -1) @ sample)
    q, _ = torch.linalg.qr(sample, mode="reduced")
    small_matrix = q.transpose(-2, -1) @ matrix
    u_hat, singular_values, vh = torch.linalg.svd(small_matrix, full_matrices=False)
    u = q @ u_hat
    return u, singular_values, vh


def _prepare_matrix_for_decomposition(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.device]:
    """Return an SVD-safe matrix and the device where factors should live."""

    output_device = matrix.device
    decomposition_matrix = matrix.detach()
    if decomposition_matrix.dtype in {torch.float16, torch.bfloat16}:
        decomposition_matrix = decomposition_matrix.to(torch.float32)
    if decomposition_matrix.device.type == "mps":
        decomposition_matrix = decomposition_matrix.cpu()
    return decomposition_matrix, output_device


def compress_tensor(
    tensor: torch.Tensor,
    config: CompressionConfig,
    generator: torch.Generator | None = None,
) -> LowRankTensorApproximation:
    """Compress a KV tensor according to configured granularity."""

    if config.granularity == "per_head":
        if tensor.ndim != 4:
            raise ValueError("per_head compression expects shape [B, H, T, D]")
        factors = tuple(
            compress_matrix(
                tensor[:, head_idx, :, :],
                rank=config.rank,
                decomposition_type=config.decomposition_type,  # type: ignore[arg-type]
                adaptive_rank=config.adaptive_rank,
                energy_threshold=config.energy_threshold,
                randomized_oversamples=config.randomized_oversamples,
                randomized_n_iter=config.randomized_n_iter,
                generator=generator,
            )
            for head_idx in range(tensor.shape[1])
        )

        # print("u.shape:", factors[0].u.shape)
        # print("s.shape:", factors[0].s.shape)
        # print("vh.shape:", factors[0].vh.shape)
        # print("original_shape:", factors[0].original_shape)

        return LowRankTensorApproximation(
            factors=factors,
            original_shape=tuple(tensor.shape),
            granularity=config.granularity,
        )

    factor = compress_matrix(
        tensor,
        rank=config.rank,
        decomposition_type=config.decomposition_type,  # type: ignore[arg-type]
        adaptive_rank=config.adaptive_rank,
        energy_threshold=config.energy_threshold,
        randomized_oversamples=config.randomized_oversamples,
        randomized_n_iter=config.randomized_n_iter,
        generator=generator,
    )
    return LowRankTensorApproximation(
        factors=(factor,),
        original_shape=tuple(tensor.shape),
        granularity=config.granularity,
    )


def compress_kv_layer(layer: KVLayerCache, config: CompressionConfig) -> LowRankKVLayer:
    """Compute low-rank approximations for one KV layer."""

    return LowRankKVLayer(
        layer_idx=layer.layer_idx,
        key=compress_tensor(layer.key, config),
        value=compress_tensor(layer.value, config),
    )


def compress_kv_snapshot(snapshot: KVCacheSnapshot, config: CompressionConfig) -> LowRankKVSnapshot:
    """Compute low-rank approximations for all layers in a KV snapshot."""

    # print("snapshot.layers[0].key.shape:", snapshot.layers[0].key.shape)
    # print("snapshot.layers[0].value.shape:", snapshot.layers[0].value.shape)

    return LowRankKVSnapshot(
        layers=tuple(compress_kv_layer(layer, config) for layer in snapshot.layers),
    )


def _spectral_energy(singular_values: torch.Tensor, rank: int) -> float:
    energy = singular_values.square()
    total = energy.sum()
    if total <= 0:
        return 1.0
    return float((energy[:rank].sum() / total).item())


def _relative_fro_error(target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(target)
    numerator = torch.linalg.vector_norm(target - reconstruction)
    if denominator <= 0:
        return numerator
    return numerator / denominator


def _low_rank_metrics(
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    rank: int,
    spectral_energy: float,
    factor_bytes: int,
) -> LowRankMetrics:
    original_bytes = _tensor_nbytes(target)
    return LowRankMetrics(
        rank=rank,
        relative_error=float(_relative_fro_error(target, reconstruction).item()),
        mse=float(torch.mean((target - reconstruction).square()).item()),
        spectral_energy=spectral_energy,
        original_bytes=original_bytes,
        factor_bytes=factor_bytes,
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
