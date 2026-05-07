import pytest
import torch

from models.loader import resolve_torch_dtype


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
    ],
)
def test_resolve_torch_dtype(name: str, expected: torch.dtype) -> None:
    assert resolve_torch_dtype(name) == expected


def test_resolve_torch_dtype_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_torch_dtype("int8")
