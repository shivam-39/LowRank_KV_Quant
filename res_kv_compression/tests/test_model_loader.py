import sys
from types import SimpleNamespace

import pytest
import torch

from models.loader import load_causal_lm_and_tokenizer, resolve_torch_dtype
from utils.config import ModelConfig


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


def test_auto_device_retries_without_device_map_when_accelerate_path_fails(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeTokenizer:
        pad_token = None
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, model_name: str, trust_remote_code: bool = False) -> "FakeTokenizer":
            assert model_name == "toy-model"
            assert trust_remote_code is False
            return cls()

    class FakeModel:
        def __init__(self) -> None:
            self.device = None
            self.evaluated = False

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs) -> "FakeModel":
            assert model_name == "toy-model"
            calls.append(kwargs)
            if kwargs.get("device_map") == "auto":
                raise ValueError(
                    "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting "
                    "`torch.set_default_device(device)` requires `accelerate`."
                )
            return cls()

        def to(self, device: str) -> "FakeModel":
            self.device = device
            return self

        def eval(self) -> None:
            self.evaluated = True

    fake_transformers = SimpleNamespace(AutoModelForCausalLM=FakeModel, AutoTokenizer=FakeTokenizer)
    fake_transformers_utils = SimpleNamespace(is_accelerate_available=lambda: True)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_transformers_utils)

    loaded = load_causal_lm_and_tokenizer(ModelConfig(model_name="toy-model", device="auto"))

    assert calls[0]["device_map"] == "auto"
    assert "device_map" not in calls[1]
    assert calls[1]["dtype"] == torch.float16
    assert loaded.model.device in {"cpu", "cuda", "mps"}
    assert loaded.model.evaluated is True


def test_auto_device_without_accelerate_does_not_pass_device_map(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeTokenizer:
        pad_token = "</s>"
        eos_token = "</s>"

        @classmethod
        def from_pretrained(cls, model_name: str, trust_remote_code: bool = False) -> "FakeTokenizer":
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs) -> "FakeModel":
            calls.append(kwargs)
            return cls()

        def to(self, device: str) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

    fake_transformers = SimpleNamespace(AutoModelForCausalLM=FakeModel, AutoTokenizer=FakeTokenizer)
    fake_transformers_utils = SimpleNamespace(is_accelerate_available=lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_transformers_utils)

    load_causal_lm_and_tokenizer(ModelConfig(model_name="toy-model", device="auto"))

    assert "device_map" not in calls[0]
