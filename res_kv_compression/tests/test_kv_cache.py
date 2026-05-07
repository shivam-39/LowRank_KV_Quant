from types import SimpleNamespace

import pytest
import torch

from models.kv_cache import AttentionHookCapture, KVCacheExtractor, KVCacheSnapshot


class ToyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        batch, seq_len = input_ids.shape
        key = torch.arange(batch * 2 * seq_len * 3, dtype=torch.float32).reshape(batch, 2, seq_len, 3)
        value = key + 100.0
        past_key_values = ((key, value), (key + 1.0, value + 1.0))
        return SimpleNamespace(past_key_values=past_key_values)


class ToyAttention(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape
        key = torch.ones(batch, 2, seq_len, 4)
        value = torch.full_like(key, 2.0)
        return x, None, (key, value)


class ToyWithAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = ToyAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)[0]


def test_extract_from_inputs_builds_layer_snapshots() -> None:
    input_ids = torch.ones(1, 5, dtype=torch.long)
    snapshot = KVCacheExtractor().extract_from_inputs(ToyCausalLM(), input_ids)

    assert snapshot.num_layers == 2
    assert snapshot.layers[0].key.shape == (1, 2, 5, 3)
    assert snapshot.layers[1].value[0, 0, 0, 0].item() == 101.0
    assert snapshot.metadata["source"] == "past_key_values"


def test_snapshot_save_writes_manifest_and_layer_files(tmp_path) -> None:
    snapshot = KVCacheExtractor().extract_from_inputs(ToyCausalLM(), torch.ones(1, 3, dtype=torch.long))

    snapshot.save(tmp_path)

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "layer_000.pt").exists()
    assert (tmp_path / "layer_001.pt").exists()
    loaded = torch.load(tmp_path / "layer_000.pt")
    assert set(loaded) == {"key", "value"}


def test_snapshot_rejects_empty_cache() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        KVCacheSnapshot(layers=())


def test_attention_hook_capture_collects_present_kv() -> None:
    model = ToyWithAttention()
    capture = AttentionHookCapture()
    capture.register(model)
    _ = model(torch.zeros(1, 6, 8))
    snapshot = capture.snapshot()
    capture.close()

    assert snapshot.num_layers == 1
    assert snapshot.layers[0].key.shape == (1, 2, 6, 4)
    assert torch.equal(snapshot.layers[0].value, torch.full((1, 2, 6, 4), 2.0))
