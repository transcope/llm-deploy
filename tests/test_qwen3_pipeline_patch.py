"""单元测试: 验证 qwen3_pipeline_patch 在 mock Qwen3 模型上的行为.

重点验证 v2 修复: 所有 decoder layer 共享一个 _PositionEmbeddingsCache.
原 v1 bug 是 cache 挂在每个 layer 上, layer 1 读不到 layer 0 缓存的 pe,
导致 layer_sequential 在第二层就回退到 basic pipeline.

模拟 layer_sequential pipeline 的真实调用顺序:
  1. capture_first_layer_intermediates: 对每个 batch 跑一次完整 model 前向,
     layer 0 收到真实 position_embeddings (这里 cache 被填充).
  2. 之后对 layer 1..N-1, 每个 layer 独立按 batch 顺序调用, 不传 pe,
     hook 应该从 shared_cache 取出对应 batch_idx 的 pe 塞回去.
"""
# 注意: llm_deploy/ 路径由 tests/conftest.py 统一注入 sys.path, 此处不再硬编码

import pytest
import torch
import torch.nn as nn
from qwen3_pipeline_patch import (
    _is_qwen3_model,
    _find_decoder_layers,
    install_qwen3_pipeline_patch,
    uninstall_qwen3_pipeline_patch,
)


class Qwen3DecoderLayer(nn.Module):
    """模拟 transformers Qwen3DecoderLayer: forward 需要 position_embeddings."""

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        if position_embeddings is None:
            raise RuntimeError("position_embeddings missing!")
        # 模拟真实 forward 对 pe 的依赖: 用 cos 做一次变换, 确认 pe 真的被用到
        cos, _sin = position_embeddings
        return (self.linear(hidden_states) + cos.mean(dim=-1).squeeze(-1),)


class Qwen3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3DecoderLayer(i) for i in range(3)])


class Qwen3ForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen3Model()


def _model_forward(model: Qwen3ForCausalLM, hs: torch.Tensor, pe):
    """模拟 Qwen3Model.forward: 把同一个 pe 传给每一层 (cache 应被 layer 0 填充)."""
    for layer in model.model.layers:
        hs = layer(hs, position_embeddings=pe)[0]
    return hs


def test_is_qwen3_and_find_layers():
    m = Qwen3ForCausalLM()
    assert _is_qwen3_model(m)
    layers = _find_decoder_layers(m)
    assert len(layers) == 3


def test_shared_cache_and_layer_sequential_flow():
    """v2 修复核心: 所有 layer 共享同一 cache, 后续 layer 按 batch 从 cache 取 pe."""
    m = Qwen3ForCausalLM()
    layers = _find_decoder_layers(m)

    installed = install_qwen3_pipeline_patch(m)
    assert installed, "install failed"

    # 每层共享的 cache 应该是同一个对象
    caches = [getattr(l, "_llmdeploy_pe_cache", None) for l in layers]
    assert all(c is caches[0] for c in caches), "cache 不是共享的!"

    NUM_BATCHES = 4
    batched_hs = [torch.randn(1, 4) for _ in range(NUM_BATCHES)]
    # 每个 batch 一对独立的 (cos, sin), 用来验证 cache 按 batch_idx 索引
    batched_pe = [(torch.randn(1, 4, 4), torch.randn(1, 4, 4)) for _ in range(NUM_BATCHES)]

    # ===== 阶段 1: capture_first_layer_intermediates =====
    # 对每个 batch 跑一次完整 model 前向, layer 0 收到真实 pe 并写入 shared_cache
    for b in range(NUM_BATCHES):
        _model_forward(m, batched_hs[b], batched_pe[b])
    assert sorted(caches[0]._store.keys()) == list(range(NUM_BATCHES)), (
        f"cache 应该存了 {NUM_BATCHES} 个 batch, 实际存了 {sorted(caches[0]._store.keys())}"
    )

    # ===== 阶段 2: layer_sequential 后续 layer 独立调用 =====
    # 每个 layer 按 batch 0..B-1 顺序独立 forward, 不传 pe.
    # 同时安装探针 hook 验证 patch hook 恢复的 pe 等于 batched_pe[b]
    # (probe 在 patch hook 之后注册, patch hook 先跑并 mutate kwargs,
    #  probe 看到的就是已恢复的 pe).
    captured = {}

    def probe(module, args, kwargs):
        captured["pe"] = kwargs.get("position_embeddings")
        return None

    probe_handles = []
    for layer_idx in [1, 2]:
        probe_handles.append(layers[layer_idx].register_forward_pre_hook(
            probe, with_kwargs=True
        ))

    for layer_idx in [1, 2]:
        for b in range(NUM_BATCHES):
            out = layers[layer_idx](batched_hs[b])  # 不传 pe!
            assert out[0].shape == (1, 4), f"layer {layer_idx} batch {b} shape 错"
            got_cos = captured["pe"][0]
            exp_cos = batched_pe[b][0]
            assert torch.allclose(got_cos, exp_cos), (
                f"layer {layer_idx} batch {b} 收到的 pe 不匹配! "
                f"max|Δ|={(got_cos - exp_cos).abs().max().item()}"
            )
    for h in probe_handles:
        h.remove()

    # ===== 阶段 3: 卸载补丁 =====
    uninstall_qwen3_pipeline_patch(m)

    # 卸载后, layer 1 不传 pe 必须抛错
    with pytest.raises(RuntimeError):
        layers[1](batched_hs[0])
