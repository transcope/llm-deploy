"""
Qwen3 pipeline patch for llmcompressor GPTQ.

Why this file exists
--------------------
llmcompressor 0.4.0 的 GPTQModifier 优先选择 sequential pipeline, 失败后降级到
layer_sequential, 最后降级到 basic pipeline. 对 Qwen3 (transformers 4.51.0):

  1. sequential pipeline: torch.fx 追踪 Qwen3ForCausalLM 时, `**kwargs: Unpack[FlashAttentionKwargs]`
     不被 HFTracer 支持, 追踪失败 -> 降级.
  2. layer_sequential pipeline: 第一层能跑通 (15s 处理 16 batches, 1s/it),
     但第二层失败. 原因是 Qwen3Model.forward 在循环外预计算
     `position_embeddings = self.rotary_emb(hidden_states, position_ids)`,
     并把它作为关键字参数传给每个 decoder layer. 而 layer_sequential 的
     `to_next_layer_kwargs` 仅从上一层输出 `(hidden_states,)` 推断下一层输入,
     position_embeddings 丢失 -> TypeError -> 降级.
  3. basic pipeline: 每次校准 batch 跑一次完整模型前向 (8B 参数穿过 36 层),
     然后 `on_sequential_batch_end` 量化全部 252 个 Linear. 更严重的是
     `on_sequential_batch_end` 内 `del self._num_samples[module]`, 导致
     Hessian 在每个 batch 后被清空, 下一个 batch 又从 1 个样本重新累积.
     最终保存的权重仅来自最后一个 batch 的 1-sample Hessian 估计,
     精度远劣于真正的 N-sample 累积; 同时吞吐 734s/it, 128 samples 需 25h.

此外 layer_sequential pipeline 还有一个 llmcompressor 0.4.0 的 bug:
helpers.match_modules 按 module 名字符串字典序排序, 对 ≥10 层的模型得到
0,1,10,11,...,19,2,20,... 的顺序. layer_sequential 把 layer 1 的输出当
layer 10 的输入, Hessian 在错位激活上累积, error 飙到 5e8. 本 patch 同时
monkey-patch match_modules 为数值排序.

Fix
---
本 patch 在 oneshot 之前做两件事:
  (a) monkey-patch llmcompressor.pipelines.layer_sequential.helpers.match_modules
      为数值排序版本, 让 layer 列表按 0,1,2,...,35 的真实顺序处理.
  (b) 给所有 Qwen3DecoderLayer 安装 forward_pre_hook, 共享一个
      _PositionEmbeddingsCache, 在 capture 阶段缓存 (cos, sin), 在后续 layer
      独立调用时恢复 position_embeddings (详见下面 "batch_idx 的识别").

(a) 解决 layer 错位, (b) 解决 position_embeddings 缺失. 两者缺一不可:
没有 (a), layer 顺序错, Hessian 在错位激活上累积; 没有 (b), layer_sequential
在第二层就 TypeError 退回 basic pipeline.

batch_idx 的识别
----------------
layer_sequential pipeline 对每个 layer 都按 batch_index 顺序遍历 dataloader,
但在 layer N 的 Calibrating pass 里调用 layer(**inputs) 时不会传 batch_index
给 layer. 所以 hook 无法直接拿到 batch_idx.

解决: 每个 layer 维护两个独立计数器
  - write_ctr: pe 被传入时自增 (capture 阶段: model 完整前向, layer 收到真实 pe)
  - read_ctr : pe 缺失时自增 (layer_sequential 后续 layer 独立调用)
两个计数器各自从 0 严格递增, 与 layer_sequential 的 batch 遍历顺序 (0..B-1)
对齐. capture 阶段所有 layer 都会被前向调用, 所以 write_ctr 在 layer 0..N-1
上都各自递增 B 次, 写入 shared_cache[0..B-1] (同一个 batch 多次写不会冲突,
因为同一个 batch 的 pe 对所有 layer 是相同的). 后续 layer 在 calibration
阶段独立调用时, read_ctr 从 0 递增, 顺序读取 shared_cache[0..B-1].

注意: Propagating pass 会 disable_hooks (HooksMixin.disable_hooks),
所以 hook 不会在 Propagating 阶段触发, 计数器只在 Calibrating pass 自增,
这与 capture 阶段 (model 完整前向, 各 layer 调用 B 次) 的顺序一致.

配对配置
--------
configs/gptq_smoke.yaml / gptq_4bit_v100.yaml 加上:
    enable_qwen3_pipeline_patch: true
本 patch 默认只对 Qwen3ForCausalLM 生效; 对其他架构无副作用.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_llmdeploy_qwen3_patch_installed"
_HANDLE_ATTR = "_llmdeploy_patch_handle"
_CACHE_ATTR = "_llmdeploy_pe_cache"            # shared cache object (on every layer)
_WRITE_COUNTER_ATTR = "_llmdeploy_write_ctr"   # per-layer: increments when pe IS provided
_READ_COUNTER_ATTR = "_llmdeploy_read_ctr"     # per-layer: increments when pe is NOT provided


class _PositionEmbeddingsCache:
    """所有 decoder layer 共享的 position_embeddings 缓存, 按 batch_idx 索引."""

    def __init__(self):
        self._store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def set(self, batch_idx: int, pe: Tuple[torch.Tensor, torch.Tensor]) -> None:
        self._store[batch_idx] = pe

    def get(self, batch_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self._store.get(batch_idx)

    def clear(self) -> None:
        self._store.clear()


def _is_qwen3_model(model: nn.Module) -> bool:
    """判断是否为 Qwen3 架构 (Qwen3ForCausalLM / Qwen3Model)."""
    cls_name = model.__class__.__name__
    return cls_name.startswith("Qwen3") and ("ForCausalLM" in cls_name or cls_name == "Qwen3Model")


def _find_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """返回所有 Qwen3DecoderLayer 实例 (按模型出现顺序)."""
    layers = []
    for _, module in model.named_modules():
        if module.__class__.__name__ == "Qwen3DecoderLayer":
            layers.append(module)
    return layers


def _make_pre_hook(layer: nn.Module, cache: _PositionEmbeddingsCache, layer_index: int):
    """构造 forward_pre_hook: 缓存/恢复 position_embeddings.

    用两个独立计数器推断 batch_idx (见模块 docstring 末尾 "batch_idx 的识别"):
      - write_ctr: pe 被传入时自增 -> capture 阶段写 cache 的下标
      - read_ctr:  pe 缺失时自增   -> layer_sequential 后续 layer 读 cache 的下标
    两个计数器各自从 0 严格递增, 与 layer_sequential 的 batch 遍历顺序对齐.
    """

    def hook(module: nn.Module, args: tuple, kwargs: Dict[str, Any]):
        pe = kwargs.get("position_embeddings", None)
        if pe is not None:
            # 真实调用 (model 完整前向): 缓存 (cos, sin), 供后续 layer 复用.
            w = getattr(module, _WRITE_COUNTER_ATTR, 0)
            cache.set(w, pe)
            object.__setattr__(module, _WRITE_COUNTER_ATTR, w + 1)
        else:
            # layer_sequential 后续 layer: position_embeddings 缺失, 从共享缓存恢复.
            r = getattr(module, _READ_COUNTER_ATTR, 0)
            cached_pe = cache.get(r)
            if cached_pe is not None:
                kwargs["position_embeddings"] = cached_pe
                object.__setattr__(module, _READ_COUNTER_ATTR, r + 1)
            elif layer_index == 0:
                # layer 0 也不该缺 pe (capture 阶段 model 完整前向时一定有);
                # 真到这一步说明 hook 顺序异常, 留个日志不阻塞.
                logger.warning(
                    f"[qwen3_patch] layer 0 read {r} 缺 position_embeddings 且缓存为空"
                )

        return None  # 直接 mutate kwargs, 不改 args

    return hook


# ===== match_modules 数值排序补丁 =====
#
# llmcompressor 0.4.0 的 layer_sequential.pipeline 调用
# `match_modules(model, target_names)` 拿到要逐层处理的 module 列表,
# 该函数 (helpers.py:21) 用 `sorted(names_layers, key=lambda x: x[0])`
# 按 module 名字字符串字典序排序. 对 36 层的 Qwen3:
#   model.layers.0, model.layers.1, model.layers.10, model.layers.11, ...,
#   model.layers.19, model.layers.2, model.layers.20, ...
# 字典序把 layer 10 排在 layer 2 前面, 导致 layer_sequential 把 layer 1 的输出
# 当成 layer 10 的输入, Hessian 在错位的激活上累积, 量化 error 飙到 5e8.
#
# 修复: monkey-patch match_modules, 提取名字末尾的数字做数值排序.
# 只在 Qwen3 (或一般 "xxx.layers.N" 命名) 上生效, 其他情况回退原排序.
#
# 通过替换 llmcompressor.pipelines.layer_sequential.helpers.match_modules
# (pipeline.py 在 import 时已 from helpers import match_modules, 所以也要替换
# pipeline 模块里的引用).

_ORIG_MATCH_MODULES = None
_PATCH_INSTALLED_FLAG = "_llmdeploy_match_modules_patched"


def _natural_sort_key(name: str):
    """把 'model.layers.10' 拆成 ['model', 'layers', 10], 数字部分按数值比."""
    import re
    parts = []
    for tok in re.split(r"(\d+)", name):
        if tok.isdigit():
            parts.append((1, int(tok)))  # 数字: (类型, 数值)
        elif tok:
            parts.append((0, tok))  # 非数字: (类型, 字符串)
    return parts


def _patched_match_modules(model, target_names):
    """match_modules 的数值排序版本."""
    from llmcompressor.pipelines.layer_sequential import helpers as _helpers

    names_layers = [
        (name, module)
        for name, module in model.named_modules()
        if _helpers.find_name_or_class_matches(name, module, target_names)
    ]
    names_layers.sort(key=lambda nl: _natural_sort_key(nl[0]))
    return [layer for _name, layer in names_layers]


def install_match_modules_patch() -> bool:
    """替换 llmcompressor 的 match_modules 为数值排序版本.

    幂等: 已安装则跳过. 必须在 oneshot() 之前调用.
    """
    global _ORIG_MATCH_MODULES
    try:
        from llmcompressor.pipelines.layer_sequential import helpers as _helpers
        from llmcompressor.pipelines.layer_sequential import pipeline as _pipeline
    except ImportError:
        return False

    if getattr(_helpers, _PATCH_INSTALLED_FLAG, False):
        return True

    _ORIG_MATCH_MODULES = _helpers.match_modules
    _helpers.match_modules = _patched_match_modules
    # pipeline.py 顶部 `from .helpers import match_modules` 已把名字绑到 pipeline 模块,
    # 必须同步替换, 否则 pipeline.run_layer_sequential 仍用旧引用.
    _pipeline.match_modules = _patched_match_modules
    object.__setattr__(_helpers, _PATCH_INSTALLED_FLAG, True)

    logger.info("[qwen3_patch] 已 monkey-patch match_modules 为数值排序 (修复 lexical 排序导致层错位)")
    return True


def uninstall_match_modules_patch() -> None:
    """还原 match_modules (幂等)."""
    global _ORIG_MATCH_MODULES
    try:
        from llmcompressor.pipelines.layer_sequential import helpers as _helpers
        from llmcompressor.pipelines.layer_sequential import pipeline as _pipeline
    except ImportError:
        return

    if not getattr(_helpers, _PATCH_INSTALLED_FLAG, False):
        return

    if _ORIG_MATCH_MODULES is not None:
        _helpers.match_modules = _ORIG_MATCH_MODULES
        _pipeline.match_modules = _ORIG_MATCH_MODULES
        _ORIG_MATCH_MODULES = None
    if hasattr(_helpers, _PATCH_INSTALLED_FLAG):
        object.__delattr__(_helpers, _PATCH_INSTALLED_FLAG)


def install_qwen3_pipeline_patch(model: nn.Module) -> bool:
    """给 Qwen3 模型安装 layer_sequential 兼容补丁.

    :return: True 表示已安装 (或模型是 Qwen3 且已安装); False 表示模型不是 Qwen3.
    """
    if not _is_qwen3_model(model):
        return False

    layers = _find_decoder_layers(model)
    if not layers:
        logger.warning("[qwen3_patch] Qwen3 model 但未发现 Qwen3DecoderLayer, 跳过 patch")
        return False

    # 先装 match_modules 数值排序补丁 (修复 layer_sequential 的 lexical 排序 bug).
    # 没有 ≥10 层就不需要, 但装上无害, 且 Qwen3-8B 有 36 层必须装.
    install_match_modules_patch()

    # 所有 layer 共享同一个 cache
    cache = _PositionEmbeddingsCache()

    installed = 0
    for layer_index, layer in enumerate(layers):
        if getattr(layer, _PATCH_MARKER, False):
            installed += 1
            continue
        handle = layer.register_forward_pre_hook(
            _make_pre_hook(layer, cache, layer_index), with_kwargs=True
        )
        object.__setattr__(layer, _PATCH_MARKER, True)
        object.__setattr__(layer, _HANDLE_ATTR, handle)
        object.__setattr__(layer, _CACHE_ATTR, cache)
        object.__setattr__(layer, _WRITE_COUNTER_ATTR, 0)
        object.__setattr__(layer, _READ_COUNTER_ATTR, 0)
        installed += 1

    logger.info(
        f"[qwen3_patch] 已为 {installed}/{len(layers)} 个 Qwen3DecoderLayer 安装 "
        f"position_embeddings 缓存 hook (shared cache), 使 layer_sequential pipeline 可用"
    )
    return True


def uninstall_qwen3_pipeline_patch(model: nn.Module) -> None:
    """卸载补丁 (oneshot 完成后调用, 避免影响后续推理)."""
    # 先还原 match_modules (无论是否 Qwen3 都尝试, 保持对称)
    uninstall_match_modules_patch()

    if not _is_qwen3_model(model):
        return
    cache_to_clear = None
    for layer in _find_decoder_layers(model):
        handle = getattr(layer, _HANDLE_ATTR, None)
        if handle is not None:
            handle.remove()
            object.__delattr__(layer, _HANDLE_ATTR)
        if getattr(layer, _PATCH_MARKER, False):
            object.__delattr__(layer, _PATCH_MARKER)
        if hasattr(layer, _WRITE_COUNTER_ATTR):
            object.__delattr__(layer, _WRITE_COUNTER_ATTR)
        if hasattr(layer, _READ_COUNTER_ATTR):
            object.__delattr__(layer, _READ_COUNTER_ATTR)
        cache = getattr(layer, _CACHE_ATTR, None)
        if cache is not None:
            cache_to_clear = cache
            object.__delattr__(layer, _CACHE_ATTR)
    if cache_to_clear is not None:
        cache_to_clear.clear()
