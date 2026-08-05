"""单元测试: 验证 match_modules 数值排序补丁.

模拟 36 个 layer (model.layers.0 .. model.layers.35), 验证 patch 后
match_modules 返回的顺序是 0,1,2,...,35 而非字典序 0,1,10,11,...,19,2,20,...
"""
# 注意: scripts/ 路径由 tests/conftest.py 统一注入 sys.path, 此处不再硬编码

import torch
import torch.nn as nn
from qwen3_pipeline_patch import (
    _natural_sort_key,
    install_match_modules_patch,
    uninstall_match_modules_patch,
)


def test_natural_sort_key():
    # 字典序: "10" < "2" -> 错
    # 数值序: 2 < 10 -> 对
    names = [f"model.layers.{i}" for i in range(36)]
    lexical = sorted(names)
    natural = sorted(names, key=_natural_sort_key)
    # 字典序前几个: 0,1,10,11,...,19,2,20,...
    assert lexical[:3] == ["model.layers.0", "model.layers.1", "model.layers.10"], lexical[:3]
    # 数值序前几个: 0,1,2,3,...
    assert natural[:5] == ["model.layers.0", "model.layers.1", "model.layers.2",
                           "model.layers.3", "model.layers.4"], natural[:5]
    assert natural[-3:] == ["model.layers.33", "model.layers.34", "model.layers.35"], natural[-3:]


def test_match_modules_patch():
    # 装补丁
    installed = install_match_modules_patch()
    assert installed, "install_match_modules_patch 失败"

    try:
        from llmcompressor.pipelines.layer_sequential import helpers as _helpers
        from llmcompressor.pipelines.layer_sequential import pipeline as _pipeline

        # 验证 helpers 和 pipeline 模块里的 match_modules 都被替换了
        assert _helpers.match_modules is _pipeline.match_modules, "helpers 和 pipeline 的 match_modules 不一致"
        assert _helpers.match_modules.__name__ == "_patched_match_modules", \
            f"match_modules 未被替换: {_helpers.match_modules.__name__}"

        # 构造 mock model: 36 个 Qwen3DecoderLayer
        class Qwen3DecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

        class Qwen3Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Qwen3DecoderLayer() for _ in range(36)])

        class Qwen3ForCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Qwen3Model()

        model = Qwen3ForCausalLM()

        # match_modules 按 class name "Qwen3DecoderLayer" 匹配
        result = _helpers.match_modules(model, ["Qwen3DecoderLayer"])
        # 取出名字
        name_by_module = {id(m): n for n, m in model.named_modules()}
        result_names = [name_by_module[id(m)] for m in result]

        # 必须是数值序
        expected = [f"model.layers.{i}" for i in range(36)]
        assert result_names == expected, f"顺序错! 期望数值序, 实际: {result_names[:15]}..."
    finally:
        uninstall_match_modules_patch()


def test_uninstall_restores_original():
    """卸载后 match_modules 恢复原函数 (名字不再是 _patched_match_modules)."""
    install_match_modules_patch()
    uninstall_match_modules_patch()
    from llmcompressor.pipelines.layer_sequential import helpers as _helpers
    assert _helpers.match_modules.__name__ != "_patched_match_modules", "卸载后 match_modules 仍是 patched"
