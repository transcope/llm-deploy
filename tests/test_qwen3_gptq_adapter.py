"""qwen3_gptq_adapter 单元测试 (离线, mock gptqmodel, 不依赖真实模型).

覆盖:
  1. install 注入 Qwen3GPTQ 到 MODEL_MAP / SUPPORTED_MODELS
  2. install 幂等: 二次调用不重复注入, 不破坏已注入的类
  3. uninstall 还原原始 MODEL_MAP / SUPPORTED_MODELS
  4. gptqmodel 未安装时 install 返回 False, 不抛异常
  5. SUPPORTED_MODELS 同时支持 list / set 两种容器
  6. _define_qwen3_gptq_class 产出的类 layer_type 为 Qwen3DecoderLayer
"""
import importlib
import sys
import types
from unittest import mock

import pytest


# ---------- mock gptqmodel ----------
class _FakeQwen2GPTQ:
    """模拟 gptqmodel.models.definitions.qwen2.Qwen2GPTQ."""
    layer_type = "Qwen2DecoderLayer"


def _build_fake_gptqmodel_module(supported_models_container):
    """构造一个 fake gptqmodel 包, 注入到 sys.modules.

    :param supported_models_container: list 或 set, 作为 SUPPORTED_MODELS
    """
    # 顶层 gptqmodel 包
    gptqmodel_pkg = types.ModuleType("gptqmodel")
    gptqmodel_pkg.__path__ = []

    # gptqmodel.models 子包
    models_pkg = types.ModuleType("gptqmodel.models")
    models_pkg.__path__ = []
    # gptqmodel.models.auto (含 MODEL_MAP)
    auto_mod = types.ModuleType("gptqmodel.models.auto")
    auto_mod.MODEL_MAP = {
        "qwen2": _FakeQwen2GPTQ,
    }
    # gptqmodel.models.definitions 子包
    definitions_pkg = types.ModuleType("gptqmodel.models.definitions")
    definitions_pkg.__path__ = []
    # gptqmodel.models.definitions.qwen2 (含 Qwen2GPTQ)
    qwen2_def = types.ModuleType("gptqmodel.models.definitions.qwen2")
    qwen2_def.Qwen2GPTQ = _FakeQwen2GPTQ
    # gptqmodel.utils 子包
    utils_pkg = types.ModuleType("gptqmodel.utils")
    utils_pkg.__path__ = []
    # gptqmodel.utils.model (含 SUPPORTED_MODELS)
    model_mod = types.ModuleType("gptqmodel.utils.model")
    model_mod.SUPPORTED_MODELS = supported_models_container

    # 组装
    sys.modules["gptqmodel"] = gptqmodel_pkg
    sys.modules["gptqmodel.models"] = models_pkg
    sys.modules["gptqmodel.models.auto"] = auto_mod
    sys.modules["gptqmodel.models.definitions"] = definitions_pkg
    sys.modules["gptqmodel.models.definitions.qwen2"] = qwen2_def
    sys.modules["gptqmodel.utils"] = utils_pkg
    sys.modules["gptqmodel.utils.model"] = model_mod

    return auto_mod, model_mod


def _unload_gptqmodel():
    for k in list(sys.modules):
        if k == "gptqmodel" or k.startswith("gptqmodel."):
            del sys.modules[k]


@pytest.fixture
def adapter_module():
    """每个测试独立加载 adapter 模块, 隔离全局状态."""
    _unload_gptqmodel()
    # 清掉可能已加载的 adapter
    sys.modules.pop("qwen3_gptq_adapter", None)
    yield importlib.import_module("qwen3_gptq_adapter")
    # teardown: 卸载 adapter 和 mock
    sys.modules.pop("qwen3_gptq_adapter", None)
    _unload_gptqmodel()


def test_install_injects_into_model_map_and_supported_models(adapter_module):
    """install 把 Qwen3GPTQ 注入 MODEL_MAP 和 SUPPORTED_MODELS (list 版本)."""
    _build_fake_gptqmodel_module(supported_models_container=["qwen2"])

    ok = adapter_module.install_qwen3_gptq_adapter()
    assert ok is True

    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.utils.model import SUPPORTED_MODELS

    assert "qwen3" in MODEL_MAP
    cls = MODEL_MAP["qwen3"]
    assert cls.layer_type == "Qwen3DecoderLayer"
    assert cls.__name__ == "Qwen3GPTQ"
    # 继承自 Qwen2GPTQ
    from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ
    assert issubclass(cls, Qwen2GPTQ)
    # SUPPORTED_MODELS 也包含 qwen3
    assert "qwen3" in SUPPORTED_MODELS


def test_install_idempotent(adapter_module):
    """二次 install 不重复注入, 类对象保持不变."""
    _build_fake_gptqmodel_module(supported_models_container=["qwen2"])

    adapter_module.install_qwen3_gptq_adapter()
    from gptqmodel.models.auto import MODEL_MAP
    cls_first = MODEL_MAP["qwen3"]

    # 二次 install: 应返回 True 且不替换已注入的类
    ok = adapter_module.install_qwen3_gptq_adapter()
    assert ok is True
    cls_second = MODEL_MAP["qwen3"]
    assert cls_second is cls_first, "二次 install 替换了已注入的类 (违反幂等)"

    # SUPPORTED_MODELS 不会被重复 append
    from gptqmodel.utils.model import SUPPORTED_MODELS
    assert SUPPORTED_MODELS.count("qwen3") == 1


def test_install_with_set_supported_models(adapter_module):
    """SUPPORTED_MODELS 为 set 时也能正确注入."""
    _build_fake_gptqmodel_module(supported_models_container={"qwen2"})

    ok = adapter_module.install_qwen3_gptq_adapter()
    assert ok is True
    from gptqmodel.utils.model import SUPPORTED_MODELS
    assert "qwen3" in SUPPORTED_MODELS


def test_install_returns_false_when_gptqmodel_missing(adapter_module):
    """gptqmodel 未安装时 install 返回 False, 不抛异常."""
    _unload_gptqmodel()
    # 让 import gptqmodel 失败
    with mock.patch.dict(sys.modules, {"gptqmodel": None}):
        ok = adapter_module.install_qwen3_gptq_adapter()
    assert ok is False


def test_uninstall_restores_original_state(adapter_module):
    """uninstall 还原 MODEL_MAP / SUPPORTED_MODELS 到安装前."""
    _build_fake_gptqmodel_module(supported_models_container=["qwen2"])
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.utils.model import SUPPORTED_MODELS

    original_map_keys = set(MODEL_MAP.keys())
    original_supported = list(SUPPORTED_MODELS)

    adapter_module.install_qwen3_gptq_adapter()
    assert "qwen3" in MODEL_MAP
    assert "qwen3" in SUPPORTED_MODELS

    adapter_module.uninstall_qwen3_gptq_adapter()
    assert "qwen3" not in MODEL_MAP
    assert set(MODEL_MAP.keys()) == original_map_keys
    assert SUPPORTED_MODELS == original_supported


def test_uninstall_idempotent(adapter_module):
    """uninstall 多次调用不抛异常."""
    _build_fake_gptqmodel_module(supported_models_container=["qwen2"])
    adapter_module.install_qwen3_gptq_adapter()
    adapter_module.uninstall_qwen3_gptq_adapter()
    # 二次 uninstall 不应抛
    adapter_module.uninstall_qwen3_gptq_adapter()


def test_qwen3_gptq_class_layer_type():
    """_define_qwen3_gptq_class 产出的类 layer_type 为 Qwen3DecoderLayer."""
    _build_fake_gptqmodel_module(supported_models_container=["qwen2"])
    sys.modules.pop("qwen3_gptq_adapter", None)
    adapter = importlib.import_module("qwen3_gptq_adapter")
    cls = adapter._define_qwen3_gptq_class()
    assert cls.layer_type == "Qwen3DecoderLayer"
    assert cls.__name__ == "Qwen3GPTQ"
    # 清理
    sys.modules.pop("qwen3_gptq_adapter", None)
    _unload_gptqmodel()


if __name__ == "__main__":
    # 不依赖 pytest 也能跑
    test_qwen3_gptq_class_layer_type()
    # 其余用例需要 pytest fixture, 直接 pytest 调用
    print("非 pytest 模式仅运行 test_qwen3_gptq_class_layer_type (OK). "
          "完整测试请用: pytest tests/test_qwen3_gptq_adapter.py -v")
