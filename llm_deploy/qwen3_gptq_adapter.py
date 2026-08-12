#!/usr/bin/env python3
"""
gptqmodel 2.0 Qwen3 兼容适配器.

问题:
  gptqmodel 2.0.0 的 MODEL_MAP 只注册了 qwen/qwen2/qwen2_moe/qwen2_vl,
  没有 qwen3. `check_and_get_model_type` 会直接 `TypeError: qwen3 isn't supported yet.`,
  导致无法用 gptqmodel 后端量化 Qwen3 模型.

解决:
  Qwen3 架构与 Qwen2 高度一致 (已验证):
    - Qwen3DecoderLayer.__init__(config, layer_idx) 签名同 Qwen2DecoderLayer
    - 子模块完全相同: self_attn.{q,k,v,o}_proj + mlp.{gate,up,down}_proj
    - 层组织结构: model.layers.[N] + model.embed_tokens + model.norm
  因此 Qwen3 可以直接复用 Qwen2GPTQ 的模块结构声明, 仅需把 layer_type
  从 "Qwen2DecoderLayer" 改成 "Qwen3DecoderLayer".

  本模块定义 Qwen3GPTQ (继承 Qwen2GPTQ, 仅覆盖 layer_type), 并把它注入到
  gptqmodel 的 MODEL_MAP / SUPPORTED_MODELS, 让 GPTQModel.from_pretrained
  能识别 qwen3 模型.

  产出的量化格式: 标准 GPTQ (quant_method=gptq, format=gptq),
  vLLM 用 GPTQConfig 加载, get_min_capability()=60, V100 (SM 7.0) 兼容
  (走 Exllama kernel), A100+ 走 Marlin kernel.

使用:
  from qwen3_gptq_adapter import install_qwen3_gptq_adapter
  install_qwen3_gptq_adapter()  # 必须在 GPTQModel.from_pretrained 之前调用

幂等: 已安装则跳过. 不会修改 gptqmodel 源码, 仅运行时注入.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PATCH_INSTALLED = False
_ORIG_MODEL_MAP = None
_ORIG_SUPPORTED_MODELS = None


def _define_qwen3_gptq_class():
    """定义 Qwen3GPTQ 类 (复用 Qwen2GPTQ 的模块结构, 仅改 layer_type).

    放在函数内延迟导入, 避免 gptqmodel 未安装时模块导入失败.
    """
    from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ

    class Qwen3GPTQ(Qwen2GPTQ):
        """Qwen3 GPTQ 量化模型.

        与 Qwen2 架构差异 (已验证):
          - layer_type: Qwen3DecoderLayer (vs Qwen2DecoderLayer)
          - 其余模块结构 (self_attn/mlp/norm) 完全相同
          - Qwen3 特有: position_embeddings 作为独立参数传入 forward,
            但 GPTQ 量化只在权重上做, 不影响 forward 调用, 无需特殊处理
        """
        layer_type = "Qwen3DecoderLayer"

    Qwen3GPTQ.__module__ = "qwen3_gptq_adapter"
    Qwen3GPTQ.__qualname__ = "Qwen3GPTQ"
    return Qwen3GPTQ


def install_qwen3_gptq_adapter() -> bool:
    """把 Qwen3GPTQ 注入 gptqmodel 的 MODEL_MAP / SUPPORTED_MODELS.

    :return: True 表示已安装 (或本次安装成功); False 表示 gptqmodel 未安装.
    """
    global _PATCH_INSTALLED, _ORIG_MODEL_MAP, _ORIG_SUPPORTED_MODELS

    # 幂等检查: 已安装则跳过
    if _PATCH_INSTALLED:
        return True

    try:
        from gptqmodel.models.auto import MODEL_MAP
        # SUPPORTED_MODELS 可能在不同位置, 尝试多个 import 路径
        try:
            from gptqmodel.utils.model import SUPPORTED_MODELS
        except ImportError:
            try:
                from gptqmodel.models.auto import SUPPORTED_MODELS
            except ImportError:
                # 旧版可能用 MODEL_MAP 的 keys 当 supported set
                SUPPORTED_MODELS = None
    except ImportError as e:
        logger.warning(f"[qwen3_gptq_adapter] gptqmodel 未安装, 跳过: {e}")
        return False

    # 备份原始状态 (仅首次)
    if _ORIG_MODEL_MAP is None:
        _ORIG_MODEL_MAP = dict(MODEL_MAP)
    if SUPPORTED_MODELS is not None and _ORIG_SUPPORTED_MODELS is None:
        if isinstance(SUPPORTED_MODELS, list):
            _ORIG_SUPPORTED_MODELS = list(SUPPORTED_MODELS)
        elif isinstance(SUPPORTED_MODELS, set):
            _ORIG_SUPPORTED_MODELS = set(SUPPORTED_MODELS)

    # 注入 Qwen3GPTQ
    Qwen3GPTQ = _define_qwen3_gptq_class()
    MODEL_MAP["qwen3"] = Qwen3GPTQ
    if SUPPORTED_MODELS is not None:
        # SUPPORTED_MODELS 可能是 list 或 set, 兼容处理
        if isinstance(SUPPORTED_MODELS, list):
            if "qwen3" not in SUPPORTED_MODELS:
                SUPPORTED_MODELS.append("qwen3")
        elif isinstance(SUPPORTED_MODELS, set):
            SUPPORTED_MODELS.add("qwen3")

    _PATCH_INSTALLED = True
    logger.info(
        "[qwen3_gptq_adapter] 已注入 Qwen3GPTQ 到 gptqmodel MODEL_MAP "
        "(复用 Qwen2GPTQ 结构, layer_type=Qwen3DecoderLayer). "
        "产出 GPTQ 格式, vLLM GPTQConfig (min_cap=60) 兼容 V100/A100."
    )
    return True


def uninstall_qwen3_gptq_adapter() -> None:
    """还原 gptqmodel MODEL_MAP / SUPPORTED_MODELS (幂等)."""
    global _ORIG_MODEL_MAP, _ORIG_SUPPORTED_MODELS

    try:
        from gptqmodel.models.auto import MODEL_MAP
    except ImportError:
        return

    if "qwen3" in MODEL_MAP and _ORIG_MODEL_MAP is not None:
        # 还原到原始状态 (如果原本没有 qwen3 就删除)
        if "qwen3" not in _ORIG_MODEL_MAP:
            MODEL_MAP.pop("qwen3", None)
        else:
            MODEL_MAP["qwen3"] = _ORIG_MODEL_MAP["qwen3"]

    try:
        from gptqmodel.utils.model import SUPPORTED_MODELS
        if _ORIG_SUPPORTED_MODELS is not None:
            # 还原: 移除新增的 qwen3
            if isinstance(SUPPORTED_MODELS, list):
                if "qwen3" in SUPPORTED_MODELS:
                    SUPPORTED_MODELS.remove("qwen3")
            elif isinstance(SUPPORTED_MODELS, set):
                SUPPORTED_MODELS.discard("qwen3")
    except ImportError:
        pass

    _ORIG_MODEL_MAP = None
    _ORIG_SUPPORTED_MODELS = None


def verify_qwen3_structure(model_path: str) -> Optional[bool]:
    """验证 Qwen3 模型结构与 Qwen3GPTQ 声明是否一致.

    :return: True 一致; False 不一致; None 无法验证 (如模型加载失败).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        import torch

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if cfg.model_type != "qwen3":
            logger.warning(f"[qwen3_gptq_adapter] 模型不是 qwen3: {cfg.model_type}")
            return None

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="cpu",
        )
        layer0 = model.model.layers[0]
        if type(layer0).__name__ != "Qwen3DecoderLayer":
            logger.warning(f"[qwen3_gptq_adapter] layer 0 类型异常: {type(layer0).__name__}")
            return False

        # 验证 layer_modules 声明的所有模块都存在
        expected_modules = [
            "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj",
        ]
        for mod_name in expected_modules:
            obj = layer0
            for part in mod_name.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    logger.warning(f"[qwen3_gptq_adapter] 缺失模块: layer0.{mod_name}")
                    return False
        return True
    except Exception as e:
        logger.warning(f"[qwen3_gptq_adapter] 验证失败: {e}")
        return None
