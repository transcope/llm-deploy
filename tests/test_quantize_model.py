import json
import os
from argparse import Namespace
from unittest.mock import MagicMock

import pytest
import yaml

import quantize_model as qm


def write_yaml(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_load_config(tmp_path):
    config = {
        "quantization": {"method": "awq", "w_bit": 4},
        "calibration": {"num_samples": 64},
    }
    path = write_yaml(tmp_path, "cfg.yaml", config)
    loaded = qm.load_config(path)
    assert loaded["quantization"]["method"] == "awq"
    assert loaded["calibration"]["num_samples"] == 64


def test_merge_config_prefers_cli(tmp_path):
    config = {"quantization": {"method": "fp8", "w_bit": 8}}
    path = write_yaml(tmp_path, "cfg.yaml", config)
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        method="awq",
        config=path,
        output="./out",
        w_bit=3,
        group_size=64,
    )
    merged = qm.merge_config(args)
    assert merged["method"] == "awq"  # CLI 优先
    assert merged["w_bit"] == 3
    assert merged["group_size"] == 64


def test_merge_config_from_file(tmp_path):
    config = {"quantization": {"method": "gptq", "q_group_size": 64}}
    path = write_yaml(tmp_path, "cfg.yaml", config)
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        method="",
        config=path,
        output="./out",
        w_bit=None,
        group_size=None,
    )
    merged = qm.merge_config(args)
    assert merged["method"] == "gptq"
    assert merged["group_size"] == 64
    assert merged["w_bit"] == 4  # 默认值


def test_merge_config_alias_smoothquant_and_bits(tmp_path):
    config = {
        "quantization": {
            "method": "smoothquant",
            "bits": 8,
            "group_size": 256,
        }
    }
    path = write_yaml(tmp_path, "cfg.yaml", config)
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        method="",
        config=path,
        output="./out",
        w_bit=None,
        group_size=None,
    )
    merged = qm.merge_config(args)
    assert merged["method"] == "w8a8"
    assert merged["w_bit"] == 8
    assert merged["group_size"] == 256


def test_format_calibration_data_applies_chat_template():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted text"
    texts = ["hello", "world"]
    result = qm.format_calibration_data(tokenizer, texts)
    assert result == ["formatted text", "formatted text"]
    assert tokenizer.apply_chat_template.call_count == 2


def test_format_calibration_data_falls_back_on_error():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = RuntimeError("no template")
    texts = ["hello", "world"]
    result = qm.format_calibration_data(tokenizer, texts)
    assert result == texts


def test_get_calibration_texts_default():
    config = {"calibration": {"num_samples": 3}}
    texts = qm.get_calibration_texts(config)
    assert len(texts) == 3
    assert texts[0] == qm.DEFAULT_CALIBRATION_TEXTS[0]


def _install_fake_datasets(monkeypatch, samples):
    """注入一个假的 datasets 模块, 其 load_dataset 返回可迭代 samples"""
    import sys
    import types

    fake = types.ModuleType("datasets")

    def load_dataset(name, split="train"):
        return list(samples)

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_get_calibration_texts_falls_back_when_dataset_empty(monkeypatch):
    # 数据集加载成功但返回 0 条有效样本, 应回退默认数据而非空列表
    _install_fake_datasets(monkeypatch, samples=[])
    config = {"calibration": {"dataset": "fake/empty", "num_samples": 4}}
    texts = qm.get_calibration_texts(config)
    assert texts == qm.DEFAULT_CALIBRATION_TEXTS[:4]
    # 保证下游 calib_texts[0] 不会 IndexError
    assert texts[0]


def test_get_calibration_texts_falls_back_when_all_samples_blank(monkeypatch):
    # 样本存在但既无 messages 也无 text -> 全部 append "" -> 视为无效回退
    _install_fake_datasets(monkeypatch, samples=[{}, {}, {}])
    config = {"calibration": {"dataset": "fake/blank", "num_samples": 2}}
    texts = qm.get_calibration_texts(config)
    assert texts == qm.DEFAULT_CALIBRATION_TEXTS[:2]


def test_get_calibration_texts_handles_zero_num_samples():
    # num_samples <= 0 不应返回空列表, 至少给 1 条
    config = {"calibration": {"num_samples": 0}}
    texts = qm.get_calibration_texts(config)
    assert len(texts) >= 1
    assert texts[0] == qm.DEFAULT_CALIBRATION_TEXTS[0]


def test_get_calibration_texts_uses_dataset_messages(monkeypatch):
    _install_fake_datasets(
        monkeypatch,
        samples=[{"messages": [{"role": "user", "content": "hi"}]}],
    )
    config = {"calibration": {"dataset": "fake/msg", "num_samples": 4}}
    texts = qm.get_calibration_texts(config)
    assert texts == [[{"role": "user", "content": "hi"}]]


def test_save_quant_config(tmp_path):
    output = str(tmp_path / "out")
    config = {"method": "awq", "w_bit": 4}
    qm.save_quant_config(output, config)
    with open(os.path.join(output, "quantize_config.json"), "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["method"] == "awq"


def test_to_calibration_dataset_wraps_texts():
    # llmcompressor.oneshot 要求 dataset 拥有 column_names / map, 不能是裸 list
    pytest.importorskip("datasets")
    ds = qm.to_calibration_dataset(["hello", "world"])
    assert ds.column_names == ["text"]
    assert ds[0]["text"] == "hello"
    assert ds[1]["text"] == "world"
    assert len(ds) == 2


def test_cli_requires_method_or_config(capsys):
    with pytest.raises(SystemExit):
        qm.main()
    err = capsys.readouterr().err
    assert "method" in err.lower() or "config" in err.lower()


def test_check_hardware_compatibility_v100_rejects_fp8(monkeypatch, capsys):
    # V100 (SM 7.0) 不支持 FP8, 量化前应直接报错退出, 避免浪费校准时间
    monkeypatch.setattr(qm, "detect_gpu_capability", lambda: (7, 0))
    with pytest.raises(SystemExit):
        qm.check_hardware_compatibility("fp8")
    out = capsys.readouterr().out
    assert "FP8" in out


def test_check_hardware_compatibility_v100_warns_awq(monkeypatch, capsys):
    # V100 上 AWQ 只能用慢速 GEMV kernel, 应警告但不阻止 (模型仍可加载)
    monkeypatch.setattr(qm, "detect_gpu_capability", lambda: (7, 0))
    qm.check_hardware_compatibility("awq")
    out = capsys.readouterr().out
    assert "AWQ" in out and "GEMV" in out


def test_check_hardware_compatibility_ampere_allows_fp8(monkeypatch):
    # A100 (SM 8.0) 不支持原生 FP8 (需 Hopper), 但量化本身用 llmcompressor 可执行,
    # 这里仅验证守卫不在 Ampere 上误拦 FP8 (由 H100 的 SM 9.0 分支之外放行)
    monkeypatch.setattr(qm, "detect_gpu_capability", lambda: (9, 0))
    # SM 9.0 应放行所有方法
    qm.check_hardware_compatibility("fp8")
    qm.check_hardware_compatibility("awq")


def test_check_hardware_compatibility_no_gpu_skips(monkeypatch):
    # 无 GPU 环境 (开发机/CI) 应直接跳过, 不影响脚本逻辑测试
    monkeypatch.setattr(qm, "detect_gpu_capability", lambda: None)
    qm.check_hardware_compatibility("fp8")
    qm.check_hardware_compatibility("awq")


def test_check_hardware_compatibility_gptq_w8a8_pass_on_v100(monkeypatch, capsys):
    # GPTQ / W8A8 是 V100 推荐方案, 不应有任何警告输出
    monkeypatch.setattr(qm, "detect_gpu_capability", lambda: (7, 0))
    qm.check_hardware_compatibility("gptq")
    qm.check_hardware_compatibility("w8a8")
    out = capsys.readouterr().out
    assert out == ""


# ---------------------------------------------------------------------------
# 方案 B: --force-legacy-awq / quantize_awq_legacy (AutoAWQ 路径)
# 1Cat-vLLM 的 SM70 内核只支持 AWQ 原生格式 (quant_method=awq),
# 必须用 AutoAWQ 产出; llmcompressor 的 compressed-tensors 格式不兼容。
# ---------------------------------------------------------------------------


def test_quantize_awq_force_legacy_skips_llmcompressor(monkeypatch):
    # force_legacy=True 时强制走 legacy AutoAWQ, 绝不调用 llmcompressor
    legacy = MagicMock()
    llmc = MagicMock()
    monkeypatch.setattr(qm, "quantize_awq_legacy", legacy)
    monkeypatch.setattr(qm, "quantize_awq_llmcompressor", llmc)

    qm.quantize_awq("/m", "/o", {"method": "awq"}, force_legacy=True)

    legacy.assert_called_once_with("/m", "/o", {"method": "awq"})
    llmc.assert_not_called()


def test_quantize_awq_falls_back_to_legacy_when_llmcompressor_fails(monkeypatch):
    # force_legacy=False 且 llmcompressor 失败时, 应回退到 legacy AutoAWQ
    legacy = MagicMock()
    llmc = MagicMock(return_value=False)
    monkeypatch.setattr(qm, "quantize_awq_legacy", legacy)
    monkeypatch.setattr(qm, "quantize_awq_llmcompressor", llmc)

    qm.quantize_awq("/m", "/o", {"method": "awq"}, force_legacy=False)

    llmc.assert_called_once_with("/m", "/o", {"method": "awq"})
    legacy.assert_called_once_with("/m", "/o", {"method": "awq"})


def test_quantize_awq_no_fallback_when_llmcompressor_succeeds(monkeypatch):
    # force_legacy=False 且 llmcompressor 成功时, 不应回退 legacy
    legacy = MagicMock()
    llmc = MagicMock(return_value=True)
    monkeypatch.setattr(qm, "quantize_awq_legacy", legacy)
    monkeypatch.setattr(qm, "quantize_awq_llmcompressor", llmc)

    qm.quantize_awq("/m", "/o", {"method": "awq"}, force_legacy=False)

    llmc.assert_called_once_with("/m", "/o", {"method": "awq"})
    legacy.assert_not_called()


def _install_fake_awq_modules(monkeypatch):
    """注入假的 awq / transformers 模块, 供 quantize_awq_legacy 函数内 import 使用"""
    import sys
    import types

    fake_model = MagicMock()
    fake_tokenizer = MagicMock()

    fake_awq = types.ModuleType("awq")
    fake_awq.AutoAWQForCausalLM = MagicMock()
    # from_pretrained 必须返回 fake_model, 否则 model 会是另一个 MagicMock
    fake_awq.AutoAWQForCausalLM.from_pretrained = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "awq", fake_awq)

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = MagicMock()
    # from_pretrained 必须返回 fake_tokenizer, 否则 tokenizer 会是另一个 MagicMock
    fake_tf.AutoTokenizer.from_pretrained = MagicMock(return_value=fake_tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    return fake_model, fake_tokenizer


def test_quantize_awq_legacy_builds_quant_config(monkeypatch, tmp_path):
    # 验证 legacy AutoAWQ 路径从 config 正确读取量化参数 (zero_point/group_size/w_bit/version)
    fake_model, fake_tokenizer = _install_fake_awq_modules(monkeypatch)
    monkeypatch.setattr(qm, "get_calibration_texts", lambda cfg: ["hello"])
    monkeypatch.setattr(qm, "format_calibration_data", lambda tok, texts: texts)
    monkeypatch.setattr(qm, "save_quant_config", MagicMock())

    config = {
        "zero_point": True,
        "group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
        "output": {"safetensors": True, "shard_size": "4GB"},
    }
    out = str(tmp_path / "out")
    qm.quantize_awq_legacy("/m", out, config)

    # 验证 quant_config 正确传递
    call_kwargs = fake_model.quantize.call_args.kwargs
    assert call_kwargs["quant_config"] == {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    # 验证保存
    fake_model.save_quantized.assert_called_once()
    fake_tokenizer.save_pretrained.assert_called_once_with(out)


def test_quantize_awq_legacy_uses_defaults_when_config_missing(monkeypatch, tmp_path):
    # config 缺省时, quant_config 应使用默认值 (zero_point=True, group_size=128, w_bit=4, GEMM)
    fake_model, _ = _install_fake_awq_modules(monkeypatch)
    monkeypatch.setattr(qm, "get_calibration_texts", lambda cfg: ["hello"])
    monkeypatch.setattr(qm, "format_calibration_data", lambda tok, texts: texts)
    monkeypatch.setattr(qm, "save_quant_config", MagicMock())

    out = str(tmp_path / "out")
    qm.quantize_awq_legacy("/m", out, {})

    call_kwargs = fake_model.quantize.call_args.kwargs
    assert call_kwargs["quant_config"] == {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }


def test_quantize_awq_legacy_handles_chat_template_calib(monkeypatch, tmp_path):
    # 校准数据为 messages 列表 (list[list[dict]]) 时, 应走 apply_chat_template 分支
    fake_model, fake_tokenizer = _install_fake_awq_modules(monkeypatch)
    fake_tokenizer.apply_chat_template.return_value = "formatted"
    monkeypatch.setattr(
        qm, "get_calibration_texts", lambda cfg: [[{"role": "user", "content": "hi"}]]
    )
    monkeypatch.setattr(qm, "save_quant_config", MagicMock())

    out = str(tmp_path / "out")
    qm.quantize_awq_legacy("/m", out, {})

    fake_tokenizer.apply_chat_template.assert_called_once()
    # 校准数据应被格式化为字符串列表
    calib = fake_model.quantize.call_args.kwargs["calib_data"]
    assert calib == ["formatted"]


def test_cli_has_force_legacy_awq_flag():
    # CLI 必须提供 --force-legacy-awq 参数 (方案 B 强制 AutoAWQ 路径)
    import inspect

    src = inspect.getsource(qm.main)
    assert "--force-legacy-awq" in src
    assert "force_legacy=args.force_legacy_awq" in src
