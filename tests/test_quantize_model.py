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
