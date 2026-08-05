import os
from argparse import Namespace

import pytest
import yaml

import deploy_server as ds


def write_yaml(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_load_config(tmp_path):
    data = {"gpu-memory-utilization": 0.85, "enable-prefix-caching": True}
    path = write_yaml(tmp_path, "serve.yaml", data)
    cfg = ds.load_config(path)
    assert cfg["gpu-memory-utilization"] == 0.85
    assert cfg["enable-prefix-caching"] is True


def test_config_to_defaults():
    cfg = {"gpu-memory-utilization": 0.85, "max-model-len": 4096}
    defaults = ds.config_to_defaults(cfg)
    assert defaults["gpu_util"] == 0.85
    assert defaults["max_model_len"] == 4096


def test_build_vllm_command_basic():
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="",
        quantization="",
        kv_dtype="",
        gpu_util=0.9,
        max_model_len=32768,
        tensor_parallel=1,
        pipeline_parallel=1,
        enable_expert_parallel=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        multimodal=False,
        max_images=5,
        trust_remote_code=True,
        tool_call_parser="",
        swap_space=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        api_key="",
        host="0.0.0.0",
        port=8000,
    )
    cmd = ds.build_vllm_command(args)
    assert "--model" in cmd
    assert "Qwen/Qwen2.5-7B-Instruct" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--trust-remote-code" in cmd


def test_build_vllm_command_awq_forces_float16():
    args = Namespace(
        model="./models/Qwen2.5-7B-AWQ",
        dtype="",
        quantization="awq",
        kv_dtype="",
        gpu_util=0.9,
        max_model_len=None,
        tensor_parallel=1,
        pipeline_parallel=1,
        enable_expert_parallel=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        multimodal=False,
        max_images=5,
        trust_remote_code=False,
        tool_call_parser="",
        swap_space=None,
        max_num_batched_tokens=None,
        max_num_seqs=None,
        api_key="",
        host="0.0.0.0",
        port=8000,
    )
    cmd = ds.build_vllm_command(args)
    idx = cmd.index("--quantization")
    assert cmd[idx + 1] == "awq"
    assert "--dtype" in cmd
    assert cmd[cmd.index("--dtype") + 1] == "float16"


def test_get_model_specific_args_qwen7b():
    cfg = ds.get_model_specific_args("Qwen/Qwen2.5-7B-Instruct")
    assert cfg["max_model_len"] == 32768
    assert cfg["trust_remote_code"] is True


def test_get_model_specific_args_deepseek_v3():
    cfg = ds.get_model_specific_args("deepseek-ai/DeepSeek-V3")
    assert cfg["tensor_parallel"] == 8
    assert cfg["pipeline_parallel"] == 2
    assert cfg["enable_expert_parallel"] is True
    assert cfg["quantization"] == "fp8"


def test_apply_config(tmp_path):
    cfg = {"gpu-memory-utilization": 0.85, "swap-space": 32}
    path = write_yaml(tmp_path, "serve.yaml", cfg)
    config = ds.load_config(path)
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        gpu_util=0.9,
        swap_space=None,
        _model_presets={},
    )
    ds.apply_config(args, config)
    # CLI 已传 gpu_util=0.9，不应被配置覆盖
    assert args.gpu_util == 0.9
    # swap_space 未传，应使用配置
    assert args.swap_space == 32


def test_apply_model_specific_args():
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=None,
        trust_remote_code=False,
        tensor_parallel=1,
        _model_presets={},
    )
    ds.apply_model_specific_args(args)
    assert args.max_model_len == 32768
    assert args.trust_remote_code is True


def test_detect_model_quantization(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"quantization_config": {"quant_method": "gptq"}}', encoding="utf-8"
    )
    assert ds.detect_model_quantization(str(model_dir)) == "gptq"


def test_detect_model_quantization_no_config(tmp_path):
    assert ds.detect_model_quantization(str(tmp_path)) is None
    # HuggingFace 模型 ID (非本地路径) 返回 None
    assert ds.detect_model_quantization("Qwen/Qwen2.5-7B-Instruct") is None


def test_apply_hardware_constraints_v100_forces_float16(monkeypatch):
    monkeypatch.setattr(ds, "detect_gpu_capability", lambda: (7, 0))
    args = Namespace(dtype="bfloat16", quantization="", kv_dtype="")
    ds.apply_hardware_constraints(args)
    assert args.dtype == "float16"


def test_apply_hardware_constraints_v100_rejects_fp8(monkeypatch):
    monkeypatch.setattr(ds, "detect_gpu_capability", lambda: (7, 0))
    args = Namespace(dtype="", quantization="fp8", kv_dtype="")
    with pytest.raises(SystemExit):
        ds.apply_hardware_constraints(args)


def test_apply_hardware_constraints_ampere_keeps_dtype(monkeypatch):
    monkeypatch.setattr(ds, "detect_gpu_capability", lambda: (8, 0))
    args = Namespace(dtype="", quantization="", kv_dtype="")
    ds.apply_hardware_constraints(args)
    assert args.dtype == ""


def test_apply_hardware_constraints_no_gpu(monkeypatch):
    monkeypatch.setattr(ds, "detect_gpu_capability", lambda: None)
    args = Namespace(dtype="", quantization="", kv_dtype="")
    ds.apply_hardware_constraints(args)  # 无 GPU 环境直接跳过, 不应报错
    assert args.dtype == ""
