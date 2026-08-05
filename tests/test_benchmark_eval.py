import json
from unittest.mock import MagicMock, patch

import pytest

import benchmark_eval as be


def test_extract_scores():
    data = {
        "results": {
            "gsm8k": {"acc": 0.75, "acc_stderr": 0.01, "alias": "gsm8k"},
            "hellaswag": {"acc": 0.60, "acc_norm": 0.65},
        }
    }
    scores = be.extract_scores(data)
    assert scores["gsm8k_acc"] == 0.75
    assert scores["hellaswag_acc_norm"] == 0.65
    assert "gsm8k_alias" not in scores


def test_build_model_args_with_quantization():
    args = be.build_model_args(
        model_path="./models/Qwen2.5-7B-AWQ",
        quantization="awq",
        max_model_len=4096,
    )
    assert "pretrained=./models/Qwen2.5-7B-AWQ" in args
    assert "quantization=awq" in args
    assert "add_bos_token=true" in args


def test_build_model_args_without_quantization():
    args = be.build_model_args(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        max_model_len=8192,
    )
    assert "quantization=" not in args
    assert "max_model_len=8192" in args


def test_build_model_args_dtype_and_tensor_parallel():
    args = be.build_model_args(
        model_path="Qwen/Qwen2.5-32B-Instruct",
        max_model_len=8192,
        dtype="float16",
        tensor_parallel=2,
    )
    assert "dtype=float16" in args
    assert "tensor_parallel_size=2" in args


def test_build_model_args_single_gpu_no_tp():
    args = be.build_model_args(model_path="m", max_model_len=4096)
    assert "tensor_parallel_size" not in args


@patch("benchmark_eval.subprocess.run")
def test_run_lm_eval_command_construction(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="done", stderr="")
    output_dir = str(tmp_path / "eval")
    be.run_lm_eval(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        tasks=["gsm8k"],
        output_dir=output_dir,
        quantization="awq",
        max_model_len=4096,
    )
    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert cmd[0].endswith("python") or cmd[0].endswith(sys_executable_name())
    assert "lm_eval" in cmd
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "add_bos_token=true" in model_args
    assert "quantization=awq" in model_args


def sys_executable_name():
    import sys
    return sys.executable.split("/")[-1]


@patch("benchmark_eval.requests.post")
def test_send_stream_request_success(mock_post):
    fake_chunk = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}],"usage":null}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}],"usage":{"completion_tokens":2}}\n\n'
        b'data: [DONE]\n\n'
    )
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = fake_chunk.split(b"\n\n")
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = be.send_stream_request("http://localhost:8000", "hi", "m", 10)
    assert "error" not in result
    assert result["tokens"] == 2
    assert result["ttft"] >= 0


@patch("benchmark_eval.requests.post")
def test_send_stream_request_error(mock_post):
    mock_post.side_effect = Exception("connection refused")
    result = be.send_stream_request("http://localhost:8000", "hi", "m", 10)
    assert "error" in result


@patch("benchmark_eval.requests.get")
@patch("benchmark_eval.requests.post")
def test_run_perf_test(mock_post, mock_get):
    # /v1/models
    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"data": [{"id": "test-model"}]}),
    )

    # streaming response for each prompt
    def make_response(*args, **kwargs):
        resp = MagicMock()
        resp.iter_lines.return_value = [
            b'data: {"choices":[{"delta":{"content":"OK"}}],"usage":{"completion_tokens":1}}',
            b'data: [DONE]',
        ]
        resp.raise_for_status.return_value = None
        return resp

    mock_post.side_effect = make_response

    result = be.run_perf_test(
        model_path="test-model",
        base_url="http://localhost:8000",
        num_prompts=5,
        max_tokens=10,
        concurrency=2,
    )
    assert result["total_requests"] == 5
    assert result["successful_requests"] == 5
    assert result["total_tokens_generated"] == 5
    assert result["ttft_avg_seconds"] >= 0
