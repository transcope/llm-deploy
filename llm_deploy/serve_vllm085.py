#!/usr/bin/env python3
"""
基于 vLLM 0.8.5 的 Qwen3 GPTQ 部署服务 (OpenAI 兼容 API, V100 专用)

用 vLLM 0.8.5 的 LLM() 直接加载模型 (避免 `vllm serve` 的 multiprocessing
内存 profiling bug), 提供 /v1/models、/v1/chat/completions、/health 接口.

V100 需要: XFORMERS attention backend + enforce_eager + V0 engine.

用法:
    # 部署 GPTQ 量化模型 (V100 生产推荐)
    python llm_deploy/serve_vllm085.py \
        --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
        --quantization gptq --port 8000 --gpu 0

    # 部署 FP16 原模型
    python llm_deploy/serve_vllm085.py \
        --model /app/local_models/Mind-SLLM-Qwen3-8B --port 8000 --gpu 0
"""
import argparse
import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

# 全局
LLM = None
TOKENIZER = None
MODEL_NAME = "Mind-SLLM-Qwen3-8B-GPTQ"
LOCK = threading.Lock()


def _read_model_name(model_path: str) -> str:
    """从模型 config.json 读取模型名, 失败时回退到目录名"""
    try:
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        name = cfg.get("_name_or_path") or cfg.get("model_type")
        if name:
            return name
    except Exception:
        pass
    return os.path.basename(model_path.rstrip("/"))


def load_model(model_path, quantization, gpu, gpu_util, max_model_len):
    global LLM, TOKENIZER, MODEL_NAME
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from vllm import LLM
    from transformers import AutoTokenizer
    print(f"加载 vLLM 0.8.5 模型: {model_path}")
    print(f"  quantization={quantization}, gpu={gpu}, gpu_util={gpu_util}, max_model_len={max_model_len}")
    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    LLM = LLM(**llm_kwargs)
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MODEL_NAME = _read_model_name(model_path)
    print("模型加载完成")


def generate(messages, max_tokens=512, temperature=0.6, top_p=0.95):
    from vllm import SamplingParams
    # 强制至少 256 tokens, 避免 Qwen3 thinking 阶段被截断产生退化输出
    max_tokens = max(int(max_tokens), 256)
    sampling_params = SamplingParams(
        temperature=temperature if temperature and temperature > 0 else 0.0,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # vLLM 0.8.5 V0 引擎 bug: 单条/无 system 消息的 prompt 会产生退化输出 "!!!!"
    # 用一条带 system 消息的不同 dummy prompt 填充到 2 条, 取第一条结果
    sys_msg = {"role": "system", "content": "你是通信领域专家，请准确回答以下问题。"}
    if not any(m.get("role") == "system" for m in messages):
        messages = [sys_msg] + list(messages)
    dummy = [sys_msg, {"role": "user", "content": "请介绍一下通信中的调制方式。"}]
    with LOCK:
        outputs = LLM.chat([messages, dummy], sampling_params)
    return outputs[0].outputs[0].text.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }],
            })
        elif path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self._send_json({"error": f"bad request: {e}"}, 400)
            return

        if path == "/v1/chat/completions":
            messages = body.get("messages", [])
            max_new = body.get("max_tokens", 512)
            temperature = body.get("temperature", 0.6)
            top_p = body.get("top_p", 0.95)
        else:
            prompt = body.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]
            max_new = body.get("max_tokens", 512)
            temperature = body.get("temperature", 0.6)
            top_p = body.get("top_p", 0.95)

        try:
            start = time.time()
            text = generate(messages, max_new, temperature, top_p)
            elapsed = time.time() - start
            resp = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(TOKENIZER.encode(messages[-1]["content"])),
                    "completion_tokens": len(TOKENIZER.encode(text)),
                    "total_tokens": len(TOKENIZER.encode(messages[-1]["content"])) + len(TOKENIZER.encode(text)),
                },
                "timing": {"elapsed_sec": round(elapsed, 2)},
            }
            self._send_json(resp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser(description="vLLM 0.8.5 部署服务 (V100 专用)")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--quantization", default=None, help="量化方式 (gptq)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0, help="使用的 GPU 编号")
    parser.add_argument("--gpu-util", type=float, default=0.9, help="GPU 内存利用率")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大序列长度")
    args = parser.parse_args()

    load_model(args.model, args.quantization, args.gpu, args.gpu_util, args.max_model_len)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"服务已启动: http://{args.host}:{args.port}")
    print(f"模型: {MODEL_NAME}")
    print(f"测试: curl http://{args.host}:{args.port}/v1/chat/completions "
          f"-H 'Content-Type: application/json' -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"你好\"}}]}}'")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("服务停止")


if __name__ == "__main__":
    main()
