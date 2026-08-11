"""基于 gptqmodel + TORCH backend 的 GPTQ 模型部署服务 (OpenAI 兼容 API)

V100 上 vLLM 0.7.1 不支持 Qwen3, 故用 gptqmodel + TORCH backend 部署.
提供 /v1/models 和 /v1/chat/completions 接口.
"""
import sys
import json
import time
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, "/volume/workspace/llm-deploy/src")

import torch
from qwen3_gptq_adapter import install_qwen3_gptq_adapter
from gptqmodel import GPTQModel
from gptqmodel.utils.backend import BACKEND
from transformers import AutoTokenizer

# 全局模型和 tokenizer
MODEL = None
TOKENIZER = None
MODEL_NAME = "Mind-SLLM-Qwen3-8B-GPTQ"
MAX_NEW_TOKENS = 512
LOCK = threading.Lock()


def load_model(model_path):
    global MODEL, TOKENIZER
    print(f"加载 gptqmodel 模型: {model_path}")
    install_qwen3_gptq_adapter()
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MODEL = GPTQModel.from_quantized(
        model_path, device="cuda:0", backend=BACKEND.TORCH,
    )
    MODEL.eval()
    print("模型加载完成")


def generate(prompt, max_new_tokens=512, temperature=0.6, top_p=0.95):
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=8192)
    input_ids = inputs.input_ids.to("cuda")
    with LOCK:
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=TOKENIZER.eos_token_id,
            )
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(do_sample=False)
            outputs = MODEL.generate(input_ids, **gen_kwargs)
    generated = outputs[0][input_ids.shape[1]:]
    return TOKENIZER.decode(generated, skip_special_tokens=True).strip()


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
            max_new = body.get("max_tokens", MAX_NEW_TOKENS)
            temperature = body.get("temperature", 0.6)
            top_p = body.get("top_p", 0.95)
            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = body.get("prompt", "")
            max_new = body.get("max_tokens", MAX_NEW_TOKENS)
            temperature = body.get("temperature", 0.6)
            top_p = body.get("top_p", 0.95)

        try:
            start = time.time()
            text = generate(prompt, max_new, temperature, top_p)
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
                    "prompt_tokens": len(TOKENIZER.encode(prompt)),
                    "completion_tokens": len(TOKENIZER.encode(text)),
                    "total_tokens": len(TOKENIZER.encode(prompt)) + len(TOKENIZER.encode(text)),
                },
                "timing": {"elapsed_sec": round(elapsed, 2)},
            }
            self._send_json(resp)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="量化模型路径")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"服务已启动: http://{args.host}:{args.port}")
    print(f"模型: {MODEL_NAME}")
    print(f"测试: curl http://{args.host}:{args.port}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"你好\"}}]}}'")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("服务停止")


if __name__ == "__main__":
    main()
