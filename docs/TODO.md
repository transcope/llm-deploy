# 1Cat-vLLM + AWQ 方案固化 TODO

> 本文件记录 1Cat-vLLM + AWQ 方案的固化任务清单。
> **这部分内容固化不动**，后续提交 GitHub 更新时保持稳定。

## 状态：✅ 已完成（待提交 GitHub）

1Cat-vLLM + AWQ 方案已在 V100 实测跑通，相关配置、环境、脚本、文档已全部整理固化。

## 方案要点（固化）

- **推理引擎**：1Cat-vLLM v1.0.0（V100/SM70 专用，`FLASH_ATTN_V100` 后端）
- **量化方式**：AutoAWQ（产出 **AWQ 原生格式** `quant_method=awq`）
- **评测方式**：`benchmark_domain.py` 领域精度评测（API 模式，禁用 thinking）

## 关键结论（固化，勿改）

| # | 结论 |
|---|------|
| 1 | **必须用 AutoAWQ 量化**，产出 AWQ 原生格式（`quant_method=awq`）。**不要用 llmcompressor**（产出 compressed-tensors 格式，1Cat-vLLM 无法加载） |
| 2 | **必须禁用 prefix caching / chunked prefill**（`--no-enable-prefix-caching --no-enable-chunked-prefill`），否则长序列评测触发共享内存错误 |
| 3 | **评测需禁用 thinking**（`--no-thinking`），避免 thinking 耗尽 max_tokens |
| 4 | **V100 不支持 bfloat16**，FP16 基线需转 float16（转换慢） |
| 5 | **环境隔离**：量化（AutoAWQ）与推理（1Cat-vLLM）分两个虚拟环境 |

## 文件清单（固化）

### 配置
- `configs/awq_4bit_v100.yaml` — AWQ 量化配置（AutoAWQ 路径参数）

### 环境
- `cases/v100/awq_1cat/install_env.sh` — 1Cat-vLLM 环境安装（`1cat-venv`）
- `cases/v100/awq_1cat/quantize.sh` — 量化环境（`venv-quant-awq`，autoawq）

### 脚本（cases/v100/awq_1cat/）
- `quantize.sh` — AWQ 量化（AutoAWQ，`--force-legacy-awq`）
- `serve.sh` — 启动 1Cat-vLLM 服务（禁用 prefix caching/chunked prefill）
- `benchmark.sh` — 领域精度评测（禁用 thinking）
- `deploy_all.sh` — 端到端一键脚本（env/quantize/serve/test/eval/perf）

### 代码
- `llm_deploy/quantize_model.py` — 新增 `--force-legacy-awq` 参数（强制 AutoAWQ 路径）

### 文档
- `docs/V100_1CAT_GUIDE.md` — 方案 B 专版文档（含实测修正、基线说明）
- `cases/v100/awq_1cat/README.md` — 方案总览（实测结论、快速开始、脚本说明、评测结果）
- `bak/1cat_awq_feedback.md` — 历史反馈记录（结论已被实测推翻，已归档）
- `docs/TODO.md` — 本文件

## 可复现流程（固化）

```bash
# 1. 安装 1Cat-vLLM 环境
bash cases/v100/awq_1cat/install_env.sh

# 2. AutoAWQ 量化 (AWQ 原生格式)
bash cases/v100/awq_1cat/quantize.sh

# 3. 启动服务 (禁用 prefix caching/chunked prefill)
bash cases/v100/awq_1cat/serve.sh

# 4. 领域精度评测 (禁用 thinking)
bash cases/v100/awq_1cat/benchmark.sh
```

## 实测评测结果（固化）

AWQ-AutoAWQ 模型（`Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ`）领域精度评测：

| 来源 | 准确率 | 平均得分 | 通过数 |
|------|--------|---------|--------|
| **math** | **88.89%** | 0.8889 | 24/27 |
| **codegen** | **80.00%** | 0.4029 | 8/10 |
| **tasks** | 50.00% | 0.4167 | 3/6 |
| **alpaca** | 36.04% | 0.3266 | 40/111 |
| **messages** | 0.00% | 0.0734 | 0/21 |
| **总体** | **42.86%** | 0.3904 | 75/175 |

## FP16 基线评测（✅ 已完成）

> **关键方案**：V100 不支持 bfloat16，FP16 原模型（bfloat16）需转 float16。
> 单卡加载转换极慢（约 3.6h），**用 8 卡 V100 张量并行（`-tp 8`）解决**，加载仅 129s。

```bash
source /app/1cat-venv/bin/activate
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100
python -m vllm.entrypoints.openai.api_server \
    --model /volume/models/Mind-SLLM-Qwen3-8B \
    --dtype float16 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --port 8000 --trust-remote-code \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --enforce-eager
```

### 精度对比（FP16 基线 vs AWQ 量化）

| 来源 | FP16 基线 | AWQ 量化 | 差异 |
|------|----------|---------|------|
| **math** | 81.48% (22/27) | 88.89% (24/27) | +7.41% |
| **codegen** | 70.00% (7/10) | 80.00% (8/10) | +10% |
| **tasks** | 33.33% (2/6) | 50.00% (3/6) | +16.67% |
| **alpaca** | 36.04% (40/111) | 36.04% (40/111) | 0% |
| **messages** | 0.00% (0/21) | 0.00% (0/21) | 0% |
| **总体** | **40.57%** (71/175) | **42.86%** (75/175) | **+2.29%** |

> **结论**：AWQ 量化模型精度与 FP16 基线相当（甚至略高），**无精度损失**。
> math/codegen 领域 AWQ 表现更优；alpaca/messages 低分是评测数据特点（开放式长答案关键词匹配难），非量化问题。

## 待办（未完成，不固化）

- [x] 验证快速部署脚本 `cases/v100/awq_1cat/serve.sh`（2026-08-15 实测通过：`/v1/models` 返回模型、`/v1/chat/completions` 推理正常）
- [x] 提交 GitHub 更新
