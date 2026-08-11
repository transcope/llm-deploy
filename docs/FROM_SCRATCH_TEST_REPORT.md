# 从零执行测试报告 —— 压缩 → 部署 → 评估全链路

> **测试时间**：2026-08-10 ~ 2026-08-11
> **测试环境**：V100 服务器 `zetta_ld` 容器（8× Tesla V100S-PCIE-32GB, SM 7.0, CUDA 12.6）
> **测试目标**：清空项目目录 + 双 venv 后，仅凭项目文档和脚本从零完成「量化压缩 → 精度评估 → 快速部署」全链路
> **模型**：Mind-SLLM-Qwen3-8B（保留原始模型 + HF 缓存，清空项目代码 + 双 venv）

---

## 1. 执行概览

| 步骤 | 状态 | 耗时 | 说明 |
|------|:---:|:---:|------|
| 1. 登录服务器 + 进入容器 | ✅ | <1 min | SSH + docker exec |
| 2. 恢复项目代码 | ✅ | ~5 min | tar 打包 + scp + docker cp 解包 |
| 3. 重建双虚拟环境 | ✅ | ~2.5h | venv-quant + venv-deploy（含多次依赖冲突修复） |
| 4. 准备校准数据 | ✅ | ~5 min | v1(256条) → v2(230条) |
| 5. 执行 GPTQ 量化 | ✅ | ~30 min | gptqmodel 后端，36 层 |
| 6. 精度评测对比 | ✅ | ~50 min | 原模型 vs 量化模型（50 条领域样本） |
| 7. 快速部署 | ✅ | ~5 min | gptqmodel + TORCH backend，OpenAI 兼容 API |
| 8. 生成报告 | ✅ | — | 本文档 |

**总体结论**：从零执行全链路**可行**，但过程中遇到 **7 个问题**（详见第 5 节），其中 3 个为阻塞性问题需先解决。

---

## 2. 环境信息

### 2.1 硬件与系统
- **GPU**：8× Tesla V100S-PCIE-32GB（SM 7.0）
- **CUDA**：容器内 CUDA 12.6 运行时
- **Python**：3.12.3

### 2.2 双虚拟环境（重建后）
| 环境 | 路径 | 用途 | 关键依赖 |
|------|------|------|----------|
| venv-quant | `/app/venv-quant` | 量化 | torch 2.5.1+cu124, gptqmodel 2.0.0, llmcompressor 0.4.0 |
| venv-deploy | `/app/venv-deploy` | 部署+评测 | torch 2.5.1+cu124, vllm 0.7.1, lm_eval 0.4.12 |

### 2.3 关键路径
| 项 | 路径 |
|----|------|
| 原始模型 | `/app/local_models/Mind-SLLM-Qwen3-8B`（16G） |
| 量化模型 | `/volume/models/Mind-SLLM-Qwen3-8B-GPTQ`（5.8G） |
| 校准数据 v1 | `data/calibration/calibration_data.jsonl`（256 条） |
| 校准数据 v2 | `data/calibration/calibration_data_v2.jsonl`（230 条） |
| 领域 Benchmark | `data/evaluation/accuracy_benchmark.jsonl`（86 条） |
| 精度对比报告 | `results/compare/compare_report.json` |

---

## 3. 量化结果

### 3.1 量化配置
- **方法**：GPTQ（gptqmodel 后端）
- **配置**：`configs/gptq_4bit_v100_gptqmodel.yaml`
- **量化格式**：W4A16（4-bit 权重，16-bit 激活）
- **group_size**：128，desc_act=true，sym=true
- **校准数据**：`calibration_data_v2.jsonl`（230 条，过滤超长序列）

### 3.2 压缩效果
| 指标 | 原模型 (FP16) | 量化模型 (GPTQ 4-bit) | 变化 |
|------|:---:|:---:|:---:|
| 模型大小 | 16G | 5.8G | **-64%** |
| 压缩比 | 1x | **2.76x** | — |

### 3.3 量化耗时
- 36 层全部量化，约 **30 分钟**（V100 单卡）
- 每层约 46 秒

---

## 4. 精度评估结果

### 4.1 对比方法
- **原模型**：transformers 加载（FP16）
- **量化模型**：gptqmodel + TORCH backend 加载
- **数据**：领域 Benchmark 50 条（alpaca/codegen/math/messages/tasks）
- **评分**：关键词匹配（score ≥ 0.35 记为通过）

### 4.2 对比结果
| 指标 | 原模型 (FP16) | 量化模型 (GPTQ 4-bit) | 精度差 |
|------|:---:|:---:|:---:|
| **准确率** | **0.4200** | **0.3800** | **-0.0400** |
| 正确数 | 21/50 | 19/50 | -2 |
| 评测耗时 | 323.5s | 2294.4s | — |

### 4.3 精度损失分析
- **精度损失 -4.0%**，略高于文档预期的 -1.07%
- 可能原因：
  1. 样本量小（50 条，统计波动大）
  2. TORCH backend 生成质量略低于 Exllama（V100 无法用 ExllamaV2）
  3. 关键词匹配评分的敏感性

---

## 5. 问题记录

### 5.1 阻塞性问题（需先解决才能继续）

| # | 问题 | 影响 | 解决方案 |
|:-:|------|------|----------|
| **P1** | **requirements 快照依赖冲突**：`requirements-quant.txt` 锁定 torch 2.6.0(CPU)/vllm 0.8.3/compressed-tensors 0.9.2/numpy 2.1.3，与 llmcompressor 0.4.0（要 numpy<2.0 + compressed-tensors 0.9.0）冲突 | 无法安装依赖 | 按 Dockerfile 权威版本修正：torch 2.5.1/vllm 0.7.1/compressed-tensors 0.9.0/numpy 1.26.4 |
| **P2** | **torch 从 PyPI 默认源装成 CPU 版**：`torch==2.5.1` 无 `+cu124` 后缀时 pip 装 CPU 版，无 CUDA 支持 | 量化无法进行 | 必须从 PyTorch cu124 索引安装，或直接下载 CUDA wheel |
| **P3** | **vLLM 0.7.1 不支持 Qwen3 架构**：`Model architectures ['Qwen3ForCausalLM'] are not supported` | 无法用 vLLM 部署/评测 | 改用 gptqmodel + TORCH backend 部署/评测 |

### 5.2 非阻塞性问题（有替代方案）

| # | 问题 | 影响 | 解决方案 |
|:-:|------|------|----------|
| **P4** | **transformers 4.51.0 无法加载 gptqmodel 2.0.0 的 GPTQ 模型**：要求 gptqmodel>=7.0.0 | 无法用 transformers 加载量化模型 | 用 gptqmodel 的 `from_quantized` 加载 |
| **P5** | **gptqmodel 2.0.0 的 from_quantized 不识别 qwen3**：`qwen3 isn't supported yet` | 无法加载量化模型 | 加载前调用 `install_qwen3_gptq_adapter()` |
| **P6** | **ExllamaV2 kernel 不支持 V100**：`no kernel image is available`（ExllamaV2 需 SM 8.0+） | 自动选 ExllamaV2 报错 | 强制 `backend=BACKEND.TORCH`（纯 torch，最兼容） |
| **P7** | **量化后 PPL 验证失败**：`'dict' object has no attribute 'get_loading_attributes'` | 无法自动验证 PPL | 跳过 PPL 验证，改用领域精度评测 |

### 5.3 文档与实际偏差

| # | 偏差 | 说明 |
|:-:|------|------|
| **D1** | `data/custom_data/` 数据源目录名与文档不一致 | 文档记录 10 个数据源（telecom_exam 等），实际 6 个（agentgen/comm_qa 等） |
| **D2** | `V100_SERVER_GUIDE.md` 3.2 节环境对比表版本错误 | 记录 torch 2.6.0/vllm 0.8.3/compressed-tensors 0.9.2，与 Dockerfile 冲突（已修正） |
| **D3** | `requirements-*.txt` 快照不可直接用于重建 | 从错误环境导出，含多处版本冲突（已修正） |

---

## 6. 部署结果

### 6.1 部署方式
- **引擎**：gptqmodel + TORCH backend（V100 兼容）
- **API**：OpenAI 兼容（`/v1/models`、`/v1/chat/completions`）
- **地址**：`http://0.0.0.0:8000`

### 6.2 部署验证
- `/v1/models`：返回模型 `Mind-SLLM-Qwen3-8B-GPTQ` ✅
- `/v1/chat/completions`：正常生成回答 ✅
  ```
  你好，我是一个大型语言模型，名为Qwen，由通义实验室开发。我具备广泛的知识和强大的语言处理能力，可以回答各种问题、创作文字、编程等。
  ```
- 生成 56 tokens，耗时 19.15s（TORCH backend 较慢，但功能正常）

### 6.3 部署性能说明
- TORCH backend 为纯 torch 实现，无自定义 CUDA kernel，**兼容性最好但速度较慢**
- 如需更高性能，可尝试 ExllamaV1（V100 兼容），但需解决 CUDA context 污染问题
- vLLM 0.7.1 不支持 Qwen3，无法用于本模型部署

---

## 7. 结论与建议

### 7.1 可行性结论
从零执行全链路**可行**，但需先解决 3 个阻塞性问题（P1-P3）。解决后全链路可跑通。

### 7.2 关键经验
1. **requirements 快照不可直接信任**：应以 Dockerfile 的版本组合为准（torch 2.5.1/vllm 0.7.1/compressed-tensors 0.9.0/numpy<2.0）
2. **torch 必须从 cu124 索引安装**：否则装成 CPU 版无 CUDA 支持
3. **V100 + Qwen3 组合受限**：vLLM 0.7.1 不支持 Qwen3，ExllamaV2 不支持 V100，只能用 gptqmodel + TORCH backend
4. **gptqmodel 2.0.0 加载 Qwen3 需注入 adapter**：`install_qwen3_gptq_adapter()`

### 7.3 后续优化建议
1. **升级 vLLM 到支持 Qwen3 的版本**（如 0.8.x），但需解决 compressed-tensors 冲突
2. **解决 ExllamaV1 的 CUDA context 污染问题**，提升推理速度
3. **扩大评测样本量**（50 → 200+），降低统计波动
4. **修正文档**：更新 V100_SERVER_GUIDE.md 环境对比表、requirements 快照、数据源清单
