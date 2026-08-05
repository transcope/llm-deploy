# 量化流程问题记录

> 本文档记录 2026-08-05 对 Mind-SLLM-Qwen3-8B 进行 GPTQ W4A16 量化过程中发现的
> 项目文档/脚本/配置不足，供后续完善参考。

---

## 1. 校准数据序列过长导致 OOM

### 现象
量化第 2 条校准样本（Agent 多轮对话，17,330 tokens）时 CUDA OOM：
```
torch.OutOfMemoryError: Tried to allocate 35.80 GiB. GPU 0 has a total capacity of 31.73 GiB
```

### 根因
- `calibration_data.jsonl` 中有 26 条（10%）超过 8192 tokens，最长 28,135 tokens
- PyTorch SDPA 在 V100（SM 7.0）上对超长序列（>16K）回退到 **math 后端**，物化完整 O(n²) 注意力矩阵
- 28K 序列的注意力矩阵需要 ~50GB，远超 V100 的 32GB 显存

### 修复
在 `quantize_gptq_with_gptqmodel()` 中强制 memory-efficient SDPA 后端（`quantize_model.py`）：
```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```
经测试，memory-efficient 后端在 V100 上支持到 28K+ tokens，内存复杂度 O(n) 而非 O(n²)。

### 项目缺口
- ❌ 未考虑 V100 上 SDPA 后端回退问题
- ❌ 无 `max_calib_length` 配置项（安全截断上限）
- ❌ `batch_size=1` 硬编码在配置中但未说明显存影响
- ✅ 校准数据格式（messages 字段）正确，代码消费正常

---

## 2. gptqmodel 多卡并行不可用

### 现象
尝试 `model_init_kwargs={"device_map": "auto"}` 利用 8 卡分摊模型权重，但：
1. gptqmodel loader 在 `loader.py:189` 强制 `model_init_kwargs["device_map"] = cpu_device_map`
2. `model_init_kwargs` 作为整体参数传入 HuggingFace 的 `from_pretrained`，不被识别

### 根因
gptqmodel 量化流程是**逐层串行**的：
1. 模型加载到 CPU（`device_map="cpu"`）
2. 每层逐个搬到 GPU → 前向传播 → 量化 → 搬回 CPU

**单次 GPTQ 量化无法利用多卡并行。**

### 项目缺口
- ❌ `V100_DEPLOY_GUIDE.md` 和 `USAGE_GUIDE.md` 均未说明量化不支持多卡
- ❌ 多卡价值仅在部署阶段，量化阶段无说明

---

## 3. SDPA 后端对 V100 兼容性缺失

### 现象
V100（SM 7.0）不支持 FlashAttention-2（需 SM 8.0+），PyTorch SDPA 在序列 >16K 时自动回退到 math 后端。

### 发现的 PyTorch SDPA 行为

| SDPA 后端 | V100 支持 | 序列限制 | 内存复杂度 |
|:----------|:---------:|:--------:|:----------:|
| FlashAttention | ❌ | — | — |
| Memory-Efficient | ✅ | 无（测试到 28K） | O(n) |
| Math（回退） | ✅ | 有（>16K 自动启用） | O(n²) |

### 项目缺口
- ❌ 没有任何文档提及 V100 + transformers 的 SDPA 后端选择
- ❌ `V100_DEPLOY_GUIDE.md` 只说了 vLLM 的 `VLLM_ATTENTION_BACKEND=XFORMERS`，未覆盖量化阶段的 transformers SDPA
- ❌ 量化脚本无任何 SDPA 后端探测或配置

---

## 4. 数据目录结构混乱

### 现象
`data/` 下多个目录功能重叠，临时文件散落：

```
data/
├── custom_data/
│   ├── accuracy_benchmark.jsonl      ← 临时文件（评测数据，不应在此）
│   ├── calibration_data.jsonl        ← 临时文件（校准数据，不应在此）
│   ├── eval_data.jsonl               ← 临时文件（PPL 评估数据，不应在此）
│   └── comm_qa/ math/ ...            ← 源数据（正确位置）
├── eval_data/                        ← 评测基准（路径名不够语义化）
│   └── accuracy_benchmark.jsonl
```

### 修复
目标结构：
```
data/
├── calibration/
│   └── calibration_data.jsonl        ← 校准数据
├── evaluation/                       ← （由 eval_data/ 改名）
│   ├── accuracy_benchmark.jsonl      ← 精度评测基准
│   └── eval_data.jsonl               ← PPL 验证数据
└── custom_data/
    └── comm_qa/ math/ ...            ← 仅源数据子目录
```

涉及的脚本更新：
- `build_accuracy_benchmark.py`：默认输出路径 `eval_data` → `evaluation`
- `build_calibration_data.py`：eval 输出路径 `custom_data/eval_data.jsonl` → `evaluation/eval_data.jsonl`

### 项目缺口
- ❌ 没有明确的 `data/` 目录规范文档
- ❌ `build_calibration_data.py` 默认输出到 `custom_data/`（与"原始数据只读"原则矛盾）
- ❌ 目录命名不一致：`calibration/` vs `calibration_data/` vs `eval_data/` vs `evaluation/`

---

## 5. 文档路径引用过期

### 现象
`EVALUATION_PROTOCOL.md` 和 `CALIBRATION_GUIDE.md` 中大量引用 `data/eval_data/` 和 `data/custom_data/eval_data.jsonl` 路径。

### 修复
全局替换 `data/eval_data/` → `data/evaluation/`，更新 `data/custom_data/eval_data.jsonl` → `data/evaluation/eval_data.jsonl`。

涉及文件：
- `docs/EVALUATION_PROTOCOL.md`：18 处引用
- `docs/CALIBRATION_GUIDE.md`：1 处引用
- `src/build_accuracy_benchmark.py`：2 处
- `src/build_calibration_data.py`：1 处

### 项目缺口
- ❌ 文档路径未随重构同步更新
- ❌ 无自动化路径一致性检查

---

## 6. 校准数据来源偏斜

### 现象
256 条校准数据中：
- 26 条（10%）是超长 Agent 多轮对话（>8192 tokens）
- 中位数仅 410 tokens → 绝大多数是短样本
- 长文本全部来自 Agent 数据

### 影响
- Agent 长序列在校准中占主导的 token 分布，可能导致量化对短文本场景精度下降
- 长序列的前向传播占总时间的绝大部分

### 项目缺口
- ❌ `build_calibration_data.py` 无 `max_seq_length` 过滤选项
- ❌ 无长文本占比配置（当前硬编码 `--long-ratio 0.5`）
- ❌ 校准数据中不同来源的序列长度差异过大，无均匀性保证

---

## 7. gptqmodel 版本依赖未锁定

### 现象
容器中安装的是 `gptqmodel 2.0.0+cu124torch2.5`，但：
- `install_quant_tools.sh` 锁定 `gptqmodel==2.0.0`
- `requirements.txt` 也有注释说明但无实际锁定
- 2.0.0 的 MODEL_MAP 没有 qwen3，需要 adapter 注入

### 项目缺口
- ❌ `requirements.txt` 中的量化工具版本与实际使用的版本可能不一致
- ❌ 无 `gptqmodel` 版本升级后的回归测试流程
- ❌ Qwen3 adapter 的长期维护方案未明确（上游支持后是否仍需 adapter）

---

## 8. 量化后验证流程不完整

### 现象
`--validate` 参数依赖 `validate_calibration.py` 模块，但：
- 该模块需要 PPL 评估数据集（`eval_data.jsonl`）
- 需要先跑基准 PPL 再对比量化后 PPL
- 文档流程缺失"先 baseline 后量化再对比"的完整步骤

### 项目缺口
- ❌ `EVALUATION_PROTOCOL.md` 5.2 节 baseline 命令中的路径仍是过时的 `data/eval_data/`
- ❌ 量化 → 验证的端到端脚本缺失
- ❌ `--validate` 在 `configs/gptq_4bit_v100_gptqmodel.yaml` 中无对应配置项

---

## 总结：项目完善优先级

| 优先级 | 问题 | 影响面 |
|:------:|:-----|:------:|
| P0 | SDPA 后端兼容性（V100 OOM） | 量化完全不可用 |
| P1 | 校准数据超长序列处理 | 量化可用性 |
| P1 | 数据目录规范 | 长期维护 |
| P2 | 文档路径同步 | 可读性 |
| P2 | 多卡并行说明 | 用户预期管理 |
| P3 | 校准数据质量控制 | 量化精度 |
| P3 | 版本依赖管理 | 可复现性 |
| P3 | 验证流程自动化 | 效率 |
