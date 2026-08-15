# 从零执行操作手册 —— 压缩 → 部署 → 评估全链路

> **场景**：V100 服务器 `zetta_ld` 容器内，已清空项目目录（`/volume/workspace/llm-deploy`）和双虚拟环境
> （`/app/venv-quant`、`/app/venv-deploy`），**仅保留原始模型与 HF 缓存**。仅凭本仓库的文档和脚本，
> 从零完成「量化压缩 → 部署 → 评估」全链路。
>
> **前置保留项**（清空时不要动）：
> - `/app/local_models/Mind-SLLM-Qwen3-8B` —— 原始 FP16 模型
> - `/volume/hf_cache` —— HF 离线缓存（校准数据集回退用）
> - `/volume/models/` —— 既有量化模型（如需复用）
> - `zetta_ld` 容器本身及系统级 CUDA / NVIDIA 驱动
>
> **本手册串联 8 个步骤**，每步含：前置条件 → 操作命令 → 验证方法 → 故障排查。
> 逐步执行即可从零跑通全链路。各步细节可跳转到对应专题文档。

---

## 可行性评估结论

在「保留模型 + HF 缓存、清空项目目录 + 双 venv」的前提下，从零执行全链路 **可行**，但需先解决以下
**3 个阻塞点**（本手册已给出对应处理方案，按步骤执行即可规避）：

| # | 阻塞点 | 风险 | 本手册处理位置 |
|:-:|--------|------|----------------|
| 1 | 项目代码获取方式未文档化（清空 `/volume/workspace/llm-deploy` 后如何恢复代码） | 高 | [步骤 2](#2-恢复项目代码) |
| 2 | `data/custom_data/` 领域数据在 `.gitignore` 中，清空后 10 个数据源丢失，校准数据无法重建（数据源已备份于容器 `/volume/datahub/`，从零恢复时复制即可） | 高 | [步骤 4](#4-准备校准数据) |
| 3 | `cases/v100/gptq_vllm085/requirements-quant.txt` 引用 `file:///app/gptqmodel-2.0.0+...whl` 定制 whl，重建 venv-quant 时若该 whl 不在则安装失败 | 中 | [步骤 3](#3-重建双虚拟环境) |

**可行路径**：步骤 2 用 `scp`/`rsync` 从本地整目录上传项目代码；步骤 4 从容器 `/volume/datahub/`
**复制** `custom_data/` 与 `calibration/` 到项目目录（**不要用 `mv` 移动**，保留 datahub 源数据）
或改用 HF 数据集回退；步骤 3 优先用定制 whl，缺失时用 `install_quant_tools.sh` 从 PyPI 安装。

> ✅ **数据已统一存放于容器 `/volume/datahub/`**（含 `custom_data/` 与 `calibration/`），
> 今后无需再从本地上传数据。从零恢复时直接从 datahub 复制即可。

---

## 步骤总览

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 登录服务器 + 进入容器                                        │
│  2. 恢复项目代码 (本地 → 容器)                                   │
│  3. 重建多虚拟环境 (venv-quant + vllm-venv)                      │
│  4. 准备校准数据 (custom_data → v1 → v2)                         │
│  5. 确认原始模型可用                                             │
│  6. 执行量化 (quantize_model.py + gptqmodel 配置)                │
│  7. 部署服务 (serve_vllm085.py, vLLM 0.8.5)                      │
│  8. 评估 (精度 + 性能 + 领域精度)                                │
└─────────────────────────────────────────────────────────────────┘
```

| 步骤 | 用时参考 | 产出 |
|------|----------|------|
| 1. 登录 | < 1 min | 容器 shell |
| 2. 恢复代码 | 5-10 min | `/volume/workspace/llm-deploy/` 完整目录 |
| 3. 重建环境 | 20-40 min | `/app/venv-quant`、`/app/vllm-venv` |
| 4. 校准数据 | 10-20 min | `data/calibration/calibration_data_v2.jsonl` |
| 5. 确认模型 | < 1 min | 模型路径校验通过 |
| 6. 量化 | 90-120 min (8B, V100 单卡) | `/volume/models/Mind-SLLM-Qwen3-8B-GPTQ` |
| 7. 部署 | 2-5 min | `http://localhost:8000` |
| 8. 评估 | 30-60 min | `./results/` 精度 + 性能报告 |

---

## 1. 登录服务器并进入容器

**前置条件**：服务器开机、网络可达、`zetta_ld` 容器存在。

```bash
# 步骤 1：SSH 登录 V100 服务器
ssh jiysh@192.168.192.186
# 密码: jiyspcl@123

# 步骤 2：进入 zetta_ld 容器
docker exec -it zetta_ld bash

# 步骤 3：确认工作目录（此时应为空或不存在）
ls -la /volume/workspace/llm-deploy/
```

**验证**：容器内 `nvidia-smi` 能看到 8 块 V100，`ls /app/local_models/` 能看到原始模型。

**故障排查**：
- SSH 连接失败 → 见 [V100_SERVER_GUIDE.md 7.1](V100_SERVER_GUIDE.md#71-ssh-连接失败)
- 容器不存在 → 见 [V100_SERVER_GUIDE.md 7.2](V100_SERVER_GUIDE.md#72-docker-容器不存在)

> 连接信息细节见 [V100_SERVER_GUIDE.md 第 1 节](V100_SERVER_GUIDE.md#1-快速连接)。

---

## 2. 恢复项目代码

> ⚠️ **阻塞点 1**：清空 `/volume/workspace/llm-deploy` 后，项目代码（含 `llm_deploy/`、`configs/`、`cases/`、
> `docs/`、`requirements-*.txt`、`init` 等）全部丢失，必须从本地恢复。

**前置条件**：本地 `D:/project/opencode/llm-deploy/` 有完整项目代码。

### 2.1 方式 A：rsync 整目录上传（推荐）

在**本地终端**（非 SSH 内）执行：

```bash
# Windows PowerShell（使用 scp 递归上传）
scp -r D:/project/opencode/llm-deploy `
    jiysh@192.168.192.186:/volume/workspace/llm-deploy-restore

# 服务器端移动到目标位置（SSH 进服务器后）
ssh jiysh@192.168.192.186
rm -rf /volume/workspace/llm-deploy
mv /volume/workspace/llm-deploy-restore /volume/workspace/llm-deploy
```

### 2.2 方式 B：docker cp 整目录上传

```bash
# 本地终端：先 scp 到服务器宿主机，再 docker cp 进容器
scp -r D:/project/opencode/llm-deploy jiysh@192.168.192.186:/tmp/llm-deploy-src

# 服务器端：docker cp 进容器
ssh jiysh@192.168.192.186
docker cp /tmp/llm-deploy-src zetta_ld:/volume/workspace/llm-deploy
rm -rf /tmp/llm-deploy-src
```

### 2.3 验证

```bash
# 容器内执行
cd /volume/workspace/llm-deploy
ls -la
# 应看到: init, README.md, requirements*.txt, llm_deploy/, configs/, cases/, docs/, docker/

# 确认关键脚本存在
test -f llm_deploy/quantize_model.py && echo "OK: quantize_model.py"
test -f configs/gptq_4bit_v100_gptqmodel.yaml && echo "OK: gptqmodel 配置"
test -f cases/v100/gptq_vllm085/requirements-quant.txt && echo "OK: requirements-quant.txt"
test -f requirements-deploy.txt && echo "OK: requirements-deploy.txt"
```

**故障排查**：
- `scp` 速度慢 → 排除 `data/`、`models/`、`results/`、`cache/`、`bak/`、`vllm-env/` 等大目录
  （它们在 `.gitignore` 中，不需要上传）
- 权限不足 → `chown -R jiysh:jiysh /volume/workspace/llm-deploy`

> ✅ **`data/` 目录无需上传**：领域数据（`custom_data/`）与校准数据（`calibration/`）已统一存放于
> 容器 `/volume/datahub/`，从零恢复时用 `cp -r` **复制**到项目 `data/` 下即可（**不要用 `mv` 移动**），
> 详见 [步骤 4](#4-准备校准数据)。

> 单文件同步方式见 [V100_SERVER_GUIDE.md 第 4 节](V100_SERVER_GUIDE.md#4-项目文件同步从本地上传到服务器)。

---

## 3. 重建多虚拟环境

> ⚠️ **阻塞点 3**：`cases/v100/gptq_vllm085/requirements-quant.txt` 第 60 行引用
> `gptqmodel @ file:///app/gptqmodel-2.0.0+cu124torch2.5-cp312-cp312-linux_x86_64.whl`。
> 重建 `venv-quant` 时若该 whl 文件不在 `/app/` 下，`pip install -r` 会失败。

**前置条件**：项目代码已恢复（步骤 2 完成）。

### 3.1 确认 gptqmodel 定制 whl 是否存在

```bash
# 容器内执行
ls -la /app/gptqmodel-2.0.0*.whl 2>/dev/null && echo "whl 存在" || echo "whl 缺失"
```

### 3.2 重建 venv-quant（量化环境）

> ⚠️ **torch 必须从 PyTorch cu124 索引安装**：`cases/v100/gptq_vllm085/requirements-quant.txt` 中的 `torch==2.5.1` 若从
> PyPI 默认源安装，会装成 **CPU 版**（无 CUDA 支持，`torch.cuda.is_available()` 为 False），量化无法进行。
> 必须先单独从 cu124 索引安装 torch 三件套，再安装其余依赖。

**情况 A：whl 文件存在**（推荐路径，与 requirements 快照完全一致）：

```bash
python3 -m venv /app/venv-quant
source /app/venv-quant/bin/activate
pip install --upgrade pip

# ① 先装 torch 三件套（CUDA 12.4 版，与 gptqmodel whl 的 cu124torch2.5 匹配）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ② 再装其余依赖（torch 已满足，pip 不会重复下载 CPU 版）
pip install -r /volume/workspace/llm-deploy/cases/v100/gptq_vllm085/requirements-quant.txt
```

**情况 B：whl 文件缺失**（回退路径，从 PyPI 安装 gptqmodel）：

```bash
python3 -m venv /app/venv-quant
source /app/venv-quant/bin/activate
pip install --upgrade pip

# 先安装除 gptqmodel 外的依赖（手动跳过 file:// 引用行）
grep -v '^gptqmodel @ file://' /volume/workspace/llm-deploy/cases/v100/gptq_vllm085/requirements-quant.txt \
    > /tmp/req-quant-no-gptq.txt
pip install -r /tmp/req-quant-no-gptq.txt

# 再用 install_quant_tools.sh 从 PyPI 安装 gptqmodel 2.0.0
bash /volume/workspace/llm-deploy/cases/v100/install_quant_tools.sh
```

> ⚠️ **情况 B 风险**：PyPI 上的 `gptqmodel==2.0.0` 可能与定制 whl 的编译选项（CUDA 12.4 / torch 2.5）
> 不完全一致。若安装后量化报错，需从本地备份恢复定制 whl 并 `scp` 到 `/app/` 下重装。

### 3.3 重建 vllm-venv（部署评测环境，vLLM 0.8.5）

> ⚠️ **torch 必须从 PyTorch cu124 索引安装**：`cases/v100/gptq_vllm085/requirements-vllm085.txt` 中的 `torch==2.6.0+cu124`
> 若从 PyPI 默认源安装，会装成 **CPU 版**（无 CUDA 支持）。必须先单独从 cu124 索引安装 torch 三件套。

```bash
python3.12 -m venv /app/vllm-venv
source /app/vllm-venv/bin/activate
pip install --upgrade pip

# ① 先装 torch 三件套（CUDA 12.4 版，与 vLLM 0.8.5 的 _C.abi3.so ABI 兼容）
pip install torch==2.6.0 torchvision==0.25.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ② 再装 vLLM 0.8.5 + transformers 4.57.6
pip install vllm==0.8.5
pip install transformers==4.57.6

# ③ 其余依赖
pip install -r /volume/workspace/llm-deploy/cases/v100/gptq_vllm085/requirements-vllm085.txt
```

> ⚠️ **版本兼容性要点**：
> - `torch==2.6.0+cu124`：必须用 cu124 版（cu126 用新 ABI 导致 `undefined symbol`）
> - `transformers==4.57.6`：5.15.0 报 `Qwen2Tokenizer has no attribute all_special_tokens_extended`
> - `VLLM_ATTENTION_BACKEND=XFORMERS`：V100 不支持 Flash Attention

### 3.4 验证环境

```bash
# 验证 venv-quant
source /app/venv-quant/bin/activate
python -c "
import torch, gptqmodel, bitsandbytes, llmcompressor
print(f'torch {torch.__version__} | CUDA {torch.version.cuda}')
print(f'gptqmodel {gptqmodel.__version__}')
print(f'bitsandbytes {bitsandbytes.__version__}')
print(f'llmcompressor {llmcompressor.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用'
print('venv-quant: OK')
"

# 验证 vllm-venv (最新)
source /app/vllm-venv/bin/activate
python -c "
import torch, vllm, transformers
print(f'torch {torch.__version__} | vllm {vllm.__version__} | transformers {transformers.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用'
assert vllm.__version__ == '0.8.5', 'vllm 版本应为 0.8.5'
print('vllm-venv: OK')
"
```

**故障排查**：
- `pip install` 超时 → `export PIP_DEFAULT_TIMEOUT=300 && export PIP_RETRIES=5`
- gptqmodel 安装失败 → 见 [V100_SERVER_GUIDE.md 3.4 节](V100_SERVER_GUIDE.md#34-从零重建)
- torch CUDA 版本不匹配 → 确认容器 CUDA 版本：`nvcc --version`，对应安装 `torch==2.6.0+cu124`
- vllm 加载报 `undefined symbol` → 确认 torch 用的是 cu124 版（非 cu126）

> 多环境设计动机、对比、激活方式见 [V100_SERVER_GUIDE.md 第 3 节](V100_SERVER_GUIDE.md#3-多虚拟环境架构)。

---

## 4. 准备校准数据

> ⚠️ **阻塞点 2**：`data/custom_data/` 在 `.gitignore` 中，清空项目目录后 10 个领域数据源
> （telecom_exam、comm_qa_selfinst2、math 等）全部丢失。`build_calibration_data.py` 无数据可采，
> `calibration_data_v2.jsonl` 无法生成。

**前置条件**：项目代码已恢复（步骤 2），`venv-quant` 已重建（步骤 3）。

### 4.1 恢复 data/custom_data/ 领域数据

> ✅ **数据源已统一存放于容器 `/volume/datahub/`，无需再从本地上传。**
> 今后领域数据（`custom_data/`）与校准数据（`calibration/`）都放在 `/volume/datahub/` 下，
> 从零恢复时直接从 datahub **复制**到项目目录即可（**不要使用 `mv` 移动**，避免破坏 datahub 源数据）。

**方式 A：从 /volume/datahub 复制（推荐，保证领域精度）**：

```bash
# 容器内执行：从 datahub 复制 custom_data 到项目目录
cd /volume/workspace/llm-deploy
mkdir -p data
cp -r /volume/datahub/custom_data data/custom_data
cp -r /volume/datahub/calibration data/calibration
```

> ⚠️ 使用 `cp -r`（复制）而非 `mv`（移动），确保 `/volume/datahub/` 源数据始终保留，
> 便于后续多次重建或复用。

**验证数据源完整**：

```bash
# 容器内执行
cd /volume/workspace/llm-deploy
source /app/venv-quant/bin/activate
python llm_deploy/build_calibration_data.py --list-sources
# 应列出 10 个数据源: telecom_exam, comm_qa_selfinst2, math, comm_qa_selfinst1,
#   agent_sft, comm_qa_seed, spec_exam, agent_general, agent_iridium, codegen
```

**方式 B：无 datahub 数据时退化为 HF 通用校准集**（精度偏离领域最优）：

跳过本步，直接在 YAML 配置中移除 `custom_data` 字段，让 `get_calibration_texts()` 回退到
`neuralmagic/LLM_compression_calibration`（依赖 `/volume/hf_cache` 离线缓存）。

### 4.2 生成 v1 校准数据集

```bash
cd /volume/workspace/llm-deploy
source /app/venv-quant/bin/activate

# 从 10 个领域数据源混合采样 256 条
python llm_deploy/build_calibration_data.py --num-samples 256 --seed 42
# 产出: data/custom_data/calibration_data.jsonl
```

### 4.3 生成 v2 校准数据集（过滤超长序列，V100 必需）

> V100 上 SDPA math 后端对 >16K tokens 的序列会物化 O(n²) 注意力矩阵导致 OOM。
> v2 过滤掉超长样本（保留 ≤8192 tokens），是 V100 生产量化的必需步骤。

```bash
# 用 Qwen3 tokenizer 逐条 tokenize，过滤超长样本
python3 -c "
import json, sys
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained('/app/local_models/Mind-SLLM-Qwen3-8B', trust_remote_code=True)
MAX_TOKENS = 8192

with open('data/calibration/calibration_data.jsonl') as f:
    lines = f.readlines()

kept, dropped = 0, 0
with open('data/calibration/calibration_data_v2.jsonl', 'w') as out:
    for line in lines:
        obj = json.loads(line)
        msgs = obj.get('messages', [])
        text = tk.apply_chat_template(msgs, tokenize=False) if msgs else obj.get('text', '')
        tokens = tk.encode(text)
        if len(tokens) <= MAX_TOKENS:
            out.write(line)
            kept += 1
        else:
            dropped += 1

print(f'v2: {kept} kept, {dropped} dropped (max {MAX_TOKENS} tokens)')
"
```

### 4.4 验证

```bash
# 确认 v2 文件存在且行数合理（预期 ~230 条）
wc -l data/calibration/calibration_data_v2.jsonl

# 确认 YAML 配置指向 v2
grep custom_data configs/gptq_4bit_v100_gptqmodel.yaml
# 应输出: custom_data: "data/calibration/calibration_data_v2.jsonl"
```

**故障排查**：
- `--list-sources` 列出数据源 < 10 → `data/custom_data/` 恢复不完整，重新从 `/volume/datahub/custom_data` 复制
- v2 生成报 `OSError: ... model not found` → 确认 `/app/local_models/Mind-SLLM-Qwen3-8B` 存在
- v2 行数为 0 → 检查 v1 是否生成成功：`wc -l data/calibration/calibration_data.jsonl`

> 校准数据格式、数据源权重、v2 背景详见 [CALIBRATION_GUIDE.md 第 7 节](CALIBRATION_GUIDE.md#7-自定义校准数据)。

---

## 5. 确认原始模型可用

**前置条件**：步骤 1 完成（模型保留，未清空）。

```bash
# 容器内执行
ls -la /app/local_models/Mind-SLLM-Qwen3-8B/
# 应看到: config.json, model-*.safetensors, tokenizer.json, tokenizer_config.json 等

# 确认模型可加载（快速校验 config）
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/app/local_models/Mind-SLLM-Qwen3-8B', trust_remote_code=True)
print(f'model_type: {cfg.model_type}')
print(f'hidden_size: {cfg.hidden_size}')
print(f'num_layers: {cfg.num_hidden_layers}')
print('模型 config 加载: OK')
"
```

**验证**：`model_type` 应为 `qwen3`，`num_hidden_layers` 应为 36（8B 模型）。

> 模型路径与关键路径速查见 [V100_SERVER_GUIDE.md 第 2 节](V100_SERVER_GUIDE.md#2-关键路径)。

---

## 6. 执行量化

**前置条件**：步骤 3（venv-quant）、步骤 4（v2 校准数据）、步骤 5（模型）均完成。

### 6.1 GPU 架构校验

```bash
source /app/venv-quant/bin/activate
python -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, SM {p.major}.{p.minor}, {p.total_memory/1024**3:.1f} GB')
"
# V100 应显示 SM 7.0, 32 GB
```

### 6.2 执行 GPTQ 量化（gptqmodel 后端，V100 生产推荐）

```bash
cd /volume/workspace/llm-deploy
source /app/venv-quant/bin/activate

python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --validate
```

**参数说明**：
- `--model`：原始 FP16 模型路径（保留项）
- `--method gptq`：量化方法（从 config 读取可省略）
- `--config`：V100 生产配置（gptqmodel 后端 + v2 校准数据）
- `--output`：量化模型输出路径
- `--validate`：量化后自动做 PPL 验证（质量红线）

**预期耗时**：90-120 分钟（8B 模型，V100 单卡，36 layers × 230 样本 Hessian）。

### 6.3 验证量化产出

```bash
# 确认量化模型已生成
ls -la /volume/models/Mind-SLLM-Qwen3-8B-GPTQ/
# 应看到: config.json, quantize_config.json, model-*.safetensors, quant_log.csv

# 查看压缩比
du -sh /app/local_models/Mind-SLLM-Qwen3-8B/       # 基线 ~16G
du -sh /volume/models/Mind-SLLM-Qwen3-8B-GPTQ/     # 量化后 ~5.7G
# 预期压缩比 ~2.8x, 节省 ~64%

# 确认 quant_method=gptq（V100 兼容格式）
python3 -c "
import json
cfg = json.load(open('/volume/models/Mind-SLLM-Qwen3-8B-GPTQ/config.json'))
qc = cfg.get('quantization_config', {})
print(f'quant_method: {qc.get(\"quant_method\")}')
print(f'bits: {qc.get(\"bits\")}')
assert qc.get('quant_method') == 'gptq', '格式错误: 应为 gptq'
print('量化格式: OK (标准 GPTQ, V100 Exllama 兼容)')
"
```

**故障排查**：
- CUDA OOM → 确认用的是 v2 校准数据（v1 有 17K tokens 超长样本会 OOM）
- `gptqmodel MODEL_MAP 不含 qwen3` → 正常，`quantize_model.py` 会自动注入 `qwen3_gptq_adapter`
- PPL delta > 5.0 → 检查校准数据格式、`num_samples` 是否 ≥ 64
- llmcompressor pipeline 报错 → 确认用的是 `gptq_4bit_v100_gptqmodel.yaml`（gptqmodel 后端），
  不是 `gptq_4bit_v100.yaml`（llmcompressor 后端，V100 不兼容）

> GPTQ 双后端选择、Qwen3 兼容、V100 量化方案详见
> [V100_DEPLOY_GUIDE.md 第 4 节](V100_DEPLOY_GUIDE.md#4-量化方案详解-v100-适配版)。
> PPL 验证阈值与典型 delta 见 [CALIBRATION_GUIDE.md 6.5](CALIBRATION_GUIDE.md#65-量化后-ppl-闭环验证---validate)。

---

## 7. 部署服务

**前置条件**：步骤 6 完成（量化模型已生成）。

> ⚠️ **V100 + Qwen3 部署方案（实际验证）**：
> - **vLLM 0.7.1 不支持 Qwen3 架构**：报错 `Model architectures ['Qwen3ForCausalLM'] are not supported for now`。
> - **vLLM 0.8.5 支持 Qwen3 + V100**（V0 引擎 + XFORMERS），推理速度约 **30 tok/s**，
>   比 gptqmodel TORCH backend（2.6 tok/s）快约 **11.5 倍**。**这是最新落地方案**。
> - **transformers 无法加载 gptqmodel 2.0.0 的 GPTQ 模型**：报错要求 `gptqmodel>=7.0.0`，版本不兼容。

### 7.1 用 vLLM 0.8.5 部署（推荐，最新落地方案）

```bash
# 启动服务（后台），llm_deploy/serve_vllm085.py 提供 OpenAI 兼容 API
docker exec -d zetta_ld bash -c 'nohup /app/vllm-venv/bin/python /volume/workspace/llm-deploy/llm_deploy/serve_vllm085.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --port 8000 --gpu 0 > /tmp/serve_vllm085.log 2>&1 &'

# 查看启动日志（确认模型加载完成）
docker exec zetta_ld tail -10 /tmp/serve_vllm085.log
# 应看到: 服务已启动: http://0.0.0.0:8000
```

> `serve_vllm085.py` 内部逻辑：
> 1. 用 vLLM 0.8.5 的 `LLM()` 直接加载（避免 `vllm serve` 的 multiprocessing 内存 profiling bug）
> 2. V100 必需参数：`VLLM_ATTENTION_BACKEND=XFORMERS` + `enforce_eager=True` + `dtype=float16`
> 3. 提供 `/v1/models`、`/v1/chat/completions`、`/health` 接口

### 7.2 验证服务

```bash
# 健康检查
curl http://localhost:8000/health
# 应返回: {"status": "ok"}

# 查看可用模型
curl http://localhost:8000/v1/models
# 应返回: {"data":[{"id":"Mind-SLLM-Qwen3-8B-GPTQ",...}]}

# 对话测试（用文件方式传 JSON，避免转义问题）
cat > /tmp/test_chat.json <<'EOF'
{"messages":[{"role":"user","content":"你好，请介绍一下自己"}],"max_tokens":100}
EOF
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" --data @/tmp/test_chat.json
```

**故障排查**：
- `Initial used memory > currently used memory` → `vllm serve` 的 multiprocessing bug，
  确认用的是 `serve_vllm085.py`（`LLM()` 直接加载，uniproc executor）
- 输出退化 "!!!!" → vLLM 0.8.5 V0 引擎 bug，`serve_vllm085.py` 已用 dummy prompt 规避
- 服务启动慢 → 首次加载模型较慢，等待日志出现"服务已启动"

### 7.3 回退：gptqmodel + TORCH backend 部署

> 若 vLLM 0.8.5 环境不可用，可回退到 gptqmodel + TORCH backend（纯 torch，V100 最兼容，速度较慢）。

```bash
# 启动服务（后台），serve_gptq.py 位于 cases/v100/
docker exec -d zetta_ld bash -c 'nohup /app/venv-deploy/bin/python /volume/workspace/llm-deploy/cases/v100/serve_gptq.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --host 0.0.0.0 --port 8000 > /tmp/serve.log 2>&1 &'

# 查看启动日志（确认模型加载完成）
docker exec zetta_ld tail -10 /tmp/serve.log
# 应看到: 服务已启动: http://0.0.0.0:8000
```

> `serve_gptq.py` 内部逻辑：
> 1. 调用 `install_qwen3_gptq_adapter()` 注入 Qwen3 支持（gptqmodel 2.0.0 默认不识别 qwen3）
> 2. 用 `GPTQModel.from_quantized(..., backend=BACKEND.TORCH)` 加载（TORCH backend 兼容 V100）
> 3. 提供 `/v1/models`、`/v1/chat/completions`、`/health` 接口

> **方案对比**：
>
> | 方案 | 环境 | 速度 | 说明 |
> |------|------|------|------|
> | **vLLM 0.8.5**（推荐） | `/app/vllm-venv` | ~30 tok/s | 支持 Qwen3，快 11.5 倍 |
> | gptqmodel TORCH（回退） | `/app/venv-deploy` | ~2.6 tok/s | 纯 torch，最兼容 |

> V100 vLLM 最佳配置、多卡并行、环境变量详见
> [V100_DEPLOY_GUIDE.md 第 5 节](V100_DEPLOY_GUIDE.md#5-vllm-在-v100-上的最佳配置)。

---

## 8. 评估

**前置条件**：步骤 6 完成（精度评测独立进行，无需先部署）；步骤 7 完成（性能测试需要服务已启动）。

> ⚠️ **V100 + Qwen3 评测方案（实际验证）**：
> - **benchmark_domain.py 用 vLLM 0.8.5 本地后端**：vLLM 0.8.5 支持 Qwen3 + V100，
>   用 `--local --quantization gptq` 直接加载量化模型评测（快 11.5 倍）。
> - **benchmark_domain.py 用 transformers 后端**：transformers 4.57.6 无法加载 gptqmodel 2.0.0 的 GPTQ 模型
>   （要求 gptqmodel>=7.0.0），仅适用于 FP16 原模型。
> - **回退方案**：用 **compare_models.py**（gptqmodel + TORCH backend 加载量化模型，
>   transformers 加载原模型，在领域 Benchmark 上对比精度）。

### 8.1 领域精度对比评测（推荐，vLLM 0.8.5 本地）

```bash
cd /volume/workspace/llm-deploy
source /app/vllm-venv/bin/activate

# ① 先构建领域 Benchmark 数据集（从 data/custom_data/ 提取）
python llm_deploy/build_accuracy_benchmark.py --num-samples 100 --seed 44
# 产出: data/evaluation/accuracy_benchmark.jsonl

# ② 评测原模型 (FP16, vLLM 0.8.5 本地)
python llm_deploy/benchmark_domain.py \
    --local \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --output results/domain_baseline.json

# ③ 评测量化模型 (GPTQ, vLLM 0.8.5 本地)
python llm_deploy/benchmark_domain.py \
    --local \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq \
    --output results/domain_gptq.json
# 产出: results/domain_baseline.json, results/domain_gptq.json
```

> `benchmark_domain.py` vLLM 0.8.5 本地后端关键逻辑：
> 1. 设置 `VLLM_ATTENTION_BACKEND=XFORMERS`（V100 必需）
> 2. 用 `LLM()` 直接加载（`enforce_eager=True` + `dtype=float16` + `quantization=gptq`）
> 3. 用带 system 消息的 dummy prompt 填充到 2 条，规避 V0 引擎退化输出 bug
> 4. 对每条数据生成回答，用关键词匹配评分（score ≥ 0.35 记为通过）

### 8.1b 回退：compare_models.py 对比评测

> 若 vLLM 0.8.5 环境不可用，可回退到 compare_models.py（gptqmodel + TORCH backend）。

```bash
cd /volume/workspace/llm-deploy
source /app/venv-deploy/bin/activate

python /volume/workspace/llm-deploy/cases/v100/compare_models.py \
    --baseline /app/local_models/Mind-SLLM-Qwen3-8B \
    --quantized /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --benchmark data/evaluation/accuracy_benchmark.jsonl \
    --num-samples 50 \
    --max-tokens 256 \
    --output results/compare
# 产出: results/compare/compare_report.json
```

> `compare_models.py` 内部逻辑：
> 1. 原模型用 `AutoModelForCausalLM.from_pretrained`（transformers, FP16）
> 2. 量化模型用 `GPTQModel.from_quantized(..., backend=BACKEND.TORCH)`（gptqmodel, V100 兼容）
> 3. 对每条数据生成回答，用关键词匹配评分（score ≥ 0.35 记为通过）
> 4. 输出对比报告（准确率、正确数、精度差 delta）

### 8.2 性能测试（需服务已启动）

```bash
# 确认步骤 7 的服务已启动，然后：
python llm_deploy/benchmark_eval.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --perf-test --skip-accuracy \
    --base-url http://localhost:8000 \
    --num-prompts 100 \
    --max-tokens 512 \
    --concurrency 10 \
    --output /volume/workspace/llm-deploy/results/perf
```

### 8.3 查看评估结果

```bash
ls -la /volume/workspace/llm-deploy/results/
# compare/compare_report.json — 领域精度对比报告
# perf/                       — 性能测试结果
```

**实际精度参考**（Mind-SLLM-Qwen3-8B, V100, GPTQ INT4, vLLM 0.8.5 本地评测, 86 条领域样本）：

| 指标 | 原模型 (FP16) | GPTQ 4-bit | 精度损失 |
|------|:------------:|:----------:|:--------:|
| 准确率 | 43.02% | 39.53% | -3.49% |

> ⚠️ GPTQ 4-bit 保留约 92% 基线精度（39.53/43.02），math 类任务无损失。
> 精度损失在可接受范围内，换取 63% 显存节省和 11.5 倍速度提升。

**故障排查**：
- 精度评测 OOM → 加 `--gpu-memory-utilization 0.45 --enforce-eager --max-num-seqs 16`
- 性能测试连接拒绝 → 确认服务已启动：`curl http://localhost:8000/v1/models`
- 领域评测 `data/custom_data/` 报错 → 确认步骤 4.1 数据已恢复

> 评测命令模板、精度损失预期、结果格式详见 [USAGE_GUIDE.md 第 3 节](USAGE_GUIDE.md#3-评测使用方式)。
> 领域精度评测协议见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)。

---

## 附录 A：一键速查（已熟悉流程时使用）

```bash
# ========== 1. 登录 ==========
ssh jiysh@192.168.192.186
docker exec -it zetta_ld bash

# ========== 2. 恢复代码（本地终端执行） ==========
scp -r D:/project/opencode/llm-deploy jiysh@192.168.192.186:/volume/workspace/llm-deploy

# ========== 3. 重建环境 ==========
cd /volume/workspace/llm-deploy
python3 -m venv /app/venv-quant && source /app/venv-quant/bin/activate
pip install -r cases/v100/gptq_vllm085/requirements-quant.txt
python3.12 -m venv /app/vllm-venv && source /app/vllm-venv/bin/activate
pip install torch==2.6.0 torchvision==0.25.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5 transformers==4.57.6
pip install -r cases/v100/gptq_vllm085/requirements-vllm085.txt

# ========== 4. 校准数据 ==========
# 4.0 从 datahub 复制领域数据与校准数据（不要用 mv，保留 datahub 源数据）
cd /volume/workspace/llm-deploy
mkdir -p data
cp -r /volume/datahub/custom_data data/custom_data
cp -r /volume/datahub/calibration data/calibration

source /app/venv-quant/bin/activate
python llm_deploy/build_calibration_data.py --num-samples 256 --seed 42
# 生成 v2（见步骤 4.3 脚本）

# ========== 5. 确认模型 ==========
ls /app/local_models/Mind-SLLM-Qwen3-8B/

# ========== 6. 量化 ==========
source /app/venv-quant/bin/activate
python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --validate

# ========== 7. 部署 (vLLM 0.8.5) ==========
source /app/vllm-venv/bin/activate
python llm_deploy/serve_vllm085.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --port 8000 --gpu 0

# ========== 8. 评估（另开终端） ==========
source /app/vllm-venv/bin/activate
# 领域精度评测 (vLLM 0.8.5 本地)
python llm_deploy/benchmark_domain.py \
    --local --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --output results/domain_baseline.json
python llm_deploy/benchmark_domain.py \
    --local --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --output results/domain_gptq.json
# 性能测试 (需服务已启动)
python llm_deploy/benchmark_eval.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --perf-test --skip-accuracy --base-url http://localhost:8000 \
    --num-prompts 100 --max-tokens 512 --concurrency 10 --output ./results/perf
```

---

## 附录 B：相关文档导航

| 文档 | 内容 |
|------|------|
| [V100_SERVER_GUIDE.md](V100_SERVER_GUIDE.md) | SSH/Docker 连接、双 venv 架构、关键路径、常见问题 |
| [V100_DEPLOY_GUIDE.md](V100_DEPLOY_GUIDE.md) | V100 量化方案、vLLM 配置、显存规划、Docker 部署 |
| [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) | 校准数据格式、数据源、v2 适配、PPL 验证 |
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | 量化/评测/部署命令模板、按 GPU 选方案 |
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | 评估协议、领域精度评测方法 |
| [GPU_ARCHITECTURE_GUIDE.md](GPU_ARCHITECTURE_GUIDE.md) | V100/A100/H100 跨硬件兼容性 |
