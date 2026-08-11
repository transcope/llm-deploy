# V100 服务器连接与操作指南

> 本文档记录 V100 服务器（192.168.192.186）的 SSH 连接、Docker 容器进入、环境激活等标准操作流程，以及注意事项和常见问题排查。

---

## 1. 快速连接

```bash
# 步骤 1：SSH 登录 V100 服务器
ssh jiysh@192.168.192.186
# 密码: jiyspcl@123

# 步骤 2：进入 zetta_ld 容器
docker exec -it zetta_ld bash

# 步骤 3：进入项目工作目录
cd /volume/workspace/llm-deploy

# 步骤 4：根据任务选择虚拟环境
#   量化模型 → 量化环境 (venv-quant)
source /app/venv-quant/bin/activate
#   或 source cases/v100/activate_quant.sh

#   部署/评测 → 部署评测环境 (venv-deploy)
source /app/venv-deploy/bin/activate
#   或 source cases/v100/activate_deploy.sh
```

**记住：每次新开终端都要执行步骤 2~4。**

**环境选择速查：**

| 任务 | 使用环境 | 原因 |
|:-----|:---------|:-----|
| 模型量化 (GPTQ/AWQ) | **venv-quant** | 含 gptqmodel 2.0.0、bitsandbytes 等量化工具链 |
| vLLM 模型部署 | **venv-deploy** | 无量化工具链，更轻量 |
| 精度评测 (benchmark_domain.py) | **venv-deploy** | 无需 gptqmodel，避免版本冲突 |
| PPL 验证 (validate_calibration.py) | **venv-deploy** | 只需 torch + transformers |
| lm-eval 标准评测 | **venv-deploy** | 只需 lm_eval + torch |

**旧环境** `/app/venv/` 保留作为兼容，新任务请优先使用以上两个环境。

> ⚠️ 密码如变更请更新本文档。不要将本文档提交到公共仓库。

---

## 2. 关键路径

| 路径 | 说明 |
|---|---|
| `/volume/workspace/llm-deploy/` | 项目工作目录 |
| `/app/local_models/Mind-SLLM-Qwen3-8B` | 原始模型（huggingface 格式） |
| `/volume/models/` | 模型存放目录 |
| `/volume/models/Mind-SLLM-Qwen3-8B-GPTQ` | **最新 GPTQ 量化模型**（5.8 GB，V100 用 TORCH backend 加载） |
| `/app/venv/` | Python 环境（旧，保留兼容） |
| `/app/venv-quant/` | **量化环境**（gptqmodel 2.0.0 + 量化工具链） |
| `/app/venv-deploy/` | **部署评测环境**（vLLM + transformers，无 gptqmodel） |
| `/volume/workspace/llm-deploy/src/activate_quant.sh` | 量化环境快捷激活脚本 |
| `/volume/workspace/llm-deploy/src/activate_deploy.sh` | 部署评测环境快捷激活脚本 |
| `/volume/workspace/llm-deploy/requirements-quant.txt` | 量化环境依赖快照 |
| `/volume/workspace/llm-deploy/requirements-deploy.txt` | 部署评测环境依赖快照 |
| `/volume/workspace/llm-deploy/data/calibration/` | 校准数据目录（含 v1/v2） |
| `/volume/workspace/llm-deploy/data/evaluation/` | 评测数据 |

---

## 3. 双虚拟环境架构

### 3.1 设计动机

项目采用双虚拟环境隔离，解决以下问题：

| 问题 | 解决方式 |
|:-----|:---------|
| **gptqmodel 版本锁定**：量化依赖 gptqmodel 2.0.0（定制 whl，编译为 cu124torch2.5），但 vLLM 0.8.x 要求 torch 2.6.0 / compressed-tensors 0.9.2，与 gptqmodel whl（torch 2.5）及 llmcompressor 0.4.0（compressed-tensors 0.9.0）冲突 | 量化与部署分离，各自独立维护依赖；统一锁定 torch 2.5.1+cu124 / vLLM 0.7.1 / compressed-tensors 0.9.0 / numpy<2.0 |
| **optimum 不兼容**：optimum 要求 gptqmodel≥7.0.0，但核心工作流（量化→vLLM 部署）不经过 optimum | deploy 环境安装 optimum 但不含 [gptq] 扩展，无版本冲突 |
| **环境轻量化**：评测脚本只需要 torch + transformers，不需要 gptqmodel、bitsandbytes 等量化工具 | deploy 环境减少 ~0.2 GB 无用依赖 |

### 3.2 环境对比

| 维度 | venv-quant | venv-deploy |
|:-----|:-----------|:------------|
| 路径 | `/app/venv-quant/` | `/app/venv-deploy/` |
| 大小 | 8.5 GB | 8.3 GB |
| 基础框架 | torch 2.5.1+cu124, transformers 4.51.0 | torch 2.5.1+cu124, transformers 4.51.0 |
| 推理引擎 | vLLM 0.7.1 | vLLM 0.7.1 |
| 量化工具 | gptqmodel 2.0.0, bitsandbytes 0.49.2, llmcompressor 0.4.0, compressed-tensors 0.9.0 | ❌ 无 |
| optimum | ✅ (不含 [gptq] 扩展) | ✅ (不含 [gptq] 扩展) |
| 快捷激活 | `source cases/v100/activate_quant.sh` | `source cases/v100/activate_deploy.sh` |

### 3.3 激活方式

```bash
# 方式 A：直接激活
source /app/venv-quant/bin/activate     # 量化环境
source /app/venv-deploy/bin/activate    # 部署评测环境

# 方式 B：快捷脚本（推荐，显示版本信息）
source cases/v100/activate_quant.sh
source cases/v100/activate_deploy.sh
```

### 3.4 从零重建

如果环境损坏，可从对应 requirements 快照重建。

> ⚠️ **gptqmodel 定制 whl 依赖**：`requirements-quant.txt` 第 60 行引用
> `gptqmodel @ file:///app/gptqmodel-2.0.0+cu124torch2.5-cp312-cp312-linux_x86_64.whl`。
> 重建 `venv-quant` 前，必须先确认该 whl 文件存在于 `/app/` 下，否则 `pip install -r` 会失败。

#### 3.4.1 确认 gptqmodel 定制 whl

```bash
# 容器内执行
ls -la /app/gptqmodel-2.0.0*.whl 2>/dev/null && echo "whl 存在" || echo "whl 缺失"
```

#### 3.4.2 重建 venv-quant

> ⚠️ **torch 必须从 PyTorch cu124 索引安装**：`requirements-quant.txt` 中的 `torch==2.5.1` 若从
> PyPI 默认源安装，会装成 **CPU 版**（无 CUDA 支持，`torch.cuda.is_available()` 为 False），量化无法进行。
> 必须先单独从 cu124 索引安装 torch 三件套，再安装其余依赖。

**情况 A：whl 文件存在**（推荐，与 requirements 快照完全一致）：

```bash
python3 -m venv /app/venv-quant
source /app/venv-quant/bin/activate
pip install --upgrade pip

# ① 先装 torch 三件套（CUDA 12.4 版，与 gptqmodel whl 的 cu124torch2.5 匹配）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ② 再装其余依赖（torch 已满足，pip 不会重复下载 CPU 版）
pip install -r /volume/workspace/llm-deploy/requirements-quant.txt
```

**情况 B：whl 文件缺失**（回退，从 PyPI 安装 gptqmodel）：

```bash
python3 -m venv /app/venv-quant
source /app/venv-quant/bin/activate
pip install --upgrade pip

# 先安装除 gptqmodel 外的依赖（跳过 file:// 引用行）
grep -v '^gptqmodel @ file://' /volume/workspace/llm-deploy/requirements-quant.txt \
    > /tmp/req-quant-no-gptq.txt
pip install -r /tmp/req-quant-no-gptq.txt

# 再用 install_quant_tools.sh 从 PyPI 安装 gptqmodel 2.0.0
bash /volume/workspace/llm-deploy/cases/v100/install_quant_tools.sh
```

> ⚠️ **情况 B 风险**：PyPI 上的 `gptqmodel==2.0.0` 可能与定制 whl 的编译选项（CUDA 12.4 / torch 2.5）
> 不完全一致。若安装后量化报错，需从本地备份恢复定制 whl 并 `scp` 到 `/app/` 下重装。
> `install_quant_tools.sh` 还会额外安装 `auto-gptq`（legacy）、`bitsandbytes`、`llmcompressor`、
> `compressed-tensors`，与 requirements 快照版本可能略有差异。

#### 3.4.3 重建 venv-deploy

```bash
python3 -m venv /app/venv-deploy
source /app/venv-deploy/bin/activate
pip install --upgrade pip
pip install -r /volume/workspace/llm-deploy/requirements-deploy.txt
```

#### 3.4.4 重建后验证

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

# 验证 venv-deploy
source /app/venv-deploy/bin/activate
python -c "
import torch, vllm, transformers
print(f'torch {torch.__version__} | vllm {vllm.__version__} | transformers {transformers.__version__}')
assert torch.cuda.is_available(), 'CUDA 不可用'
print('venv-deploy: OK')
"
```

> 完整的从零重建流程（含项目代码恢复、校准数据准备）见
> [FROM_SCRATCH_RUNBOOK.md](FROM_SCRATCH_RUNBOOK.md)。

---

## 4. 项目文件同步（从本地上传到服务器）

### 4.1 单文件同步

在本地终端（非 SSH 内）执行：

```bash
# 复制单个文件到容器
cat "D:/project/opencode/llm-deploy/src/build_accuracy_benchmark.py" | ssh jiysh@192.168.192.186 "docker exec -i zetta_ld bash -c 'cat > /volume/workspace/llm-deploy/src/build_accuracy_benchmark.py'"
```

### 4.2 整目录同步（从零恢复项目代码时使用）

> 当 `/volume/workspace/llm-deploy/` 被清空时，需从本地整目录上传恢复项目代码。
> 排除 `data/`、`models/`、`results/`、`cache/`、`bak/`、`vllm-env/` 等大目录（在 `.gitignore` 中）。

**方式 A：scp 递归上传**（Windows PowerShell）：

```bash
# 本地终端执行
scp -r D:/project/opencode/llm-deploy `
    jiysh@192.168.192.186:/volume/workspace/llm-deploy-restore

# 服务器端移动到目标位置
ssh jiysh@192.168.192.186
rm -rf /volume/workspace/llm-deploy
mv /volume/workspace/llm-deploy-restore /volume/workspace/llm-deploy
```

**方式 B：docker cp**（先 scp 到宿主机，再 cp 进容器）：

```bash
# 本地终端
scp -r D:/project/opencode/llm-deploy jiysh@192.168.192.186:/tmp/llm-deploy-src

# 服务器端
ssh jiysh@192.168.192.186
docker cp /tmp/llm-deploy-src zetta_ld:/volume/workspace/llm-deploy
rm -rf /tmp/llm-deploy-src
```

**验证**：

```bash
# 容器内执行
cd /volume/workspace/llm-deploy
ls -la
# 应看到: init, README.md, requirements*.txt, src/, configs/, cases/, docs/, docker/
```

> 完整的从零恢复流程见 [FROM_SCRATCH_RUNBOOK.md 步骤 2](FROM_SCRATCH_RUNBOOK.md#2-恢复项目代码)。

---

## 5. 硬件信息

- **GPU**: 8 × Tesla V100S-PCIE-32GB（SM 7.0）
- **显存**: 32 GB/卡
- **CUDA Compute Capability**: 7.0
- **不支持**: bfloat16（V100 无 BF16 支持）
- **Dtype 建议**: `float16` 或 `float32`

---

## 6. V100 适配要点

### 6.1 模型推理

| 场景 | 推荐后端 | 说明 |
|---|---|---|
| Qwen3 系列精度评测 | `transformers` | vLLM 在 V100+Qwen3 会崩溃（LLVM ERROR: Failed to compute parent layout） |
| 其他模型推理 | `vllm` 或 `transformers` | 可按需选择 |

### 6.2 transformers 推理注意事项

```python
# ✅ 正确：单卡加载方式
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=None,          # 不要用 "auto"
).to("cuda")

# ❌ 错误：device_map="auto" 会触发 accelerate 钩子，引入额外开销
# ❌ 错误：vLLM 在 V100+Qwen3 会崩溃
```

### 6.3 Qwen3 Thinking 模式

Qwen3 默认输出中文思维链（CoT）。精度评测时必须抑制：

```python
# 生成时传入
tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
```

在 `benchmark_domain.py` 中使用 `--backend transformers --no-thinking` 自动处理。

---

## 7. 常见问题

### 7.1 SSH 连接失败

```
ssh: connect to host 192.168.192.186 port 22: Connection refused
```

**排查**: 确认服务器是否开机、网络是否可达（ping 192.168.192.186）。

### 7.2 Docker 容器不存在

```
Error response from daemon: No such container: zetta_ld
```

**排查**: `docker ps -a` 查看所有容器，确认容器名是否正确。

### 7.3 Python 虚拟环境未激活

```
ModuleNotFoundError: No module named 'torch'
```

**解决**: 执行 `source /app/venv/bin/activate`。

### 7.4 vLLM 在 V100 上崩溃

**实际报错**（vLLM 0.7.1 + Qwen3）：
```
ValueError: Model architectures ['Qwen3ForCausalLM'] are not supported for now.
```
（旧版 vLLM 也可能报 `LLVM ERROR: Failed to compute parent layout for slice layout`）

**原因**: vLLM 0.7.1 的模型注册表不含 `Qwen3ForCausalLM`（只有 Qwen2），且 V100（SM 7.0）与部分 vLLM kernel 不兼容。

**解决**:
1. **首选**：用 **gptqmodel + TORCH backend** 部署（V100 兼容），见
   [V100_DEPLOY_GUIDE.md 4.5 节](V100_DEPLOY_GUIDE.md#45-v100--qwen3-部署方案实际验证)
2. **注意**：`--backend transformers` 无法加载 gptqmodel 2.0.0 的 GPTQ 模型（transformers 要求 gptqmodel>=7.0.0），
   仅适用于 FP16 原模型
3. 升级 vLLM 到支持 Qwen3 的版本（如 0.8.x），但需解决 compressed-tensors 冲突

### 7.5 显存不足 OOM

```
CUDA out of memory
```

**解决**: 
- 检查是否有其他进程占用 GPU：`nvidia-smi`
- 减少 `batch_size`
- 使用 `float16` 而非 `float32`

### 7.6 模型加载慢

首次加载 HuggingFace 模型需要下载权重到缓存目录。确认 `HF_HOME` 或 `TRANSFORMERS_CACHE` 指向有足够空间的目录。

---

## 8. 快捷命令备忘录

```bash
# 查看 GPU 状态
nvidia-smi

# 查看容器日志
docker logs zetta_ld

# 查看所有容器
docker ps -a

# 在容器内执行单条命令（无需交互式）
docker exec zetta_ld bash -c 'source /app/venv/bin/activate && python src/build_accuracy_benchmark.py --help'
```

---

## 9. 历史教训

> ⚠️ 不要混淆服务器 IP 地址。V100 服务器的正确 IP 为 **192.168.192.186**（端口 22）。
> 之前的错误 IP 192.168.1.24 已废弃。连接信息如有变更，请更新本文档。
>
> 每次新会话开始，先查阅本文档确认连接参数，避免使用过时信息。
