# 测试指南 (TESTING)

> 本文档记录项目的单元测试体系：测试文件组织、覆盖范围、运行方式与新增测试规范。
> 适用对象：所有需要在本地或容器内验证代码正确性的开发者。

---

## 1. 测试概览

项目采用 **pytest** 作为测试框架，测试代码全部位于 `tests/` 目录。
所有测试均为 **离线单元测试**（mock GPU / 模型 / 依赖库），**不需要真实 GPU 或模型即可运行**，
适合在本地开发环境、CI 以及服务器容器内快速回归验证。

| 测试文件 | 测试对象 | 覆盖重点 |
|---------|---------|---------|
| `test_quantize_model.py` | `llm_deploy/quantize_model.py` | 量化配置加载、CLI 参数覆盖 YAML 配置 |
| `test_deploy_server.py` | `llm_deploy/deploy_server.py` | 部署命令构建、硬件约束（V100/A100）、量化方式检测 |
| `test_benchmark_eval.py` | `llm_deploy/benchmark_eval.py` | 评测分数提取、模型参数构建、lm-eval 命令、流式请求、性能测试 |
| `test_qwen3_gptq_adapter.py` | `llm_deploy/qwen3_gptq_adapter.py` | Qwen3GPTQ 注入/卸载、幂等性、list/set 容器兼容 |
| `test_qwen3_pipeline_patch.py` | `llm_deploy/qwen3_pipeline_patch.py` | Qwen3 decoder layer 共享 PE 缓存（v2 修复验证） |
| `test_match_modules_patch.py` | `llm_deploy/qwen3_pipeline_patch.py` | match_modules 数值排序补丁（36 layer 顺序） |
| `conftest.py` | — | 统一注入 `llm_deploy/` 到 `sys.path`（测试基础设施） |

---

## 2. 运行测试

### 2.1 一键运行（推荐）

```bash
./run_tests.sh
```

`run_tests.sh` 会自动：
1. 定位项目根目录
2. 优先激活项目本地虚拟环境 `vllm-env/`；容器内则复用 `/app/venv`
3. 执行 `python -m pytest tests -v`

### 2.2 直接运行（已激活环境时）

```bash
python -m pytest tests/ -v
```

### 2.3 运行单个测试文件

```bash
python -m pytest tests/test_qwen3_gptq_adapter.py -v
```

### 2.4 运行单个测试用例

```bash
python -m pytest tests/test_match_modules_patch.py::test_natural_sort_key -v
```

### 2.5 输出简化（只看汇总）

```bash
python -m pytest tests/ --no-header -q
```

---

## 3. 测试设计要点

### 3.1 无需 GPU / 真实模型

所有测试通过 mock 方式隔离外部依赖：

| 依赖 | 模拟方式 |
|------|---------|
| `gptqmodel` 包 | 构造 fake 模块注入 `sys.modules`（`test_qwen3_gptq_adapter.py`） |
| `torch.nn.Module` 模型 | 自定义 `Qwen3ForCausalLM` / `Qwen3DecoderLayer` mock 类 |
| 模型权重 | 不加载，仅验证逻辑层行为 |
| 推理服务 | 无（仅验证命令字符串构建） |

### 3.2 关键测试场景说明

- **`test_qwen3_pipeline_patch.py`**：验证 v2 修复——所有 decoder layer 共享一个 `_PositionEmbeddingsCache`。
  模拟 `capture_first_layer_intermediates`（layer 0 填充缓存）→ layer_sequential（layer 1..N 从缓存取 PE）
  的真实调用顺序，并验证按 `batch_idx` 正确索引。
- **`test_match_modules_patch.py`**：验证数值排序补丁——36 个 layer 在 patch 后按 `0,1,2,...,35`
  而非字典序 `0,1,10,11,...` 排序，这是 Qwen3 量化时必须的修复。
- **`test_qwen3_gptq_adapter.py`**：验证适配器幂等性——二次 `install` 不重复注入，`uninstall` 还原初始状态，
  且兼容 `SUPPORTED_MODELS` 的 list/set 两种容器类型。

### 3.3 环境隔离

`conftest.py` 统一处理 `sys.path` 注入，各测试文件内部自行管理模块加载/卸载（`importlib` + `sys.modules.pop`），
避免全局状态污染影响测试结果。

---

## 4. 新增测试规范

新增单元测试时请遵循：

1. **文件命名**：`test_<被测模块>.py`，放在 `tests/` 目录
2. **测试函数命名**：`test_<行为描述>`，一个函数只验证一个行为点
3. **离线原则**：不依赖真实 GPU、模型权重、外部服务；如需模型结构，用 mock 类
4. **独立原则**：每个测试用例互不依赖，用 fixture 或手动清理模块状态
5. **覆盖新逻辑**：新增功能（如新配置项、新命令参数）应同步补充对应测试

---

## 5. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: llm_deploy` | 未走 `run_tests.sh`，`sys.path` 未注入 | 用 `./run_tests.sh` 运行，或确认 conftest.py 生效 |
| `ImportError: gptqmodel` | 相关测试依赖 mock，未正确注入 | 检查测试文件头部的 `_build_fake_gptqmodel_module` 是否被调用 |
| 测试卡在 GPU 相关错误 | 环境变量污染 | 确认未设置 `CUDA_VISIBLE_DEVICES`，测试应纯 CPU 运行 |

---

## 6. 与端到端用例的关系

- **单元测试**（本文件）：验证代码逻辑正确性，离线、快速（秒级），适合频繁回归
- **端到端部署用例**：`cases/v100/awq_1cat/deploy_all.sh`（方案 B）、`cases/v100/gptq_vllm085/deploy_all.sh`（方案 A），
  需要真实 GPU + 模型 + 服务，用于验证完整链路
- **领域评测数据**：`data/custom_data/`（容器内 `/volume/datahub/custom_data/`），用于量化精度验证

三者构成完整测试金字塔：单元测试 → 端到端部署 → 领域精度评测。
