#!/usr/bin/env python3
"""
HuggingFace 模型 / 数据集下载工具 (镜像加速)

用途:
    - 预下载校准数据集到本地缓存, 避免量化时才联网
    - 预下载模型权重 (可选)
    - 容器内默认走 hf-mirror.com, 缓存到 /volume/hf_cache

用法:
    # 下载校准数据集 (量化前预拉取)
    python llm_deploy/hf_download.py --dataset neuralmagic/LLM_compression_calibration \\
        --save_dir /volume/hf_cache

    # 下载模型
    python llm_deploy/hf_download.py --model Qwen/Qwen2.5-7B-Instruct \\
        --save_dir /volume/models

    # 用官方站点 (不走镜像)
    python llm_deploy/hf_download.py --dataset <name> --save_dir <dir> --use_mirror False

注: 改造自容器内 /volume/workspace/hf_download.py (作者 Xiaojian Yuan),
    精简为函数化实现, 增加校准数据集校验与缓存目录约定。
"""

import argparse
import os
import sys


def setup_env(use_mirror: bool, use_hf_transfer: bool):
    """配置 HF 镜像与传输加速环境变量"""
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"[env] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    if use_hf_transfer:
        try:
            import hf_transfer  # noqa: F401
        except ImportError:
            print("[env] 安装 hf-transfer ...")
            os.system("pip install -U hf-transfer -q")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        print(f"[env] HF_HUB_ENABLE_HF_TRANSFER={os.environ['HF_HUB_ENABLE_HF_TRANSFER']}")


def ensure_hub():
    """确认 huggingface_hub 已安装"""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[env] 安装 huggingface_hub ...")
        os.system("pip install -U huggingface_hub -q")


def build_save_path(name: str, save_dir: str, repo_type: str) -> str:
    """根据仓库名构造本地 hub 缓存路径, 与 HF 命名约定一致 (复数形式)"""
    parts = name.split("/")
    if len(parts) > 1:
        subdir = f"{repo_type}s--{parts[0]}--{parts[1]}"
    else:
        subdir = f"{repo_type}s--{parts[0]}"
    return os.path.join(save_dir, "hub", subdir)


def download(name: str, save_dir: str, repo_type: str, token: str = "",
             include: str = "", exclude: str = ""):
    """拉取仓库内容到 HF hub 缓存目录, 与 load_dataset 共享缓存

    用 cache_dir (而非 local_dir) 让 snapshot_download 写入标准 hub 缓存,
    这样后续 load_dataset(name) 能直接命中, 不会重复下载。
    """
    from huggingface_hub import snapshot_download

    is_dataset = repo_type == "dataset"
    cache_dir = save_dir if save_dir else None
    hub_path = build_save_path(name, save_dir, repo_type) if save_dir else None

    print(f"[download] {repo_type}: {name}")
    print(f"[download] cache_dir: {cache_dir or '(HF 默认缓存)'}")
    if hub_path:
        print(f"[download] hub 路径: {hub_path}")

    kwargs = dict(
        repo_id=name,
        repo_type="dataset" if is_dataset else "model",
        cache_dir=cache_dir,
    )
    if token:
        kwargs["token"] = token
    if include:
        kwargs["allow_patterns"] = [include]
    if exclude:
        kwargs["ignore_patterns"] = [exclude]

    snapshot_download(**kwargs)
    print(f"[download] 完成: {hub_path or name}")
    return hub_path


def verify_calibration_dataset(name: str, save_dir: str):
    """校验校准数据集可被 datasets 正常加载, 返回行数与列名"""
    from datasets import load_dataset

    # save_dir 作为 HF_HOME, load_dataset 会自动在 save_dir/hub 下查找缓存
    if save_dir:
        os.environ.setdefault("HF_HOME", save_dir)
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(save_dir, "datasets"))
    print(f"[verify] 加载数据集: {name}")
    ds = load_dataset(name, split="train")

    print(f"[verify] 行数: {len(ds)}")
    print(f"[verify] 列: {ds.column_names}")
    return ds


def main():
    parser = argparse.ArgumentParser(description="HuggingFace 下载工具 (镜像加速)")
    parser.add_argument("--model", "-M", default=None,
                        help="模型名, 如 Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", "-D", default=None,
                        help="数据集名, 如 neuralmagic/LLM_compression_calibration")
    parser.add_argument("--save_dir", "-S", default=None,
                        help="保存目录, 如 /volume/hf_cache")
    parser.add_argument("--token", "-T", default=None, help="HF access token")
    parser.add_argument("--include", default=None, help="仅下载匹配的文件 (glob)")
    parser.add_argument("--exclude", default=None, help="排除匹配的文件 (glob)")
    parser.add_argument("--use_mirror", default="True", type=eval,
                        help="走 hf-mirror.com 镜像, 默认 True")
    parser.add_argument("--use_hf_transfer", default="True", type=eval,
                        help="启用 hf-transfer 加速, 默认 True")
    parser.add_argument("--verify", action="store_true",
                        help="下载数据集后校验可加载性与行数")
    args = parser.parse_args()

    if not args.model and not args.dataset:
        parser.error("请指定 --model 或 --dataset 之一")
    if args.model and args.dataset:
        parser.error("--model 与 --dataset 只能二选一")

    setup_env(args.use_mirror, args.use_hf_transfer)
    ensure_hub()

    if args.dataset:
        download(args.dataset, args.save_dir, "dataset",
                 token=args.token or "", include=args.include or "",
                 exclude=args.exclude or "")
        if args.verify:
            print("[verify] 校验数据集 ...")
            verify_calibration_dataset(args.dataset, args.save_dir)
    else:
        download(args.model, args.save_dir, "model",
                 token=args.token or "", include=args.include or "",
                 exclude=args.exclude or "")


if __name__ == "__main__":
    main()
