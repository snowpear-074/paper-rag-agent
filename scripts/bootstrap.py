#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""启动脚本 —— 首次运行时准备数据目录与样例语料。

用法::

    python scripts/bootstrap.py              # 准备目录 + 生成样例 PDF
    python scripts/bootstrap.py --index      # 额外建立索引（可直接开始提问）
    python scripts/bootstrap.py --check      # 只做环境自检

本脚本是幂等的：重复运行不会破坏已有数据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> int:
    """环境自检：Python 版本、必需依赖、可选依赖可用性。"""
    print("=" * 70)
    print("  环境自检")
    print("=" * 70)

    problems = 0
    print(f"  Python 版本      : {sys.version.split()[0]}", end="")
    if sys.version_info >= (3, 10):
        print("  ✔")
    else:
        print("  ✘ 需要 3.10 或更高")
        problems += 1

    try:
        import numpy

        print(f"  NumPy            : {numpy.__version__}  ✔")
    except ImportError:
        print("  NumPy            : 未安装  ✘（必需依赖，请执行 pip install numpy）")
        problems += 1

    optional = [
        ("pypdf", "PDF 解析增强"),
        ("fitz", "PyMuPDF 高速 PDF 解析"),
        ("pdfplumber", "PDF 表格抽取"),
        ("docx", "Word 解析增强"),
        ("jieba", "中文分词"),
        ("yaml", "YAML 配置解析"),
        ("openpyxl", "Excel 报告导出"),
        ("scipy", "BM25 稀疏矩阵加速"),
        ("sentence_transformers", "本地语义向量与重排序"),
    ]
    print("\n  可选依赖（不装也能完整运行）：")
    for module, purpose in optional:
        try:
            __import__(module)
            print(f"    ✔ {module:24s} {purpose}")
        except ImportError:
            print(f"    – {module:24s} {purpose}（未安装，将使用内置实现）")

    print("\n  数据目录：")
    from src.utils.config import load_config

    cfg = load_config()
    for key in ("data_dir", "upload_dir", "index_dir", "report_dir"):
        path = cfg.resolve_path(str(cfg.paths.get(key, "")))
        exists = "✔" if path.exists() else "–"
        print(f"    {exists} {path}")

    print("\n" + "=" * 70)
    if problems:
        print(f"  自检发现 {problems} 个问题，请先解决。")
    else:
        print("  自检通过，可以运行 `python run.py` 启动系统。")
    print("=" * 70)
    return problems


def bootstrap(with_index: bool = False) -> int:
    """准备目录、生成样例语料（可选）建立索引。"""
    from src.utils.config import load_config

    cfg = load_config()
    cfg.ensure_dirs()
    print("✔ 已创建数据目录")

    # 生成样例论文（若已存在则跳过）
    samples_dir = cfg.resolve_path(str(cfg.paths.data_dir)) / "samples"
    try:
        from tests.fixtures.make_sample_pdf import write_sample_pdfs

        existing = list(samples_dir.glob("*.pdf")) if samples_dir.exists() else []
        if len(existing) >= 3:
            print(f"✔ 样例论文已存在（{len(existing)} 篇），跳过生成")
        else:
            paths = write_sample_pdfs(samples_dir)
            print(f"✔ 已生成 {len(paths)} 篇样例论文 → {samples_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"! 生成样例论文失败（不影响使用真实论文）: {exc}")

    if not with_index:
        print("\n提示：加 --index 参数可立即为样例论文建立索引。")
        return 0

    print("\n正在建立索引…")
    from src.frontend import get_container

    container = get_container(cfg)
    container.initialize()

    def progress(info: dict) -> None:
        percent = float(info.get("percent", 0.0))
        filled = int(28 * percent / 100.0)
        bar = "█" * filled + "░" * (28 - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}%  {str(info.get('message', ''))[:46]:<46}")
        sys.stdout.flush()

    report = container.index_files([str(samples_dir)], on_progress=progress)
    print()
    print(f"✔ 索引完成：{report.get('num_documents', 0)} 篇文档，{report.get('num_chunks', 0)} 个片段")
    print("\n现在可以运行 `python run.py` 启动系统并开始提问。")
    return 0


def main() -> int:
    from src.utils.console import enable_utf8_output

    enable_utf8_output()
    parser = argparse.ArgumentParser(description="智能科研助理 —— 启动准备脚本")
    parser.add_argument("--check", action="store_true", help="只做环境自检")
    parser.add_argument("--index", action="store_true", help="为样例论文建立索引")
    args = parser.parse_args()

    if args.check:
        return check_environment()
    code = check_environment()
    if code:
        print("\n请先解决上述问题再继续。")
        return code
    return bootstrap(with_index=args.index)


if __name__ == "__main__":
    raise SystemExit(main())
