# -*- coding: utf-8 -*-
"""测试公共工具：项目导入路径、临时配置、样例语料。

所有测试模块统一从这里取配置与样例数据，保证：

1. 测试之间**互不污染**（每个用例独立的临时数据目录）；
2. 不依赖网络与外部服务（LLM 走 mock，Embedding 走 hash，向量库走 numpy）；
3. 结果**可复现**（固定随机性来源：hash 向量化 + 确定性 mock LLM）。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 项目根目录加入 sys.path（使 `import src.xxx` 在任意工作目录下都可用）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def temp_dir(prefix: str = "ragtest_") -> Path:
    """创建独立临时目录（调用方负责清理，或交给 :class:`TempWorkspace`）。"""
    return Path(tempfile.mkdtemp(prefix=prefix))


class TempWorkspace:
    """临时工作区：提供隔离的数据目录与配置。

    用法::

        with TempWorkspace() as ws:
            cfg = ws.config()
            engine = create_retrieval_engine(cfg)
    """

    def __init__(self, name: str = "ws") -> None:
        self.root = temp_dir()
        self.name = name

    def __enter__(self) -> "TempWorkspace":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.cleanup()
        return False

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def dir(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def config(self, **overrides: Any) -> Any:
        """构造测试配置：全部走零依赖后端，数据落在本工作区。"""
        from src.utils.config import load_config

        base: Dict[str, Any] = {
            "llm": {"provider": "mock", "model": "mock-model", "temperature": 0.0, "max_retries": 1},
            "embedding": {"provider": "hash", "dimension": 256, "batch_size": 16},
            "vector_store": {
                "backend": "numpy",
                "persist_dir": str(self.dir("vector_store")),
                "collection": "test",
                "distance": "cosine",
            },
            # 关键：让 embedding 缓存与向量库分离，避免跨用例串味
            "paths": {
                "data_dir": str(self.dir("data")),
                "upload_dir": str(self.dir("uploads")),
                "index_dir": str(self.dir("index")),
                "report_dir": str(self.dir("reports")),
            },
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 300,
                "chunk_overlap": 40,
                "min_chunk_size": 40,
            },
            "retrieval": {"top_k": 5, "candidate_k": 10, "rrf_k": 60, "low_relevance_threshold": 0.25},
            "reranker": {"provider": "heuristic"},
            "generation": {
                "max_context_chars": 3000,
                "max_chunk_chars": 800,
                "cache": {"enabled": True, "path": str(self.path("cache", "semantic.db"))},
            },
            "agent": {
                "max_iterations": 6,
                "tool_timeout": 10,
                "max_retries_per_tool": 1,
                "loop_detect_window": 2,
            },
            "memory": {"backend": "sqlite", "path": str(self.path("memory", "chat.db"))},
            "observability": {
                "log_dir": str(self.dir("logs")),
                "level": "ERROR",
                "trace_file": str(self.path("logs", "traces.jsonl")),
                "metrics_file": str(self.path("logs", "metrics.jsonl")),
                "console_color": False,
            },
            "server": {"host": "127.0.0.1", "port": 0},
            "web_search": {"enabled": False},
        }
        merged = _deep_merge(base, overrides)
        return load_config(None, merged)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# 样例语料
# ---------------------------------------------------------------------------
SAMPLE_PDF_NAMES = ("sample_transformer.pdf", "sample_resnet.pdf", "sample_bert.pdf")


def ensure_sample_pdfs(directory: Optional[Path] = None) -> List[Path]:
    """生成（或复用）3 篇样例论文 PDF，返回路径列表。"""
    target = Path(directory) if directory else (PROJECT_ROOT / "data" / "samples")
    target.mkdir(parents=True, exist_ok=True)
    existing = [target / name for name in SAMPLE_PDF_NAMES]
    if all(path.exists() and path.stat().st_size > 1000 for path in existing):
        return existing
    from tests.fixtures.make_sample_pdf import write_sample_pdfs

    return write_sample_pdfs(target)


def sample_documents(config: Any = None) -> List[Any]:
    """加载 3 篇样例论文为 :class:`LoadedDocument` 列表。"""
    from src.data_loader import DocumentLoader

    paths = ensure_sample_pdfs()
    loader = DocumentLoader(pdf_backend="builtin")
    return [loader.load(path) for path in paths]


def build_test_text() -> str:
    """构造一段可控的中英文混合测试文本（用于分块/检索单测）。"""
    return (
        "# 注意力机制综述\n\n"
        "注意力机制最早由 Bahdanau 等人于 2015 年提出，用于改进神经机器翻译。\n"
        "其核心思想是在解码每一步时，动态地为编码器的各个位置分配权重。\n\n"
        "## 自注意力\n\n"
        "自注意力（Self-Attention）把同一个序列同时当作查询、键和值。\n"
        "计算复杂度为 O(n^2 d)，其中 n 是序列长度，d 是表示维度。\n\n"
        "## 多头注意力\n\n"
        "多头注意力把表示空间划分为 h 个子空间并行计算，最后拼接。\n"
        "实验表明 h=8 时效果最好，每个头的维度为 d_k = d_v = d/h = 64。\n\n"
        "## 实验结果\n\n"
        "在 WMT 2014 英德翻译任务上取得 28.4 BLEU，比此前最好结果高 2.0 BLEU。\n"
    )


def build_mock_agent_config(ws: "TempWorkspace", **overrides: Any) -> Any:
    """构造用于 Agent 测试的配置（缩短超时、减少轮次，加快测试）。"""
    return ws.config(
        agent={
            "max_iterations": 4,
            "tool_timeout": 5,
            "max_retries_per_tool": 1,
            "enable_router": True,
            "enable_parallel": True,
        },
        **overrides,
    )
