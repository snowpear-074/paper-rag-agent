"""统一文档加载入口 + 批量导入管理。

对应课程要求 5.1.1 的"批量文档导入与状态追踪（进度显示、失败重试机制）"。

组件
----
:class:`DocumentLoader`   按扩展名分派到具体加载器，统一异常处理
:class:`ImportJob`        单次批量导入的状态快照（供前端进度条消费）
:class:`BatchImporter`    批量导入 + 重试 + 进度回调 + 失败清单
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..utils.logger import get_logger
from ..utils.types import LoadedDocument, new_id
from .docx_loader import docx_available, load_docx
from .pdf_loader import available_pdf_backends, load_pdf
from .text_loader import TEXT_EXTENSIONS, load_text_file

__all__ = [
    "DocumentLoader",
    "BatchImporter",
    "ImportJob",
    "ImportResult",
    "SUPPORTED_EXTENSIONS",
    "loader_from_config",
]

_log = get_logger("loader")

SUPPORTED_EXTENSIONS: frozenset = frozenset(
    {".pdf", ".docx", ".doc"} | TEXT_EXTENSIONS
)


class LoaderError(RuntimeError):
    """文档加载失败。"""


# ==============================================================================
# 单文件加载分派
# ==============================================================================
class DocumentLoader:
    """多格式文档加载器（策略模式 + 统一异常包装）。"""

    def __init__(
        self,
        pdf_backend: str = "auto",
        drop_header_footer: bool = True,
        prefer_python_docx: bool = False,
        text_strategy: Optional[str] = None,
    ) -> None:
        self.pdf_backend = pdf_backend
        self.drop_header_footer = drop_header_footer
        self.prefer_python_docx = prefer_python_docx
        self.text_strategy = text_strategy

    def supported(self, path: Any) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def load(self, path: Any, doc_id: Optional[str] = None) -> LoadedDocument:
        """加载单个文档；失败抛 :class:`LoaderError`（附原始堆栈摘要）。"""
        target = Path(path)
        if not target.exists():
            raise LoaderError(f"文件不存在: {target}")
        if target.stat().st_size == 0:
            raise LoaderError(f"文件为空: {target.name}")

        suffix = target.suffix.lower()
        try:
            if suffix == ".pdf":
                return load_pdf(
                    target,
                    backend=self.pdf_backend,
                    drop_header_footer=self.drop_header_footer,
                    doc_id=doc_id,
                )
            if suffix in (".docx", ".doc"):
                if suffix == ".doc":
                    raise LoaderError(
                        "不支持旧版 .doc 格式，请先用 Word 另存为 .docx 后重试"
                    )
                return load_docx(
                    target,
                    prefer_python_docx=self.prefer_python_docx,
                    doc_id=doc_id,
                )
            if suffix in TEXT_EXTENSIONS:
                return load_text_file(target, strategy=self.text_strategy, doc_id=doc_id)
        except LoaderError:
            raise
        except Exception as exc:  # noqa: BLE001
            detail = traceback.format_exc(limit=3).splitlines()[-3:]
            raise LoaderError(
                f"加载 {target.name} 失败: {type(exc).__name__}: {exc} | {' / '.join(detail)}"
            ) from exc

        raise LoaderError(
            f"不支持的文件类型: {suffix}（支持 {sorted(SUPPORTED_EXTENSIONS)}）"
        )

    def capability_report(self) -> Dict[str, Any]:
        """输出当前环境的加载能力清单（用于 /api/health 与实验报告）。"""
        return {
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "pdf_backends": [
                {"name": r.name, "available": r.available, "version": r.version, "note": r.note}
                for r in available_pdf_backends()
            ],
            "python_docx": docx_available(),
            "docx_backend": "python-docx" if (self.prefer_python_docx and docx_available()) else "builtin-ooxml",
            "text_strategy": self.text_strategy or "auto-by-suffix",
        }


# ==============================================================================
# 批量导入
# ==============================================================================
@dataclass
class ImportResult:
    """单个文件的导入结果。"""

    path: str
    ok: bool
    doc_id: str = ""
    title: str = ""
    num_pages: int = 0
    num_chars: int = 0
    loader: str = ""
    error: str = ""
    attempts: int = 1
    latency_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    # 成功时的文档对象（不参与序列化，避免进度事件里塞入全文）
    document: Optional[LoadedDocument] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "doc_id": self.doc_id,
            "title": self.title,
            "num_pages": self.num_pages,
            "num_chars": self.num_chars,
            "loader": self.loader,
            "error": self.error,
            "attempts": self.attempts,
            "latency_ms": round(self.latency_ms, 1),
            "warnings": self.warnings,
        }


@dataclass
class ImportJob:
    """批量导入任务的状态快照（前端进度条消费）。"""

    job_id: str = field(default_factory=lambda: new_id("job_"))
    total: int = 0
    finished: int = 0
    succeeded: int = 0
    failed: int = 0
    current_file: str = ""
    stage: str = "pending"            # pending | running | done | error
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    results: List[ImportResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def percent(self) -> float:
        return round(100.0 * self.finished / self.total, 1) if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total": self.total,
            "finished": self.finished,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "percent": self.percent,
            "current_file": self.current_file,
            "stage": self.stage,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 2),
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors[:20],
        }


ProgressCallback = Callable[[ImportJob, ImportResult], None]


class BatchImporter:
    """批量文档导入器：进度回调 + 失败重试 + 结果汇总。

    Parameters
    ----------
    loader: :class:`DocumentLoader` 实例
    max_retries: 单文件最大重试次数（0 表示不重试）
    """

    def __init__(self, loader: Optional[DocumentLoader] = None, max_retries: int = 1) -> None:
        self.loader = loader or DocumentLoader()
        self.max_retries = max(0, int(max_retries))

    def collect_files(self, root: Any) -> List[Path]:
        """收集单个文件或目录下的所有受支持文件（递归）。"""
        target = Path(root)
        if target.is_file():
            return [target] if self.loader.supported(target) else []
        if not target.is_dir():
            return []
        return sorted(
            p for p in target.rglob("*") if p.is_file() and self.loader.supported(p)
        )

    def import_documents(
        self,
        sources: Iterable[Any],
        on_progress: Optional[ProgressCallback] = None,
        on_loaded: Optional[Callable[[LoadedDocument], None]] = None,
    ) -> tuple[ImportJob, List[LoadedDocument]]:
        """批量导入。

        Parameters
        ----------
        sources: 文件路径列表（可混合目录）
        on_progress: 每完成一个文件回调一次，用于前端进度推送
        on_loaded: 每成功加载一个文档回调一次（可用于"边导入边建索引"）

        Returns
        -------
        (job, documents) —— 失败的文件记录在 job.results 中，不中断整体流程。
        """
        files: List[Path] = []
        for source in sources:
            files.extend(self.collect_files(source))

        job = ImportJob(total=len(files), stage="running")
        documents: List[LoadedDocument] = []

        if not files:
            job.stage = "done"
            job.finished_at = time.time()
            job.errors.append("没有找到受支持的文件（支持 PDF/DOCX/TXT/MD/CSV/BIB/JSON）")
            return job, documents

        for path in files:
            job.current_file = path.name
            result = self._import_one(path)

            if result.ok:
                job.succeeded += 1
            else:
                job.failed += 1
                job.errors.append(f"{path.name}: {result.error}")
                _log.warning("导入失败 %s: %s", path.name, result.error)

            job.finished += 1
            job.results.append(result)
            if on_progress:
                try:
                    on_progress(job, result)
                except Exception as exc:  # noqa: BLE001 - 回调异常不应影响导入
                    _log.warning("进度回调异常: %s", exc)

        # 第二轮：把成功的文档统一交给 on_loaded（先全部解析完再建索引，失败隔离更清晰）
        for result in job.results:
            if not result.ok or result.document is None:
                continue
            loaded = result.document
            if on_loaded:
                try:
                    on_loaded(loaded)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("文档回调异常 %s: %s", result.path, exc)
            documents.append(loaded)

        job.stage = "done"
        job.finished_at = time.time()
        job.current_file = ""
        _log.info(
            "批量导入完成：成功 %d / 失败 %d，耗时 %.1fs",
            job.succeeded,
            job.failed,
            job.finished_at - job.started_at,
        )
        return job, documents

    def _import_one(self, path: Path) -> ImportResult:
        """加载单个文件，失败时按策略重试。"""
        start = time.perf_counter()
        last_error = ""
        for attempt in range(1, self.max_retries + 2):
            try:
                document = self.loader.load(path)
                return ImportResult(
                    path=str(path),
                    ok=True,
                    doc_id=document.doc_id,
                    title=document.meta.title,
                    num_pages=document.meta.num_pages,
                    num_chars=len(document.text),
                    loader=document.loader,
                    attempts=attempt,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    warnings=list(document.load_errors),
                    document=document,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt <= self.max_retries:
                    _log.info("重试 (%d/%d): %s", attempt, self.max_retries, path.name)
                    time.sleep(0.3 * attempt)
        return ImportResult(
            path=str(path),
            ok=False,
            error=last_error,
            attempts=self.max_retries + 1,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


# ==============================================================================
# 工厂
# ==============================================================================
def loader_from_config(config: Optional[Any] = None, **overrides: Any) -> DocumentLoader:
    """按配置创建文档加载器。

    读取 ``chunking.paper_optimize.*`` 与 ``paths`` 段的相关开关。
    """
    from ..utils.config import get_config

    cfg = config or get_config()
    optimize = cfg.chunking.paper_optimize
    kwargs: Dict[str, Any] = {
        "pdf_backend": str(cfg.get("data_loader.pdf_backend", "auto")),
        "drop_header_footer": bool(optimize.get("remove_references", True)),
        "prefer_python_docx": bool(cfg.get("data_loader.prefer_python_docx", False)),
        "text_strategy": cfg.get("data_loader.text_strategy") or None,
    }
    kwargs.update(overrides)
    return DocumentLoader(**kwargs)
