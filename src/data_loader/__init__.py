"""模块一 —— 多格式文档加载器。

对外主入口：

    from src.data_loader import DocumentLoader, BatchImporter

支持 PDF / DOCX / TXT / MD / CSV / JSON / BIB，并给出环境能力探测报告。
"""

from .docx_loader import docx_available, load_docx
from .loader import (
    SUPPORTED_EXTENSIONS,
    BatchImporter,
    DocumentLoader,
    ImportJob,
    ImportResult,
    LoaderError,
    loader_from_config,
)
from .pdf_loader import (
    PDF_LOADERS,
    available_pdf_backends,
    extract_metadata,
    load_pdf,
    register_pdf_loader,
)
from .text_loader import (
    TEXT_EXTENSIONS,
    detect_encoding,
    load_bibtex,
    load_csv,
    load_json_text,
    load_markdown,
    load_text,
    load_text_file,
)

__all__ = [
    "DocumentLoader", "BatchImporter", "ImportJob", "ImportResult",
    "LoaderError", "loader_from_config", "SUPPORTED_EXTENSIONS",
    "load_pdf", "extract_metadata", "available_pdf_backends", "register_pdf_loader", "PDF_LOADERS",
    "load_docx", "docx_available",
    "load_text", "load_text_file", "load_markdown", "load_bibtex", "load_csv",
    "load_json_text", "detect_encoding", "TEXT_EXTENSIONS",
]
