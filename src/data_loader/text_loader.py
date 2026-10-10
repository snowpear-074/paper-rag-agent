"""纯文本类文档加载器：.txt / .md / .markdown / .rst / .csv / .json / .bib。

设计要点
--------
- **编码自动探测**：按 ``BOM → utf-8 → utf-8-sig → gbk → gb18030 → big5 →
  latin-1`` 顺序试解码，覆盖国内学术写作常见的 GBK/GB18030 文本文件。
- **Markdown 结构保留**：把 ``#`` 标题层级原样保留（对分块边界识别有价值），
  同时剥离代码围栏标记噪音、保留表格。
- **BibTeX 支持**：把 ``.bib`` 文献条目解析成"一篇论文一段文本"，
  便于快速搭建知识库（无需真实 PDF）。
- **分页策略**：Markdown 按一级/二级标题切"页"（引用溯源到章节），
  纯文本按固定行数切"页"。
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.logger import get_logger
from ..utils.text import normalize_text
from ..utils.types import DocumentMeta, LoadedDocument, stable_hash

__all__ = ["load_text", "load_markdown", "load_bibtex", "detect_encoding",
           "TEXT_EXTENSIONS", "STRATEGY_BY_SUFFIX"]

_log = get_logger("loader.text")

TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".bib"}

# 各后缀对应的解析策略
STRATEGY_BY_SUFFIX: Dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "markdown",
    ".bib": "bibtex",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
}

_LINES_PER_PAGE = 45
_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "utf-16", "latin-1")


# ==============================================================================
# 编码探测
# ==============================================================================
def detect_encoding(data: bytes) -> Tuple[str, str]:
    """探测字节串编码。

    Returns
    -------
    (encoding, text) —— 已成功解码的文本。
    """
    if not data:
        return "utf-8", ""
    # BOM 优先
    for bom, enc in (
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    ):
        if data.startswith(bom):
            try:
                return enc, data.decode(enc)
            except UnicodeDecodeError:
                break
    for enc in _ENCODINGS:
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        # GBK 系解码常把 UTF-8 中文变成乱码但"能解码成功"，用替换符比例校验
        if enc in ("gb18030", "gbk", "big5", "latin-1"):
            if _garble_ratio(text) > 0.02:
                continue
        return enc, text
    return "utf-8", data.decode("utf-8", errors="replace")


def _garble_ratio(text: str) -> float:
    """估算乱码比例：出现 U+FFFD 或大量罕见私有区/控制字符时判为乱码。"""
    sample = text[:4000]
    if not sample:
        return 0.0
    bad = sum(
        1
        for ch in sample
        if ch == "\ufffd" or (0xE000 <= ord(ch) <= 0xF8FF) or (ord(ch) < 32 and ch not in "\n\r\t")
    )
    return bad / len(sample)


# ==============================================================================
# Markdown / 纯文本
# ==============================================================================
_RE_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_RE_MD_FENCE = re.compile(r"^```.*?$", re.MULTILINE)


def _strip_markdown_noise(text: str) -> str:
    """轻度清洗：去掉代码围栏标记、图片语法，保留标题/列表/表格结构。"""
    text = _RE_MD_FENCE.sub("", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)          # 图片 -> alt
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)     # 链接保留 URL
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # 分隔线
    return text


def _paginate_markdown(text: str) -> List[str]:
    """按一/二级标题切分为"逻辑页"，无标题时退化为固定行数分页。"""
    lines = text.split("\n")
    pages: List[str] = []
    buffer: List[str] = []
    found_heading = False
    for line in lines:
        is_top = bool(re.match(r"^#{1,2}\s+\S", line))
        if is_top and buffer and any(l.strip() for l in buffer):
            pages.append("\n".join(buffer).strip())
            buffer = []
            found_heading = True
        buffer.append(line)
    if buffer:
        pages.append("\n".join(buffer).strip())
    pages = [p for p in pages if p.strip()]
    if not found_heading or len(pages) < 2:
        return _paginate_plain(text, _LINES_PER_PAGE)
    return [p.strip() for p in pages]


def _paginate_plain(text: str, lines_per_page: int = _LINES_PER_PAGE) -> List[str]:
    """按固定行数分页（对纯文本是唯一可行的页码近似）。"""
    lines = text.split("\n")
    pages: List[str] = []
    for start in range(0, len(lines), max(1, lines_per_page)):
        chunk = "\n".join(lines[start : start + lines_per_page]).strip()
        if chunk:
            pages.append(chunk)
    return pages or [""]


def _guess_text_meta(path: Path, text: str) -> Dict[str, Any]:
    """从纯文本/Markdown 正文中猜测标题、年份、摘要等。"""
    info: Dict[str, Any] = {}
    md_title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if md_title:
        info["title"] = md_title.group(1).strip()
    else:
        for line in text.split("\n")[:6]:
            stripped = line.strip()
            if 8 <= len(stripped) <= 160 and not stripped.startswith(("|", ">", "-", "*")):
                info["title"] = stripped
                break
    year = re.search(r"\b(19[89]\d|20[0-4]\d)\b", text[:1500])
    if year:
        info["year"] = int(year.group(1))
    abstract = re.search(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:abstract|摘\s*要)\s*[:：]?\s*(.{60,2000}?)(?=\n\s*(?:#{1,3}\s*)?(?:keywords?|关键词)|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract:
        info["abstract"] = normalize_text(abstract.group(1))[:2000]
    return info


def load_markdown(path: Any, doc_id: Optional[str] = None) -> LoadedDocument:
    """加载 Markdown / RST 文档（按标题分页）。"""
    target = Path(path)
    encoding, raw = detect_encoding(target.read_bytes())
    text = normalize_text(_strip_markdown_noise(raw))
    pages = _paginate_markdown(text)
    info = _guess_text_meta(target, text)

    meta = _base_meta(target, text, pages, info, file_type="md")
    meta.extra.update({"encoding": encoding, "strategy": "markdown"})
    meta.doc_id = doc_id or stable_hash(f"{target.name}:{meta.file_size}:{meta.title}")
    return LoadedDocument(doc_id=meta.doc_id, text=text, pages=pages, meta=meta,
                          loader=f"text:markdown({encoding})")


def load_text(path: Any, doc_id: Optional[str] = None) -> LoadedDocument:
    """加载纯文本文件（按固定行数分页）。"""
    target = Path(path)
    encoding, raw = detect_encoding(target.read_bytes())
    text = normalize_text(raw)
    pages = _paginate_plain(text, _LINES_PER_PAGE)
    info = _guess_text_meta(target, text)

    meta = _base_meta(target, text, pages, info, file_type="txt")
    meta.extra.update({"encoding": encoding, "strategy": "plain"})
    meta.doc_id = doc_id or stable_hash(f"{target.name}:{meta.file_size}:{meta.title}")
    return LoadedDocument(doc_id=meta.doc_id, text=text, pages=pages, meta=meta,
                          loader=f"text:plain({encoding})")


# ==============================================================================
# CSV / JSON
# ==============================================================================
def load_csv(path: Any, max_rows: int = 5000, doc_id: Optional[str] = None) -> LoadedDocument:
    """加载 CSV/TSV，转成 Markdown 表格文本（可用于问答实验与评测集查看）。"""
    target = Path(path)
    encoding, raw = detect_encoding(target.read_bytes())
    delimiter = "\t" if target.suffix.lower() == ".tsv" else ","
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows: List[List[str]] = []
    for idx, row in enumerate(reader):
        if idx >= max_rows:
            break
        rows.append([cell.strip() for cell in row])
    if not rows:
        raise ValueError(f"CSV 文件为空: {target}")

    width = max(len(r) for r in rows)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * width)
    text = normalize_text("\n".join(lines))

    meta = _base_meta(target, text, [text], {"title": target.stem}, file_type="csv")
    meta.extra.update({"encoding": encoding, "rows": len(rows), "columns": width})
    meta.doc_id = doc_id or stable_hash(f"{target.name}:{meta.file_size}:{len(rows)}")
    return LoadedDocument(doc_id=meta.doc_id, text=text, pages=[text], meta=meta, loader="text:csv")


def load_json_text(path: Any, doc_id: Optional[str] = None) -> LoadedDocument:
    """加载 JSON 文件（对象数组时逐条渲染为文本，评测集文件常用）。"""
    target = Path(path)
    encoding, raw = detect_encoding(target.read_bytes())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {exc}") from exc

    if isinstance(data, list):
        lines = [json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item for item in data]
    elif isinstance(data, dict):
        lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in data.items()]
    else:
        lines = [str(data)]
    text = normalize_text("\n".join(lines))
    pages = _paginate_plain(text, _LINES_PER_PAGE)

    meta = _base_meta(target, text, pages, {"title": target.stem}, file_type="json")
    meta.extra.update({"encoding": encoding, "records": len(lines)})
    meta.doc_id = doc_id or stable_hash(f"{target.name}:{meta.file_size}:{len(lines)}")
    return LoadedDocument(doc_id=meta.doc_id, text=text, pages=pages, meta=meta, loader="text:json")


# ==============================================================================
# BibTeX
# ==============================================================================
_RE_BIB_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.DOTALL | re.IGNORECASE)
_RE_BIB_FIELD = re.compile(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)", re.DOTALL)


def _clean_bib_value(raw: str) -> str:
    value = raw.strip().rstrip(",").strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    elif value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    value = re.sub(r"\s+", " ", value)
    value = value.replace("{", "").replace("}", "")
    return value.strip()


def load_bibtex(path: Any, doc_id: Optional[str] = None) -> LoadedDocument:
    """加载 .bib 文献库：每个条目渲染为一段结构化文本。"""
    target = Path(path)
    encoding, raw = detect_encoding(target.read_bytes())

    entries: List[Dict[str, str]] = []
    for match in _RE_BIB_ENTRY.finditer(raw):
        entry_type, key, body = match.group(1).lower(), match.group(2).strip(), match.group(3)
        fields: Dict[str, str] = {"entry_type": entry_type, "cite_key": key}
        for field_match in _RE_BIB_FIELD.finditer(body):
            fields[field_match.group(1).lower()] = _clean_bib_value(field_match.group(2))
        entries.append(fields)

    if not entries:
        # 不是标准 bib，退化为纯文本
        _log.warning("%s 未解析出 BibTeX 条目，按纯文本处理", target.name)
        return load_text(target, doc_id=doc_id)

    blocks: List[str] = []
    for fields in entries:
        title = fields.get("title", "（无标题）")
        authors = fields.get("author", "").replace(" and ", ", ")
        year = fields.get("year", "")
        venue = fields.get("journal") or fields.get("booktitle") or ""
        lines = [f"# {title}"]
        if authors:
            lines.append(f"作者: {authors}")
        if year:
            lines.append(f"年份: {year}")
        if venue:
            lines.append(f"发表: {venue}")
        for key in ("doi", "url", "abstract", "keywords", "pages", "volume", "publisher"):
            if fields.get(key):
                lines.append(f"{key}: {fields[key]}")
        blocks.append("\n".join(lines))

    text = normalize_text("\n\n".join(blocks))
    pages = _paginate_markdown(text)

    first = entries[0]
    info = {
        "title": f"文献库 {target.stem}（{len(entries)} 条）",
        "year": int(first["year"]) if first.get("year", "").isdigit() else None,
        "abstract": "",
    }
    meta = _base_meta(target, text, pages, info, file_type="bib")
    meta.extra.update({"encoding": encoding, "num_entries": len(entries), "entries": entries[:200]})
    meta.doc_id = doc_id or stable_hash(f"{target.name}:{meta.file_size}:{len(entries)}")
    return LoadedDocument(doc_id=meta.doc_id, text=text, pages=pages, meta=meta, loader="text:bibtex")


# ==============================================================================
# 统一入口
# ==============================================================================
def _base_meta(
    path: Path,
    text: str,
    pages: Sequence[str],
    info: Dict[str, Any],
    file_type: str,
) -> DocumentMeta:
    return DocumentMeta(
        title=str(info.get("title") or path.stem),
        authors=list(info.get("authors") or []),
        year=info.get("year"),
        abstract=str(info.get("abstract") or ""),
        keywords=list(info.get("keywords") or []),
        source_path=str(path),
        file_type=file_type,
        file_size=path.stat().st_size,
        num_pages=max(1, len(pages)),
        language=_guess_language(text),
        extra={"text_chars": len(text)},
    )


def _guess_language(text: str) -> str:
    sample = text[:3000]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if cjk == 0:
        return "en"
    if latin == 0:
        return "zh"
    return "zh" if cjk / (cjk + latin / 2) > 0.5 else "en"


def load_text_file(
    path: Any,
    strategy: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> LoadedDocument:
    """按后缀自动选择策略加载文本类文件。

    Parameters
    ----------
    strategy: 强制指定 ``plain`` / ``markdown`` / ``bibtex`` / ``csv`` / ``json``
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"文件不存在: {target}")
    chosen = strategy or STRATEGY_BY_SUFFIX.get(target.suffix.lower(), "plain")
    if chosen == "markdown":
        return load_markdown(target, doc_id=doc_id)
    if chosen == "bibtex":
        return load_bibtex(target, doc_id=doc_id)
    if chosen == "csv":
        return load_csv(target, doc_id=doc_id)
    if chosen == "json":
        return load_json_text(target, doc_id=doc_id)
    return load_text(target, doc_id=doc_id)
