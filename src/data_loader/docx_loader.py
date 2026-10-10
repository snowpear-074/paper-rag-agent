"""Word（.docx）文档加载器。

实现要点
--------
- **不依赖 python-docx**：直接解析 OOXML（``word/document.xml``），
  用标准库 ``zipfile`` + 正则/轻量 XML 扫描完成抽取。若环境装有
  python-docx，可选启用其更完整的样式读取能力（见 ``prefer_python_docx``）。
- **保留文档结构**：标题（Heading N）转 Markdown ``#``，列表项转 ``-``，
  表格转 Markdown 表格 —— 这些结构对后续分块与检索质量有实质影响。
- **页码映射**：docx 无固定分页概念，按"每 N 个段落为一页"切分，
  保证引用溯源仍能给出稳定的段落区间定位。
- **公式保留**：OOXML 的 ``m:oMath`` 数学公式转为 ``$...$``，避免公式被丢弃。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..utils.logger import get_logger
from ..utils.text import normalize_text
from ..utils.types import DocumentMeta, LoadedDocument, stable_hash

__all__ = ["load_docx", "docx_available"]

_log = get_logger("loader.docx")

# 每多少个"逻辑段落"折算为一页（用于引用页码定位）
_PARAGRAPHS_PER_PAGE = 40

_RE_HEADING = re.compile(r'w:val="Heading(\d)"')
_RE_STYLE = re.compile(r'<w:pStyle[^>]*w:val="([^"]+)"')
# 大纲级别：<w:outlineLvl w:val="0"/> 表示一级标题（0-based）
_RE_OUTLINE = re.compile(r'<w:outlineLvl[^>]*w:val="(\d+)"')
# 中文版 Word 的标题样式名（"标题 1" / "標題 1"）
_RE_ZH_HEADING = re.compile(r"^(?:标题|標題)\s*(\d)")
_RE_PARA = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.DOTALL)
_RE_TABLE = re.compile(r"<w:tbl>.*?</w:tbl>", re.DOTALL)
_RE_ROW = re.compile(r"<w:tr[ >].*?</w:tr>", re.DOTALL)
_RE_CELL = re.compile(r"<w:tc>.*?</w:tc>", re.DOTALL)
_RE_TEXT = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.DOTALL)
_RE_TAB = re.compile(r"<w:tab\s*/>")
_RE_BREAK = re.compile(r"<w:br\s*/>")
_RE_OMML = re.compile(r"<m:oMath[ >].*?</m:oMath>", re.DOTALL)
_RE_OMML_TEXT = re.compile(r"<m:t(?: [^>]*)?>(.*?)</m:t>", re.DOTALL)
_RE_LIST = re.compile(r'<w:numId w:val="(\d+)"')
_RE_INSTR = re.compile(r"<w:instrText[^>]*>(.*?)</w:instrText>", re.DOTALL)


def docx_available() -> bool:
    """检测 python-docx 是否可用。"""
    try:  # pragma: no cover
        import docx  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _unescape(text: str) -> str:
    """XML 实体反转义。"""
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )


def _paragraph_text(xml_fragment: str) -> str:
    """从 ``<w:p>`` 片段中提取纯文本（含公式、制表符、软换行）。"""
    parts: List[str] = []
    # 位置感知地扫描：公式优先整体替换
    cursor = 0
    for match in _RE_OMML.finditer(xml_fragment):
        parts.append(_plain_run_text(xml_fragment[cursor : match.start()]))
        formula = "".join(_RE_OMML_TEXT.findall(match.group(0)))
        if formula.strip():
            parts.append(f" ${_unescape(formula).strip()}$ ")
        cursor = match.end()
    parts.append(_plain_run_text(xml_fragment[cursor:]))
    text = "".join(parts)
    # 清理域代码残留（如目录域、超链接域）
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _plain_run_text(fragment: str) -> str:
    """提取普通 ``<w:t>`` 文本，并把 tab/br 还原为字符。"""
    out = fragment
    out = _RE_TAB.sub("\t", out)
    out = _RE_BREAK.sub("\n", out)
    # 删除域的指令文本，避免 "HYPERLINK "http://..."" 混入正文
    out = _RE_INSTR.sub("", out)
    return _unescape("".join(_RE_TEXT.findall(out)))


def _load_style_names(path: Path) -> Dict[str, str]:
    """读取 ``word/styles.xml``，返回 ``{styleId: 样式名}`` 映射。

    必要性：Word 文档的 ``<w:pStyle w:val="...">`` 里存的是**样式 ID**，而非样式名。
    英文版 Word 常用 ``Heading1``，但 WPS / 中文版 Word 常用纯数字 ID（如 ``"2"``），
    此时只能靠 ``styles.xml`` 才能知道 ``"2"`` 到底是不是标题样式。
    """
    mapping: Dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            if "word/styles.xml" not in archive.namelist():
                return mapping
            xml = archive.read("word/styles.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError, OSError):
        return mapping

    for match in re.finditer(r"<w:style\b[^>]*>(.*?)</w:style>", xml, re.DOTALL):
        block = match.group(1)
        style_id = re.search(r'w:styleId="([^"]+)"', match.group(0))
        name = re.search(r'<w:name[^>]*w:val="([^"]+)"', block)
        if style_id and name:
            mapping[style_id.group(1)] = name.group(1)
        elif style_id:
            mapping.setdefault(style_id.group(1), "")
    return mapping


def _paragraph_style(xml_fragment: str, style_names: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
    """返回 (样式名, 标题级别)。非标题级别为 0。

    判定优先级：

    1. ``<w:pStyle w:val="HeadingN">``（英文版 Word / python-docx 导出）
    2. 样式名经 ``styles.xml`` 解析后匹配 ``Heading N`` / ``标题 N``
    3. ``<w:outlineLvl w:val="N">``（0-based，直接给出大纲级别）——
       WPS 与中文版 Word 常只写大纲级别而不写标题样式名
    4. 样式名本身形如 ``标题1`` / ``Title``
    """
    style_match = _RE_STYLE.search(xml_fragment)
    style_id = style_match.group(1) if style_match else ""

    # 1) 直接是 HeadingN
    heading = _RE_HEADING.search(xml_fragment)
    if heading:
        return style_id or f"Heading{heading.group(1)}", int(heading.group(1))

    # 2) 通过 styles.xml 解析样式 ID → 样式名
    style_name = style_id
    if style_id and style_names:
        style_name = style_names.get(style_id, style_id)
    resolved = _level_from_style_name(style_name)
    if resolved:
        return style_name, resolved

    # 3) 大纲级别（0-based → 1-based）
    outline = _RE_OUTLINE.search(xml_fragment)
    if outline:
        level = int(outline.group(1))
        # outlineLvl=9 是"正文文本"的约定值，不视为标题
        if 0 <= level <= 8:
            return style_name or style_id, level + 1

    # 4) 其它特殊样式
    lowered = (style_name or "").lower()
    if lowered in ("title", "subtitle"):
        return style_name, 1 if lowered == "title" else 2
    return style_name or style_id, 0


def _level_from_style_name(style_name: str) -> int:
    """样式名 → 标题级别（0 表示不是标题）。"""
    if not style_name:
        return 0
    text = style_name.strip()
    en = re.match(r"^heading\s*(\d)", text, re.IGNORECASE)
    if en:
        return int(en.group(1))
    zh = _RE_ZH_HEADING.match(text)
    if zh:
        return int(zh.group(1))
    return 0


def _table_to_markdown(table_xml: str) -> str:
    """把 ``<w:tbl>`` 转为 Markdown 表格。"""
    rows: List[List[str]] = []
    for row_xml in _RE_ROW.findall(table_xml):
        cells = [
            normalize_text(_paragraph_text(cell)).replace("\n", " ")
            for cell in _RE_CELL.findall(row_xml)
        ]
        if any(c.strip() for c in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    lines = []
    for idx, row in enumerate(rows):
        padded = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if idx == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def _extract_document_xml(path: Path) -> str:
    """读取 ``word/document.xml``。"""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        target = "word/document.xml"
        if target not in names:
            candidates = [n for n in names if n.endswith("document.xml")]
            if not candidates:
                raise ValueError(f"{path.name} 不是有效的 docx（缺少 word/document.xml）")
            target = candidates[0]
        return archive.read(target).decode("utf-8", errors="ignore")


def _extract_core_properties(path: Path) -> Dict[str, str]:
    """读取 docProps/core.xml 中的标题/作者/关键词等元信息。"""
    props: Dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            if "docProps/core.xml" not in archive.namelist():
                return props
            xml = archive.read("docProps/core.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError):
        return props
    for tag, key in (
        ("dc:title", "title"),
        ("dc:creator", "author"),
        ("cp:lastModifiedBy", "last_author"),
        ("dc:subject", "subject"),
        ("cp:keywords", "keywords"),
        ("dc:description", "description"),
    ):
        match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
        if match:
            props[key] = _unescape(match.group(1)).strip()
    return props


def _blocks_in_order(xml: str) -> List[Tuple[str, str]]:
    """按文档顺序返回顶层块：('p', xml) 或 ('tbl', xml)。

    只看 body 的直接子元素，避免嵌套表格/段落被重复计入。
    """
    body_match = re.search(r"<w:body>(.*)</w:body>", xml, re.DOTALL)
    body = body_match.group(1) if body_match else xml
    blocks: List[Tuple[str, str]] = []
    cursor = 0
    pattern = re.compile(r"<w:tbl>|<w:p[ >]|<w:p/>", re.DOTALL)
    while True:
        match = pattern.search(body, cursor)
        if not match:
            break
        if match.group(0).startswith("<w:tbl"):
            end = _find_matching(body, match.start(), "<w:tbl", "</w:tbl>")
            if end == -1:
                cursor = match.end()
                continue
            blocks.append(("tbl", body[match.start() : end]))
            cursor = end
        else:
            end = _find_matching(body, match.start(), "<w:p", "</w:p>", self_closing=True)
            if end == -1:
                cursor = match.end()
                continue
            blocks.append(("p", body[match.start() : end]))
            cursor = end
    return blocks


def _find_matching(
    text: str, start: int, open_tag: str, close_tag: str, self_closing: bool = False
) -> int:
    """从 ``start`` 起找到配对闭合标签的结束位置（返回值含闭合标签）。

    正确处理三类情况：

    1. **自闭合标签**（``<w:p/>``）—— 不增加嵌套深度；
    2. **嵌套同名标签**（表格单元格里的 ``<w:p>``）；
    3. **前缀相同的不同标签** —— 找 ``<w:p`` 时绝不能把 ``<w:pPr>``、
       ``<w:pStyle>`` 当成 ``<w:p>``。这是必须用**词边界**而非简单字符串
       前缀匹配的原因：早期实现漏了这一点，导致所有带 ``<w:pPr>`` 的段落
       （即所有带样式的段落，包括全部标题）在平衡扫描中被判定为"标签未闭合"
       而被整段丢弃。

    解析失败（标签未闭合）时返回 -1，由调用方安全跳过。
    """
    if self_closing and text.startswith(f"{open_tag}/>", start):
        return start + len(open_tag) + 2

    # 标签名后必须紧跟空白、'>' 或 '/'，避免匹配到 <w:pPr> 之类的前缀标签
    name = re.escape(open_tag)
    token = re.compile(rf"{name}(?=[\s/>])(?:[^>]*?/>|[^>]*>)", re.DOTALL)

    depth = 0
    pos = start
    while pos < len(text):
        next_open = token.search(text, pos)
        next_close = text.find(close_tag, pos)
        if next_close == -1:
            return -1
        if next_open is not None and next_open.start() < next_close:
            # 自闭合标签不增加深度
            if not next_open.group(0).rstrip().endswith("/>"):
                depth += 1
            pos = next_open.end()
            continue
        depth -= 1
        pos = next_close + len(close_tag)
        if depth <= 0:
            return pos
    return -1


def load_docx(
    path: Any,
    prefer_python_docx: bool = False,
    paragraphs_per_page: int = _PARAGRAPHS_PER_PAGE,
    doc_id: Optional[str] = None,
) -> LoadedDocument:
    """加载 .docx 文档。

    Parameters
    ----------
    prefer_python_docx:
        为 True 且环境装有 python-docx 时，改用其解析（对复杂样式支持更好）。
    paragraphs_per_page:
        逻辑分页粒度，用于把段落序号映射成"页码"以便引用溯源。
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Word 文件不存在: {target}")

    errors: List[str] = []
    loader_name = "docx:builtin"

    if prefer_python_docx and docx_available():
        try:
            text, pages = _load_via_python_docx(target, paragraphs_per_page)
            props = _extract_core_properties(target)
            loader_name = "docx:python-docx"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"python-docx 解析失败，回退内置解析器: {exc}")
            text, pages, props = _load_via_ooxml(target, paragraphs_per_page)
    else:
        text, pages, props = _load_via_ooxml(target, paragraphs_per_page)

    meta = _build_meta(target, text, pages, props, errors)
    meta.extra["docx_loader"] = loader_name
    return LoadedDocument(
        doc_id=meta.doc_id,
        text=text,
        pages=pages,
        meta=meta,
        load_errors=errors,
        loader=loader_name,
    )


def _load_via_python_docx(path: Path, paragraphs_per_page: int) -> Tuple[str, List[str]]:
    import docx  # type: ignore

    document = docx.Document(str(path))
    lines: List[str] = []
    for paragraph in document.paragraphs:
        content = paragraph.text.strip()
        style = (paragraph.style.name or "") if paragraph.style else ""
        if not content:
            lines.append("")
            continue
        heading = re.match(r"Heading (\d)", style) or re.match(r"标题\s*(\d)", style)
        if heading:
            level = int(heading.group(1))
            lines.append(f"{'#' * min(level + 1, 6)} {content}")
        elif "List" in style or "列表" in style:
            lines.append(f"- {content}")
        else:
            lines.append(content)
    for table in document.tables:
        rows = [[cell.text.strip().replace("\n", " ") for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        width = max(len(r) for r in rows)
        lines.append("")
        for idx, row in enumerate(rows):
            padded = row + [""] * (width - len(row))
            lines.append("| " + " | ".join(padded) + " |")
            if idx == 0:
                lines.append("|" + "---|" * width)
        lines.append("")
    text = normalize_text("\n".join(lines))
    return text, _paginate(lines, paragraphs_per_page)


def _load_via_ooxml(path: Path, paragraphs_per_page: int) -> Tuple[str, List[str], Dict[str, str]]:
    xml = _extract_document_xml(path)
    props = _extract_core_properties(path)
    style_names = _load_style_names(path)
    lines: List[str] = []
    for kind, fragment in _blocks_in_order(xml):
        if kind == "tbl":
            table_md = _table_to_markdown(fragment)
            if table_md:
                lines.extend(["", table_md, ""])
            continue
        content = _paragraph_text(fragment)
        if not content:
            lines.append("")
            continue
        _style, level = _paragraph_style(fragment, style_names)
        if level:
            lines.append(f"{'#' * min(level + 1, 6)} {content}")
        elif _RE_LIST.search(fragment):
            lines.append(f"- {content}")
        else:
            lines.append(content)
    text = normalize_text("\n".join(lines))
    return text, _paginate(lines, paragraphs_per_page), props


def _paginate(lines: Sequence[str], paragraphs_per_page: int) -> List[str]:
    """把行序列按固定行数切成"逻辑页"。"""
    size = max(1, paragraphs_per_page)
    pages: List[str] = []
    buffer: List[str] = []
    count = 0
    for line in lines:
        buffer.append(line)
        if line.strip() and not line.startswith("#"):
            count += 1
        if count >= size:
            pages.append("\n".join(buffer).strip())
            buffer, count = [], 0
    if buffer:
        pages.append("\n".join(buffer).strip())
    return [p for p in pages] or [""]


def _build_meta(
    path: Path,
    text: str,
    pages: Sequence[str],
    props: Dict[str, str],
    errors: List[str],
) -> DocumentMeta:
    """从正文与文档属性构造元信息。"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    title = props.get("title", "").strip()

    if not title:
        # 优先取第一个 Markdown 标题
        for line in lines[:12]:
            if line.startswith("#"):
                title = line.lstrip("# ").strip()
                break
    if not title:
        # 再取首个长度合适的非空行
        for line in lines[:8]:
            if 8 <= len(line) <= 160 and not line.startswith(("|", "-")):
                title = line
                break
    if not title:
        title = path.stem

    authors = [
        a.strip()
        for a in re.split(r"[;,，；、]", props.get("author", "") or "")
        if a.strip()
    ]
    keywords = [
        k.strip() for k in re.split(r"[,;，；、]", props.get("keywords", "") or "") if k.strip()
    ]

    year_match = re.search(r"\b(19[89]\d|20[0-4]\d)\b", text[:1500])
    year = int(year_match.group(1)) if year_match else None

    abstract = ""
    abstract_match = re.search(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:abstract|摘\s*要)\s*[:：]?\s*(.{80,2000}?)(?=\n\s*(?:#{1,3}\s*)?(?:keywords?|关键词|1\s*[\.、]|一、|引言)|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract = normalize_text(abstract_match.group(1))[:2000]

    meta = DocumentMeta(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        keywords=keywords,
        doi="",
        source_path=str(path),
        file_type="docx",
        file_size=path.stat().st_size,
        num_pages=len(pages),
        language=_guess_language(text),
    )
    meta.extra["text_chars"] = len(text)
    meta.extra["core_properties"] = props
    if errors:
        meta.extra["load_warnings"] = errors
    meta.doc_id = stable_hash(f"{path.name}:{meta.file_size}:{title}")
    return meta


def _guess_language(text: str) -> str:
    """粗略判断文档主语言（中/英）。"""
    sample = text[:3000]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if cjk == 0:
        return "en"
    if latin == 0:
        return "zh"
    return "zh" if cjk / (cjk + latin / 2) > 0.5 else "en"
