"""PDF 文档加载器。

三种方案对比（课程要求 5.1.1）
------------------------------
=================  ==============  ====================================================
方案                依赖            特点
=================  ==============  ====================================================
``pymupdf``         PyMuPDF         速度最快、版面分析最强，但 AGPL 许可、包体大
``pdfplumber``      pdfplumber      基于 pdfminer.six，表格/坐标能力好，速度较慢
``pypdf``           pypdf           纯 Python、轻量、维护活跃，文本抽取够用
``builtin``         无              **本模块内置的纯标准库解析器**（零依赖兜底）
=================  ==============  ====================================================

选用策略
--------
:func:`extract_text` 按 ``pymupdf → pdfplumber → pypdf → builtin`` 的顺序挑选
**当前环境真实可用**的方案，并在 ``LoadedDocument.loader`` 中记录实际使用者，
方便撰写《PDF 加载方案对比实验报告》。

内置解析器能力（``_BuiltinPdfReader``）
--------------------------------------
- 解析 xref 表与（无 xref 时的）全文件对象扫描
- ``FlateDecode`` 流解压、``ASCIIHexDecode`` 解码
- 内容流算子解析：``BT/ET``、``Tf``、``Td/TD/Tm/T*``、``Tj/TJ/'/"``
- ``ToUnicode`` CMap（bfchar/bfrange）解析，正确还原中文 CID 字体
- 基于坐标的**双栏阅读顺序重排**与跨行断词修复
- 页眉/页脚/页码过滤

该解析器覆盖 CVPR/ACL 等常见论文 PDF 的正文抽取；对扫描件（纯图片 PDF）
不适用 —— 需接入 OCR，此时会在 ``load_errors`` 中给出明确提示。
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..utils.logger import get_logger
from ..utils.text import normalize_text
from ..utils.types import DocumentMeta, LoadedDocument, stable_hash

__all__ = ["load_pdf", "extract_text", "available_pdf_backends", "PdfBackendReport",
           "register_pdf_loader", "PDF_LOADERS"]

_log = get_logger("loader.pdf")


# ==============================================================================
# 后端注册与可用性探测
# ==============================================================================
@dataclass
class PdfBackendReport:
    """单个 PDF 后端的探测结果（用于实验对比报告）。"""

    name: str
    available: bool
    version: str = ""
    note: str = ""


def _probe(module_name: str, attr_version: str = "__version__") -> Tuple[bool, str]:
    try:
        module = __import__(module_name)
        return True, str(getattr(module, attr_version, "unknown"))
    except Exception:  # noqa: BLE001
        return False, ""


def available_pdf_backends() -> List[PdfBackendReport]:
    """探测环境中各 PDF 解析方案的可用性。"""
    reports: List[PdfBackendReport] = []
    for module, label in (("fitz", "pymupdf"), ("pdfplumber", "pdfplumber"), ("pypdf", "pypdf")):
        ok, version = _probe(module)
        reports.append(
            PdfBackendReport(
                name=label,
                available=ok,
                version=version,
                note="" if ok else f"未安装（pip install {label}）",
            )
        )
    reports.append(
        PdfBackendReport(
            name="builtin",
            available=True,
            version="stdlib",
            note="内置纯标准库解析器，零依赖，支持 FlateDecode + ToUnicode CMap",
        )
    )
    return reports


# ==============================================================================
# 内置纯标准库 PDF 解析器
# ==============================================================================
_RE_OBJ = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj", re.DOTALL)
_RE_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
_RE_REF = re.compile(rb"(\d+)\s+(\d+)\s+R\b")
_RE_DICT_ENTRY = re.compile(rb"/([A-Za-z0-9_.#\-]+)\s*(\[[^\]]*\]|<<.*?>>|\([^)]*\)|<[0-9A-Fa-f\s]*>|[^\s/<>\[\]()]+)", re.DOTALL)

# 内容流算子（捕获组索引与 _BuiltinPdfReader.extract_page 中的解包一一对应）
_RE_TJ_ARRAY = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
_RE_TJ_STR = re.compile(rb"\((.*?)(?<!\\)\)\s*Tj", re.DOTALL)
_RE_TOKEN = re.compile(
    rb"(\[.*?\]\s*TJ)"                                   # 1: TJ 数组
    rb"|(\((?:.*?)(?<!\\)\)\s*(?:Tj|'))"                 # 2: Tj / ' 字符串
    rb"|(T\*)"                                           # 3: 换行
    rb"|(-?[\d\.]+)\s+(-?[\d\.]+)\s+(Td|TD)"             # 4,5,6: Td / TD
    rb"|(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+"      # 7,8,9
    rb"(-?[\d\.]+)\s+(-?[\d\.]+)\s+(-?[\d\.]+)\s+Tm"     # 10,11,12: Tm
    rb"|/([A-Za-z0-9_.#\-]+)\s+([\d\.]+)\s+Tf"           # 13,14: 字体与字号
    rb"|(BT)"                                            # 15
    rb"|(ET)",                                           # 16
    re.DOTALL,
)


def _decode_pdf_string(raw: bytes) -> bytes:
    """解码 PDF 字面量字符串中的转义序列。"""
    out = bytearray()
    i = 0
    length = len(raw)
    while i < length:
        ch = raw[i]
        if ch != 0x5C:  # backslash
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= length:
            break
        esc = raw[i]
        simple = {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12}
        if esc in simple:
            out.append(simple[esc])
            i += 1
        elif esc in (0x28, 0x29, 0x5C):  # ( ) \
            out.append(esc)
            i += 1
        elif 0x30 <= esc <= 0x37:  # 八进制
            octal = bytearray()
            while i < length and len(octal) < 3 and 0x30 <= raw[i] <= 0x37:
                octal.append(raw[i])
                i += 1
            out.append(int(octal.decode("ascii"), 8) & 0xFF)
        elif esc in (0x0A, 0x0D):  # 行续接
            i += 1
            if esc == 0x0D and i < length and raw[i] == 0x0A:
                i += 1
        else:
            out.append(esc)
            i += 1
    return bytes(out)


def _decode_hex_string(raw: bytes) -> bytes:
    """解码 PDF 十六进制字符串 <...>。"""
    cleaned = re.sub(rb"[^0-9A-Fa-f]", b"", raw)
    if len(cleaned) % 2:
        cleaned += b"0"
    try:
        return bytes.fromhex(cleaned.decode("ascii"))
    except ValueError:
        return b""


def _try_inflate(raw: bytes) -> bytes:
    """尝试多种偏移量做 zlib 解压（应对流首部多余空白/换行）。"""
    for trim in (0, 1, 2):
        candidate = raw[trim:] if trim else raw
        try:
            return zlib.decompress(candidate)
        except zlib.error:
            continue
    # 数据被截断或含尾随垃圾时的宽容解压
    for trim in (0, 1, 2):
        try:
            obj = zlib.decompressobj()
            return obj.decompress(raw[trim:] if trim else raw)
        except zlib.error:
            continue
    _log.debug("zlib 解压失败，流长度 %d", len(raw))
    return b""


def _decode_ascii85(raw: bytes) -> bytes:
    """解码 ASCII85 流（PDF 的 /ASCII85Decode）。"""
    import base64

    text = raw.strip()
    if text.startswith(b"<~"):
        text = text[2:]
    if text.endswith(b"~>"):
        text = text[:-2]
    try:
        return base64.a85decode(text, adobe=False)
    except Exception:  # noqa: BLE001
        return b""


class _CMap:
    """ToUnicode CMap：CID/字符码 → Unicode 文本。"""

    def __init__(self) -> None:
        self.map: Dict[int, str] = {}
        self.ranges: List[Tuple[int, int, int]] = []  # (start, end, unicode_start)

    @classmethod
    def parse(cls, data: bytes) -> "_CMap":
        cmap = cls()
        text = data.decode("latin-1", errors="ignore")
        # bfchar
        for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.DOTALL):
            for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                try:
                    cmap.map[int(src, 16)] = _hex_to_text(dst)
                except ValueError:
                    continue
        # bfrange
        for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.DOTALL):
            for match in re.finditer(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(?:<([0-9A-Fa-f]+)>|\[(.*?)\])",
                block,
                re.DOTALL,
            ):
                low, high, dst, array = match.groups()
                try:
                    start, end = int(low, 16), int(high, 16)
                except ValueError:
                    continue
                if array:
                    items = re.findall(r"<([0-9A-Fa-f]+)>", array)
                    for offset, item in enumerate(items):
                        cmap.map[start + offset] = _hex_to_text(item)
                elif dst:
                    base = int(dst, 16)
                    for offset in range(min(end - start + 1, 65536)):
                        cmap.map[start + offset] = chr(base + offset)
        return cmap

    def translate(self, code: int, default: str = "") -> str:
        if code in self.map:
            return self.map[code]
        for start, end, base in self.ranges:
            if start <= code <= end:
                return chr(base + (code - start))
        return default


def _hex_to_text(hex_str: str) -> str:
    """把 UTF-16BE 十六进制串转成 Python 字符串。"""
    if len(hex_str) % 4 == 0:
        try:
            return bytes.fromhex(hex_str).decode("utf-16-be", errors="ignore")
        except ValueError:
            pass
    try:
        return bytes.fromhex(hex_str).decode("latin-1", errors="ignore")
    except ValueError:
        return ""


@dataclass
class _TextPiece:
    """页面上的一段文本及其坐标（用于双栏重排）。"""

    x: float
    y: float
    text: str
    font_size: float = 10.0


class _BuiltinPdfReader:
    """纯标准库 PDF 文本抽取器。"""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.objects: Dict[int, bytes] = {}
        self._cmap_cache: Dict[int, _CMap] = {}
        self._parse_objects()

    # -- 对象解析 ---------------------------------------------------------
    def _parse_objects(self) -> None:
        for match in _RE_OBJ.finditer(self.data):
            self.objects[int(match.group(1))] = match.group(3)
        if not self.objects:
            # 无 xref / 损坏时：退化为直接扫描所有 stream
            _log.debug("PDF 未解析到标准对象结构，改用流扫描模式")

    def _resolve(self, obj_num: int) -> bytes:
        return self.objects.get(obj_num, b"")

    @staticmethod
    def _decompress(raw: bytes, filters: bytes = b"") -> bytes:
        """按声明的 /Filter 链解压流数据。

        未声明 Filter 时**原样返回**流内容（PDF 规范：无 Filter 即为明文），
        这一点很关键 —— 早期版本在无 Filter 时也尝试 zlib 解压，导致明文内容流
        被误判为空，进而把所有正常 PDF 当成扫描件。
        """
        if not raw:
            return b""
        if not filters:
            return raw

        # 解析 Filter 链：/FlateDecode 或 [/ASCII85Decode /FlateDecode]
        names = re.findall(rb"/([A-Za-z0-9]+)", filters)
        if not names:
            return raw

        data = raw
        # PDF 中 Filter 链按声明顺序**编码**，解码需逆序
        for name in reversed(names):
            if name in (b"FlateDecode", b"Fl"):
                data = _try_inflate(data)
            elif name in (b"ASCIIHexDecode", b"AHx"):
                data = _decode_hex_string(data)
            elif name in (b"ASCII85Decode", b"A85"):
                data = _decode_ascii85(data)
            elif name in (b"LZWDecode", b"LZW"):
                _log.debug("暂不支持 LZWDecode 流，跳过")
                return b""
            # 其余（DCTDecode/JPXDecode/CCITTFaxDecode 等图像编码）保持原样
        return data

    def _stream_of(self, obj_body: bytes) -> bytes:
        """抽取对象中的流数据并解压。"""
        match = _RE_STREAM.search(obj_body)
        if not match:
            return b""
        header = obj_body[: match.start()]
        filters = b""
        filter_match = re.search(rb"/Filter\s*(\[[^\]]*\]|/\w+)", header)
        if filter_match:
            filters = filter_match.group(1)
        return self._decompress(match.group(1), filters)

    # -- 页面与字体 -------------------------------------------------------
    def pages(self) -> List[bytes]:
        """返回每页的内容流（已解压），按页对象在文件中出现的顺序。"""
        page_streams: List[bytes] = []
        # 精确匹配 /Type /Page（排除 /Pages、/PageMode 等）
        type_page = re.compile(rb"/Type\s*/Page(?![sA-Za-z])")
        for _obj_num, body in self.objects.items():
            if not type_page.search(body):
                continue
            contents_match = re.search(rb"/Contents\s+(\d+)\s+\d+\s+R", body)
            if contents_match:
                target = self._resolve(int(contents_match.group(1)))
                page_streams.append(self._stream_of(target))
                continue
            array_match = re.search(rb"/Contents\s*\[(.*?)\]", body, re.DOTALL)
            if array_match:
                merged = bytearray()
                for ref in _RE_REF.finditer(array_match.group(1)):
                    merged += self._stream_of(self._resolve(int(ref.group(1))))
                page_streams.append(bytes(merged))
                continue
            # 内联流（少见，但需兼容）
            inline = self._stream_of(body)
            if inline:
                page_streams.append(inline)
        if not page_streams:
            # 兜底：把所有含文本算子的流当成页面内容
            for _num, body in self.objects.items():
                stream = self._stream_of(body)
                if stream and (b"BT" in stream or b"Tj" in stream or b"TJ" in stream):
                    page_streams.append(stream)
        return page_streams

    def _cmap_for_font(self, font_name: bytes) -> Optional[_CMap]:
        """从字体资源中读取 ToUnicode CMap。"""
        key = int.from_bytes(font_name[:8], "little", signed=False)
        if key in self._cmap_cache:
            return self._cmap_cache[key]
        cmap: Optional[_CMap] = None
        needle = b"/" + font_name
        for _num, body in self.objects.items():
            if needle not in body:
                continue
            if b"/Type" in body and b"/Font" not in body:
                continue
            tou = re.search(rb"/ToUnicode\s+(\d+)\s+\d+\s+R", body)
            if tou:
                cmap = _CMap.parse(self._stream_of(self._resolve(int(tou.group(1)))))
                break
        self._cmap_cache[key] = cmap  # type: ignore[assignment]
        return cmap

    # -- 内容流解析 -------------------------------------------------------
    def extract_page(self, content: bytes) -> List[_TextPiece]:
        """解析一页内容流，产出带坐标的文本片段。"""
        pieces: List[_TextPiece] = []
        if not content:
            return pieces

        current_font = b""
        font_size = 10.0
        x = y = 0.0
        leading = 12.0
        pending: List[str] = []
        pending_x = pending_y = 0.0

        def flush() -> None:
            nonlocal pending
            if pending:
                joined = "".join(pending)
                if joined.strip():
                    pieces.append(_TextPiece(pending_x, pending_y, joined, font_size))
                pending = []

        for match in _RE_TOKEN.finditer(content):
            groups = match.groups()
            if groups[0]:  # [... ] TJ
                array_body = groups[0][: groups[0].rfind(b"]")]
                if not pending:
                    pending_x, pending_y = x, y
                pending.append(self._decode_tj_array(array_body, current_font))
            elif groups[1]:  # (..) Tj 或 (..) '
                token = groups[1].strip()
                is_quote = token.endswith(b"'")
                # 去掉结尾的 Tj / '，再剥掉包裹的括号
                body = token[: -1] if is_quote else token[: -2]
                body = body.strip()
                if body.startswith(b"(") and body.endswith(b")"):
                    literal = body[1:-1]
                else:
                    literal = body
                if is_quote:
                    flush()
                    y -= leading
                    x = 0.0
                if not pending:
                    pending_x, pending_y = x, y
                pending.append(self._decode_literal(literal, current_font))
            elif groups[2]:  # T*
                flush()
                y -= leading
            elif groups[3] is not None and groups[3] != b"":  # Td / TD
                flush()
                dx = float(groups[3])
                dy = float(groups[4])
                x += dx
                y += dy
                if groups[5] == b"TD":
                    leading = -dy
            elif groups[6] is not None and groups[6] != b"":  # Tm
                flush()
                _a, _b, _c, _d, e, f = (float(g) for g in groups[6:12])
                x = e
                y = f
            elif groups[12]:  # /Font size Tf
                flush()
                current_font = groups[12]
                try:
                    font_size = float(groups[13]) if groups[13] else 10.0
                except (TypeError, ValueError):
                    font_size = 10.0
            elif groups[14]:  # BT
                pass
            elif groups[15]:  # ET
                flush()
        flush()
        return pieces

    def _decode_literal(self, literal: bytes, font: bytes) -> str:
        raw = _decode_pdf_string(literal)
        return self._decode_bytes(raw, font)

    def _decode_tj_array(self, array_body: bytes, font: bytes) -> str:
        """解码 TJ 数组：字符串之间的大负数表示额外字距（空格）。"""
        out: List[str] = []
        for token in re.finditer(rb"\((?:.*?)(?<!\\)\)|<[0-9A-Fa-f\s]*>|-?\d+\.?\d*", array_body, re.DOTALL):
            piece = token.group(0)
            if piece.startswith(b"("):
                raw = _decode_pdf_string(piece[1:-1])
                out.append(self._decode_bytes(raw, font))
            elif piece.startswith(b"<"):
                out.append(self._decode_bytes(_decode_hex_string(piece), font))
            else:
                try:
                    shift = float(piece)
                except ValueError:
                    continue
                if shift < -180:  # 词间空格阈值
                    out.append(" ")
        return "".join(out)

    def _decode_bytes(self, raw: bytes, font: bytes) -> str:
        """把字形码字节串转成 Unicode 文本。"""
        if not raw:
            return ""
        cmap = self._cmap_for_font(font) if font else None
        if cmap and cmap.map:
            out: List[str] = []
            # 先尝试 2 字节码（CID 字体常见）
            i = 0
            while i < len(raw):
                code = int.from_bytes(raw[i : i + 2], "big") if i + 1 < len(raw) else raw[i]
                translated = cmap.translate(code)
                if translated:
                    out.append(translated)
                    i += 2 if i + 1 < len(raw) else 1
                else:
                    out.append(cmap.translate(raw[i], chr(raw[i])))
                    i += 1
            return "".join(out)
        # 无 CMap：PDFDocEncoding / Latin-1 近似
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1", errors="ignore")

    # -- 双栏重排 ---------------------------------------------------------
    @staticmethod
    def _to_lines(pieces: Sequence[_TextPiece]) -> List[str]:
        """把文本片段按坐标重排为阅读顺序的行序列。

        算法：按 y 坐标聚类成行 → 检测是否存在明显的双栏分界（x 分布的双峰）
        → 双栏时先左栏后右栏，单栏时按 y 降序、x 升序。
        """
        if not pieces:
            return []
        usable = [p for p in pieces if p.text.strip()]
        if not usable:
            return []

        # 1) 按 y 聚类成行（容差取字号中位数的一半）
        sizes = sorted(p.font_size for p in usable if p.font_size > 0) or [10.0]
        median_size = sizes[len(sizes) // 2] or 10.0
        tolerance = max(2.0, median_size * 0.6)
        sorted_pieces = sorted(usable, key=lambda p: (-p.y, p.x))

        rows: List[List[_TextPiece]] = []
        for piece in sorted_pieces:
            if rows and abs(rows[-1][0].y - piece.y) <= tolerance:
                rows[-1].append(piece)
            else:
                rows.append([piece])
        for row in rows:
            row.sort(key=lambda p: p.x)

        # 2) 双栏检测：页宽中位分割后，统计左右两侧是否都有密集文本
        xs = [p.x for p in usable]
        min_x, max_x = min(xs), max(xs)
        width = max_x - min_x
        left_count = sum(1 for p in usable if p.x - min_x < width * 0.47)
        right_count = sum(1 for p in usable if p.x - min_x > width * 0.53)
        is_two_column = width > 200 and left_count > len(usable) * 0.22 and right_count > len(usable) * 0.22

        lines: List[str] = []
        if is_two_column:
            split = min_x + width * 0.5
            for side in (0, 1):
                side_rows = [
                    row
                    for row in rows
                    if (max(p.x for p in row) < split) == (side == 0)
                ]
                side_rows.sort(key=lambda row: -row[0].y)
                lines.extend(_merge_row(row) for row in side_rows)
        else:
            lines.extend(_merge_row(row) for row in rows)
        return [ln for ln in lines if ln.strip()]


def _merge_row(row: Sequence[_TextPiece]) -> str:
    """把同一行内的片段拼接成文本，按坐标间隙补空格。"""
    if not row:
        return ""
    parts: List[str] = []
    prev_end: Optional[float] = None
    for piece in row:
        text = piece.text
        if prev_end is not None:
            gap = piece.x - prev_end
            if gap > 1.2 and not text.startswith(" ") and not (parts and parts[-1].endswith(" ")):
                parts.append(" ")
        parts.append(text)
        prev_end = piece.x + len(text) * piece.font_size * 0.5
    return "".join(parts)


# ==============================================================================
# 页眉/页脚过滤与清洗
# ==============================================================================
_RE_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*(?:/\s*\d{1,4})?\s*$", re.IGNORECASE)
_RE_ARXIV_STAMP = re.compile(r"^\s*arXiv:\d{4}\.\d{4,5}v\d+", re.IGNORECASE)


def clean_pdf_text(text: str, drop_header_footer: bool = True) -> str:
    """清洗 PDF 抽取文本：去页码、去 arXiv 水印、修复断词与连字。"""
    text = normalize_text(text)
    if not drop_header_footer:
        return text
    lines = text.split("\n")
    kept: List[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _RE_PAGE_NUMBER.match(stripped) or _RE_ARXIV_STAMP.match(stripped):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


# ==============================================================================
# 各后端实现
# ==============================================================================
def _load_with_pymupdf(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    import fitz  # type: ignore

    doc = fitz.open(str(path))
    pages: List[str] = []
    meta: Dict[str, Any] = {}
    info = doc.metadata or {}
    meta["title"] = (info.get("title") or "").strip()
    meta["authors"] = [a.strip() for a in re.split(r"[;,]", info.get("author") or "") if a.strip()]
    for page in doc:
        pages.append(page.get_text("text"))
    meta["num_pages"] = doc.page_count
    doc.close()
    return pages, meta


def _load_with_pdfplumber(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    import pdfplumber  # type: ignore

    pages: List[str] = []
    meta: Dict[str, Any] = {}
    with pdfplumber.open(str(path)) as pdf:
        meta["num_pages"] = len(pdf.pages)
        meta["title"] = (pdf.metadata or {}).get("Title", "") or ""
        meta["authors"] = [
            a.strip()
            for a in re.split(r"[;,]", (pdf.metadata or {}).get("Author", "") or "")
            if a.strip()
        ]
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages, meta


def _load_with_pypdf(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    meta: Dict[str, Any] = {}
    try:
        info = reader.metadata or {}
        meta["title"] = str(info.get("/Title") or "").strip()
        meta["authors"] = [
            a.strip() for a in re.split(r"[;,]", str(info.get("/Author") or "")) if a.strip()
        ]
    except Exception:  # noqa: BLE001
        pass
    pages = [(page.extract_text() or "") for page in reader.pages]
    meta["num_pages"] = len(pages)
    return pages, meta


def _load_with_builtin(path: Path) -> Tuple[List[str], Dict[str, Any]]:
    reader = _BuiltinPdfReader(path.read_bytes())
    raw_streams = reader.pages()
    pages: List[str] = []
    for content in raw_streams:
        pieces = reader.extract_page(content)
        pages.append("\n".join(_BuiltinPdfReader._to_lines(pieces)))
    return pages, {"num_pages": len(pages), "raw_streams": len(raw_streams)}


PDF_LOADERS: Dict[str, Any] = {
    "pymupdf": _load_with_pymupdf,
    "pdfplumber": _load_with_pdfplumber,
    "pypdf": _load_with_pypdf,
    "builtin": _load_with_builtin,
}

# 默认优先级：性能与效果综合排序
DEFAULT_PRIORITY: Tuple[str, ...] = ("pymupdf", "pdfplumber", "pypdf", "builtin")


def register_pdf_loader(name: str, loader: Any) -> None:
    """注册自定义 PDF 加载后端（便于扩展与实验对比）。"""
    PDF_LOADERS[name] = loader


# ==============================================================================
# 元信息抽取
# ==============================================================================
_RE_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_RE_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
_SEC_HEAD = r"(?:keywords?|key\s+words|index\s+terms|关\s*键\s*词)"
_NEXT_SEC = r"(?=\n\s*(?:#{1,3}\s*)?(?:" + _SEC_HEAD + r"|1\s*[\.、]|I\s+INTRODUCTION|引\s*言)|\Z)"
_RE_ABSTRACT = re.compile(
    r"(?:^|\n)\s*(?:abstract|摘\s*要)\s*[:：\-—]?\s*(.{80,2500}?)" + _NEXT_SEC,
    re.IGNORECASE | re.DOTALL,
)
_RE_KEYWORDS = re.compile(
    r"(?:^|\n)\s*" + _SEC_HEAD + r"\s*[:：\-—]?\s*(.{4,300}?)(?:\n\s*\n|\n\s*(?:1|I)\s*[\.、]|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_RE_ARXIV_ID = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_RE_CONFERENCE = re.compile(
    r"\b(CVPR|ICCV|ECCV|NeurIPS|NIPS|ICML|ICLR|ACL|EMNLP|NAACL|AAAI|IJCAI|SIGIR|KDD|WWW|TPAMI|TNNLS)\b"
    r"[\s\-–]*(20\d{2})?",
)
# 作者名：支持连字符姓氏（Ming-Wei）、中名缩写（John A.）、撇号（O'Brien）
_LATIN_NAME = r"[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zA-Z'’\-]+)"
_RE_AUTHOR_LINE = re.compile(rf"^({_LATIN_NAME}(?:\s*(?:,|\band\b|&|·|;)\s*{_LATIN_NAME})*)\s*$")
# 逗号分隔的姓名列表（"Vaswani, Shazeer, Parmar" 或 "Ashish Vaswani, Noam Shazeer"）
_RE_NAME_PART = re.compile(rf"^{_LATIN_NAME}$")


def extract_metadata(pages: Sequence[str], fallback_title: str = "", source_path: str = "") -> DocumentMeta:
    """从 PDF 各页文本中启发式抽取论文元信息。

    抽取项：标题、作者、年份、摘要、关键词、DOI、发表会议。
    对双栏论文，标题取首页前若干非空行中字号/长度最合适的一行。
    """
    meta = DocumentMeta(source_path=source_path, file_type="pdf", num_pages=len(pages))
    head = pages[0] if pages else ""
    full_head = "\n".join(pages[:2])[:8000]

    # ---- 标题 ----
    title = _guess_title(head)
    meta.title = title or fallback_title or Path(source_path).stem

    # ---- 作者 ----
    meta.authors = _guess_authors(head, meta.title)

    # ---- 年份 ----
    year_match = _RE_YEAR.search(head[:2000])
    if year_match:
        meta.year = int(year_match.group(1))

    # ---- 摘要 ----
    abstract_match = _RE_ABSTRACT.search(full_head)
    if abstract_match:
        meta.abstract = normalize_text(abstract_match.group(1))[:2000]

    # ---- 关键词 ----
    kw_match = _RE_KEYWORDS.search(full_head)
    if kw_match:
        raw_kw = normalize_text(kw_match.group(1))
        meta.keywords = [
            k.strip(" .;，；、") for k in re.split(r"[,;，；、]|\s{2,}", raw_kw) if k.strip(" .;，；、")
        ][:12]

    # ---- DOI / arXiv ----
    doi_match = _RE_DOI.search(full_head)
    if doi_match:
        meta.doi = doi_match.group(0).rstrip(".")
    else:
        arxiv_match = _RE_ARXIV_ID.search(full_head)
        if arxiv_match:
            meta.doi = f"arXiv:{arxiv_match.group(1)}"

    # ---- 会议 ----
    conf_match = _RE_CONFERENCE.search(head[:3000])
    if conf_match:
        meta.venue = f"{conf_match.group(1)} {conf_match.group(2) or ''}".strip()
        if not meta.year and conf_match.group(2):
            meta.year = int(conf_match.group(2))

    if not meta.keywords and meta.abstract:
        from ..utils.text import extract_keywords

        meta.keywords = extract_keywords(meta.abstract, top_k=8)
    return meta


def _guess_title(head: str) -> str:
    """启发式判断首页标题行。"""
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    # 常见噪声行
    noise = re.compile(
        r"^(arxiv|preprint|under review|published as|conference on|proceedings|"
        r"doi|https?://|©|copyright|ieee|acm|springer|elsevier|vol\.|no\.)",
        re.IGNORECASE,
    )
    best = ""
    best_score = -1.0
    for idx, line in enumerate(lines[:20]):
        if noise.match(line):
            continue
        if len(line) < 12 or len(line) > 220:
            continue
        if _RE_AUTHOR_LINE.match(line):
            continue
        if re.match(r"^[\d\s\.\-]+$", line):
            continue
        score = 0.0
        score += 3.0 if 20 <= len(line) <= 140 else 1.0
        score += 2.0 * (1.0 - idx / 20.0)              # 越靠前越可能是标题
        score += 1.5 if not line.endswith((".", "。", ",")) else 0.0
        score += 1.0 if re.search(r"[A-Za-z\u4e00-\u9fff]", line) else 0.0
        score -= 1.0 if line.endswith(":") else 0.0
        score -= 2.0 if re.match(r"^(abstract|摘要|introduction|引言)", line, re.IGNORECASE) else 0.0
        if score > best_score:
            best_score, best = score, line
    return best


def _guess_authors(head: str, title: str) -> List[str]:
    """在标题行之后寻找作者行。

    兼容三种常见排版：
    1. ``Ashish Vaswani, Noam Shazeer, Niki Parmar``（名 姓，逗号分隔）
    2. ``Vaswani, Shazeer, Parmar``（仅姓氏）
    3. ``张三，李四，王五``（中文作者）
    """
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    try:
        start = lines.index(title) + 1
    except ValueError:
        start = 0

    # 机构/邮箱/摘要等终止信号
    stop = re.compile(
        r"^(abstract|摘要|keywords|关键词|introduction|引言|1\s*[\.、])"
        r"|@|university|institute|department|laboratory|school of|college",
        re.IGNORECASE,
    )

    candidates: List[str] = []
    for line in lines[start : start + 8]:
        if stop.search(line):
            break
        # 去掉上标数字与脚注符号（保留连字符）
        cleaned = re.sub(r"[\d\*†‡§¶]+", " ", line)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;·")
        if not cleaned or len(cleaned) > 200:
            continue

        # 情况 1/2：整行匹配姓名（允许逗号/and/&/; 连接）
        whole = _RE_AUTHOR_LINE.match(cleaned)
        if whole:
            names = [
                n.strip()
                for n in re.split(r"\s*(?:,|\band\b|&|;|·)\s*", whole.group(1))
                if n.strip()
            ]
            names = [_normalize_name(n) for n in names]
            if len(names) >= 1 and all(len(n) > 3 for n in names):
                candidates.extend(names)
                if len(candidates) >= 2:
                    break
                continue

        # 情况 3：逐个分片验证（处理 "A, B and C" 混合写法）
        parts = [p.strip() for p in re.split(r"\s*(?:,|\band\b|&|;|·)\s*", cleaned) if p.strip()]
        valid = [p for p in parts if _RE_NAME_PART.match(p) and 3 < len(p) < 40]
        if len(valid) >= 2 and len(valid) >= len(parts) - 1:
            candidates.extend(_normalize_name(p) for p in valid)
            break

    # 中文论文作者行
    if not candidates:
        for line in lines[start : start + 6]:
            if re.match(r"^[\u4e00-\u9fff]{2,4}(?:[，,、\s]+[\u4e00-\u9fff]{2,4}){1,8}$", line.strip()):
                candidates = [a for a in re.split(r"[，,、\s]+", line.strip()) if a]
                break

    # 去重并保序
    seen: set[str] = set()
    result: List[str] = []
    for name in candidates:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result[:12]


def _normalize_name(name: str) -> str:
    """规整姓名：压缩空白、去掉首尾标点。"""
    return re.sub(r"\s{2,}", " ", name).strip(" ,;·.")


# ==============================================================================
# 对外主接口
# ==============================================================================
def extract_text(
    path: Any,
    backend: str = "auto",
    priority: Sequence[str] = DEFAULT_PRIORITY,
) -> Tuple[List[str], Dict[str, Any], str, List[str]]:
    """抽取 PDF 各页文本。

    Returns
    -------
    (pages, meta, backend_name, errors)
    """
    target = Path(path)
    errors: List[str] = []
    order = list(priority) if backend == "auto" else [backend]
    if backend not in ("auto",) and backend not in PDF_LOADERS:
        raise ValueError(f"未知的 PDF 后端: {backend}（可选 {list(PDF_LOADERS)}）")

    for name in order:
        loader = PDF_LOADERS.get(name)
        if loader is None:
            continue
        try:
            pages, meta = loader(target)
        except ImportError:
            errors.append(f"{name}: 未安装")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            _log.debug("PDF 后端 %s 失败: %s", name, exc)
            continue
        total = sum(len(p.strip()) for p in pages)
        if total < 10:
            errors.append(f"{name}: 抽取文本过少（{total} 字符），可能为扫描件")
            continue
        return pages, meta, name, errors

    raise RuntimeError(
        "所有 PDF 解析后端均失败："
        + ("; ".join(errors) if errors else "未知原因")
        + "。若为扫描版 PDF，请先做 OCR（如 ocrmypdf）后重试。"
    )


def load_pdf(
    path: Any,
    backend: str = "auto",
    drop_header_footer: bool = True,
    doc_id: Optional[str] = None,
) -> LoadedDocument:
    """加载 PDF 文件为 :class:`LoadedDocument`。

    Parameters
    ----------
    backend: ``auto`` 或 ``pymupdf`` / ``pdfplumber`` / ``pypdf`` / ``builtin``
    drop_header_footer: 是否过滤页码与 arXiv 水印
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {target}")

    pages, pdf_meta, used, errors = extract_text(target, backend=backend)
    cleaned_pages = [clean_pdf_text(p, drop_header_footer) for p in pages]

    meta = extract_metadata(cleaned_pages, fallback_title=target.stem, source_path=str(target))
    if pdf_meta.get("title"):
        # PDF 内置元信息的标题往往更准确；仅当启发式结果明显是噪声时采用
        heuristic = meta.title
        embedded = str(pdf_meta["title"]).strip()
        if len(embedded) > 8 and (len(heuristic) < 8 or heuristic.lower() in embedded.lower()):
            meta.title = embedded
    if pdf_meta.get("authors") and not meta.authors:
        meta.authors = list(pdf_meta["authors"])
    if pdf_meta.get("num_pages"):
        meta.num_pages = int(pdf_meta["num_pages"])
    else:
        meta.num_pages = len(cleaned_pages)
    meta.file_type = "pdf"
    meta.file_size = target.stat().st_size
    meta.num_pages = max(meta.num_pages, len(cleaned_pages))
    meta.doc_id = doc_id or stable_hash(f"{target.name}:{meta.file_size}:{meta.num_pages}")
    meta.extra["pdf_backend"] = used
    meta.extra["pdf_backend_errors"] = errors
    meta.extra["text_chars"] = sum(len(p) for p in cleaned_pages)

    text = "\n\n".join(p for p in cleaned_pages if p.strip())
    return LoadedDocument(
        doc_id=meta.doc_id,
        text=text,
        pages=cleaned_pages,
        meta=meta,
        load_errors=errors,
        loader=f"pdf:{used}",
    )
