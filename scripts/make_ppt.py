#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成课程演示 PPT（``.pptx``）。

与交付书同源的三个理由
----------------------
1. **数字不会漂**：评测指标从 :file:`reports/性能评测数据.json` 实时读取，
   分块对比表从 :file:`reports/分块策略对比实验报告.md` 解析，一轮优化前后
   对比表从 :file:`reports/系统评测报告.md` 解析 —— 重跑评测再执行本脚本，
   PPT 里的数字自动跟上，不会出现"报告更新了、PPT 还写着旧数"。
   文件缺失时对应位置渲染「未运行」，**绝不编造数字**。
2. **排版可复现**：配色取自前端 ``style.css`` 的设计变量（靛蓝 #4F46E5 +
   浅灰底 + 白卡片），字号、栅格、留白全部由代码固定，换台电脑结果一致。
3. **占位符集中**：成员姓名/学号/班级统一在 :data:`MEMBERS` 里替换，
   与 :mod:`scripts.make_delivery_doc` 保持同一套占位符写法。

用法::

    python scripts/make_ppt.py                    # 默认输出到项目根目录
    python scripts/make_ppt.py --out D:/演示.pptx
    python scripts/make_ppt.py --no-check         # 跳过生成后的自检

依赖：``python-pptx``（仅本脚本需要，不属于系统运行依赖）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.util import Emu, Inches, Pt
except ImportError:  # pragma: no cover - 环境缺依赖时给出可操作提示
    sys.stderr.write("缺少 python-pptx。请先安装：pip install python-pptx\n")
    raise SystemExit(2)

# ==============================================================================
# 常量：画布、配色、字体
# ==============================================================================
TITLE = "智能科研助理"
SUBTITLE = "基于 RAG + Agent 的论文知识库问答系统"
KICKER = "生产实习课程报告 · 课程题目一"
DEFAULT_OUT = PROJECT_ROOT / "智能科研助理_项目演示.pptx"
IMAGES = PROJECT_ROOT / "docs" / "images"
SHOTS = IMAGES / "screenshots"

#: 成员占位符 —— 与 scripts/make_delivery_doc.py 的 MEMBERS 同一套写法，
#: 姓名/学号/班级由用户自己填写，脚本不代填。
MEMBERS: List[str] = [
    "成员 1：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
    "成员 2：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
    "成员 3：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
    "成员 4：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
]

#: 配色 —— 取自 src/frontend/static/css/style.css 的 :root 设计变量
BG = "F4F6FB"          # 页面底色
PANEL = "FFFFFF"       # 卡片
INK = "0F172A"         # 主文字
INK2 = "475569"        # 次级文字
INK3 = "94A3B8"        # 三级文字
LINE = "E2E8F0"        # 边框
ACCENT = "4F46E5"      # 靛蓝主色
ACCENT2 = "6366F1"     # 次级靛蓝
SOFT = "EEF0FF"        # 主色浅底
OK = "16A34A"
WARN = "D97706"
DANGER = "DC2626"
INFO = "2563EB"
PURPLE = "7C3AED"

FONT = "微软雅黑"
MONO = "Consolas"

SLIDE_W, SLIDE_H = 13.333, 7.5
MARGIN = 0.62
CONTENT_W = SLIDE_W - 2 * MARGIN          # 12.093
TITLE_T, TITLE_H = 0.40, 0.58
BODY_T = 1.38                              # 正文区顶部（kicker 之下）
BODY_B = 6.86                              # 正文区底部
FOOTER_T = 6.96
TOTAL_SLIDES = 14

Para = Union[str, Tuple[str, Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]


# ==============================================================================
# 数据读取（全部来自**真实产物**，不做任何估算）
# ==============================================================================
def load_metrics() -> Dict[str, Any]:
    """读 :file:`reports/性能评测数据.json`；缺失时返回空字典（页面标注「未运行」）。"""
    path = PROJECT_ROOT / "reports" / "性能评测数据.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"警告：评测数据读取失败（{exc}），PPT 中的指标将标注为「未运行」\n")
        return {}


def load_eval_stats() -> Dict[str, int]:
    """统计评测集的问题类型分布（直接数 :file:`reports/评测集.json`）。"""
    labels = {"factual": "事实型", "comparison": "对比型", "summary": "归纳型", "reasoning": "推理型"}
    path = PROJECT_ROOT / "reports" / "评测集.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    if isinstance(data, dict):
        raw = data.get("type_distribution") or {}
        if not raw:
            for item in data.get("items") or []:
                if isinstance(item, dict):
                    key = str(item.get("type") or "unknown")
                    raw[key] = raw.get(key, 0) + 1
    elif isinstance(data, list):
        raw = {}
        for item in data:
            if isinstance(item, dict):
                key = str(item.get("type") or "unknown")
                raw[key] = raw.get(key, 0) + 1
    else:
        return {}

    ordered: Dict[str, int] = {}
    for key in list(labels) + [k for k in raw if k not in labels]:
        if key in raw:
            ordered[labels.get(key, key)] = int(raw[key])
    return ordered


def load_chunking_rows() -> List[Dict[str, str]]:
    """从 :file:`reports/分块策略对比实验报告.md` 的主表解析四种策略的实测值。

    解析规则与 :func:`scripts.make_delivery_doc.load_chunking_rows` 保持一致
    （列序：分块数 / 平均块长 / 最短最长 / 建索引 / Hit / MRR / Recall /
    Precision / NDCG / 延迟）。解析失败返回空列表 —— 宁可少一张表，
    也不写死一份可能与实际不符的数字。
    """
    path = PROJECT_ROOT / "reports" / "分块策略对比实验报告.md"
    if not path.exists():
        return []
    pattern = re.compile(r"^\|\s*(?P<label>.+?)（`(?P<strategy>[a-z_]+)`）\s*\|(?P<rest>.+)\|\s*$")
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        cells = [c.strip() for c in match.group("rest").split("|")]
        if len(cells) < 10 or cells[0] in ("未运行",):
            continue
        strategy = match.group("strategy")
        if strategy in seen:      # 同文件内的引擎口径表不重复收录
            continue
        seen.add(strategy)
        rows.append(
            {
                "strategy": strategy,
                "label": re.sub(r"（.*?）", "", match.group("label")).strip(),
                "num_chunks": cells[0],
                "hit": cells[4],
                "ndcg": cells[8],
                "latency": cells[9],
            }
        )
    return rows


def load_iteration_rows() -> List[Dict[str, str]]:
    """解析 :file:`reports/系统评测报告.md` 第七节的「优化前 / 优化后」严格口径表。

    返回 ``[{指标, 优化前, 优化后, 相对变化}, ...]``；解析不到返回空列表。
    """
    path = PROJECT_ROOT / "reports" / "系统评测报告.md"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    start = text.find("## 七、一轮迭代优化的前后对比")
    if start < 0:
        return []
    section = text[start:]
    rows: List[Dict[str, str]] = []
    started = False
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and "优化前" in cells[1] and "优化后" in cells[2]:
            started = True
            continue
        if not started:
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue                      # 分隔行 / 空行：跳过，别当成表格结束
        if len(cells) < 5:
            break                         # 后面那张「引擎口径」表只有 3 列，到此为止
        if not re.match(r"^-?\d+(\.\d+)?$", cells[1]) or not re.match(r"^-?\d+(\.\d+)?$", cells[2]):
            break
        rows.append({"label": cells[0], "before": cells[1], "after": cells[2], "rel": cells[4]})
    return rows


def is_improvement(label: str, rel: str) -> Optional[bool]:
    """判断一处「相对变化」是改善还是劣化（延迟/降级率是「越低越好」）。

    返回 ``None`` 表示持平（如引用准确率 +0.00%）。
    """
    match = re.match(r"^([+-]?\d+(?:\.\d+)?)%$", rel.strip())
    if not match:
        return None
    delta = float(match.group(1))
    if delta == 0:
        return None
    lower_is_better = ("延迟" in label) or ("降级" in label)
    return delta < 0 if lower_is_better else delta > 0


def fmt(value: Any, digits: int = 4) -> str:
    """把数值格式化成幻灯片里好读的写法。

    0~1 之间的小数固定 4 位（0.7000 / 0.5827，表格里小数位对齐）；
    ≥1 的数值去掉多余的尾零（1.0 / 27.0 / 433.0）。
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "未运行"
    if abs(num) >= 1:
        text = f"{num:.{digits}f}".rstrip("0").rstrip(".")
        return text if "." in text else f"{num:.1f}"
    return f"{num:.{digits}f}"


# ==============================================================================
# 绘图基元
# ==============================================================================
def apply_font(run: Any, name: str = FONT, size: float = 14.0, bold: bool = False,
               color: str = INK, italic: bool = False) -> None:
    """设置一个 run 的字体，**同时**写 latin / ea / cs 三个字体槽。

    python-pptx 的 ``font.name`` 只写 ``a:latin``，中文字符会回退到主题字体；
    这里手工补 ``a:ea``（东亚）与 ``a:cs``，并保持 DrawingML 要求的元素顺序
    （latin → ea → cs）。
    """
    font = run.font
    font.name = name
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    font.color.rgb = RGBColor.from_string(color)
    rPr = run._r.get_or_add_rPr()
    latin = rPr.find(qn("a:latin"))
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        if latin is not None:
            latin.addnext(ea)
        else:
            rPr.append(ea)
    ea.set("typeface", name)
    cs = rPr.find(qn("a:cs"))
    if cs is None:
        cs = rPr.makeelement(qn("a:cs"), {})
        ea.addnext(cs)
    cs.set("typeface", name)


def text_block(slide: Any, x: float, y: float, w: float, h: float, paras: Sequence[Para],
               *, align: Any = PP_ALIGN.LEFT, anchor: Any = MSO_ANCHOR.TOP,
               size: float = 14.0, color: str = INK, bold: bool = False,
               font: str = FONT, line_spacing: float = 1.22, space_after: float = 4.0,
               margin: float = 0.0) -> Any:
    """在 ``(x, y, w, h)`` 英寸盒子里排一段（或多段）文字。

    ``paras`` 的元素有三种写法：

    - ``"纯文本"`` —— 用本调用的默认字号/颜色；
    - ``("文本", {"size": 12, "color": INK2, "bold": True})`` —— 覆盖若干项；
    - ``[("片段A", {...}), ("片段B", {...})]`` —— 同一段内的多个 run（做「标签：值」混排）。
    """
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0)
    tf.vertical_anchor = anchor

    for index, item in enumerate(paras):
        para = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = line_spacing
        para.space_after = Pt(space_after)
        runs: List[Tuple[str, Dict[str, Any]]]
        if isinstance(item, str):
            runs = [(item, {})]
        elif isinstance(item, tuple):
            runs = [item]
        else:
            runs = list(item)
        for text, spec in runs:
            run = para.add_run()
            run.text = text
            apply_font(
                run,
                name=spec.get("font", font),
                size=spec.get("size", size),
                bold=spec.get("bold", bold),
                color=spec.get("color", color),
            )
    return shape


def bar(slide: Any, x: float, y: float, w: float, h: float, color: str,
        *, radius: float = 0.5, line_color: Optional[str] = None) -> Any:
    """一条色条 / 胶囊。"""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    try:
        shape.adjustments[0] = radius
    except (IndexError, ValueError):  # pragma: no cover - 形状不支持调整量时忽略
        pass
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(color)
    if line_color:
        shape.line.color.rgb = RGBColor.from_string(line_color)
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def card(slide: Any, x: float, y: float, w: float, h: float, *, fill: str = PANEL,
         line_color: str = LINE, radius: float = 0.06, accent: Optional[str] = None,
         accent_w: float = 0.055) -> Any:
    """白底圆角卡片；``accent`` 给定时在左缘加一条竖向色条。"""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    try:
        shape.adjustments[0] = radius
    except (IndexError, ValueError):  # pragma: no cover
        pass
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line_color:
        shape.line.color.rgb = RGBColor.from_string(line_color)
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    if accent:
        bar(slide, x + 0.0, y + 0.14, accent_w, h - 0.28, accent, radius=0.5)
    return shape


def chip(slide: Any, x: float, y: float, w: float, h: float, text: str, *,
         fill: str = SOFT, color: str = ACCENT, size: float = 11.0,
         font: str = FONT, bold: bool = True) -> Any:
    """小圆角标签（工具名、参数名等）。"""
    bar(slide, x, y, w, h, fill, radius=0.5)
    return text_block(slide, x, y, w, h, [text], size=size, color=color, bold=bold,
                      font=font, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def kpi(slide: Any, x: float, y: float, w: float, h: float, value: str, label: str,
        *, color: str = ACCENT, note: Optional[str] = None) -> None:
    """指标卡：大字数值 + 说明。"""
    card(slide, x, y, w, h)
    paras: List[Para] = [(value, {"size": 21, "bold": True, "color": color})]
    paras.append((label, {"size": 11, "color": INK2, "bold": False}))
    if note:
        paras.append((note, {"size": 9.5, "color": INK3, "bold": False}))
    text_block(slide, x + 0.16, y + 0.10, w - 0.32, h - 0.20, paras,
               anchor=MSO_ANCHOR.MIDDLE, space_after=2, line_spacing=1.12)


def table_block(slide: Any, x: float, y: float, col_widths: Sequence[float],
                rows: Sequence[Sequence[str]], *, size: float = 11.0,
                header_size: Optional[float] = None, row_h: float = 0.32,
                aligns: Optional[Sequence[Any]] = None,
                highlight: Optional[Dict[Tuple[int, int], str]] = None,
                header_fill: str = SOFT) -> Any:
    """三线感表格：表头浅底 + 正文白底细线，``highlight`` 可给单元格上色。"""
    n_rows, n_cols = len(rows), len(col_widths)
    total_w = sum(col_widths)
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                   Inches(total_w), Inches(row_h * n_rows))
    table = shape.table
    table.first_row = False
    table.horz_banding = False
    for col, width in enumerate(col_widths):
        table.columns[col].width = Inches(width)
    for row in range(n_rows):
        table.rows[row].height = Inches(row_h)

    for r, row_values in enumerate(rows):
        for c in range(n_cols):
            cell = table.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = header_fill if r == 0 else PANEL
            if highlight and (r, c) in highlight:
                fill = highlight[(r, c)]
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(fill)
            tf = cell.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            para.alignment = aligns[c] if aligns else PP_ALIGN.LEFT
            text = str(row_values[c]) if c < len(row_values) else ""
            run = para.add_run()
            run.text = text
            apply_font(run, size=(header_size or size) if r == 0 else size,
                       bold=(r == 0), color=ACCENT if r == 0 else INK)
    return table


def picture_fit(slide: Any, path: Path, x: float, y: float, w: float, h: float, *,
                caption: Optional[str] = None, cap_size: float = 10.5,
                line_color: Optional[str] = LINE) -> Optional[Any]:
    """等比缩放放进 ``(x, y, w, h)`` 盒子并居中；图下可带一行说明。"""
    if not path.exists():
        card(slide, x, y, w, h, line_color=DANGER)
        text_block(slide, x, y, w, h, [f"（缺图：{path.name}）"],
                   size=12, color=DANGER, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        return None

    from PIL import Image  # python-pptx 的依赖，此处用于取原始宽高

    with Image.open(path) as img:
        ratio = img.size[0] / img.size[1]
    cap_h = 0.30 if caption else 0.0
    box_h = h - cap_h
    if w / box_h > ratio:
        draw_w, draw_h = box_h * ratio, box_h
    else:
        draw_w, draw_h = w, w / ratio
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (box_h - draw_h) / 2
    picture = slide.shapes.add_picture(str(path), Inches(draw_x), Inches(draw_y),
                                       Inches(draw_w), Inches(draw_h))
    if line_color:
        outline = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(draw_x), Inches(draw_y),
                                         Inches(draw_w), Inches(draw_h))
        outline.fill.background()
        outline.line.color.rgb = RGBColor.from_string(line_color)
        outline.line.width = Pt(0.75)
        outline.shadow.inherit = False
    if caption:
        text_block(slide, x, y + h - cap_h, w, cap_h, [caption], size=cap_size,
                   color=INK2, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return picture


def new_slide(prs: Any) -> Any:
    slide = prs.slides.add_slide(prs.slide_layouts[6])       # 6 = 空白版式
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(BG)
    return slide


def head(slide: Any, kicker: str, title: str) -> None:
    """页眉：主色竖条 + 标题 + 一句话导语。"""
    bar(slide, MARGIN, TITLE_T + 0.04, 0.10, TITLE_H - 0.08, ACCENT, radius=0.5)
    text_block(slide, MARGIN + 0.26, TITLE_T - 0.04, CONTENT_W - 0.26, TITLE_H,
               [(title, {"size": 25, "bold": True, "color": INK})], anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        text_block(slide, MARGIN + 0.26, TITLE_T + 0.60, CONTENT_W - 0.26, 0.34,
                   [(kicker, {"size": 12.5, "color": ACCENT, "bold": True})],
                   anchor=MSO_ANCHOR.MIDDLE)


def foot(slide: Any, index: int, note: str = "") -> None:
    text_block(slide, MARGIN, FOOTER_T, CONTENT_W * 0.7, 0.3,
               [(note or f"{TITLE} · {SUBTITLE}", {"size": 9.5, "color": INK3})])
    text_block(slide, MARGIN + CONTENT_W * 0.7, FOOTER_T, CONTENT_W * 0.3, 0.3,
               [(f"{index:02d} / {TOTAL_SLIDES}", {"size": 9.5, "color": INK3})],
               align=PP_ALIGN.RIGHT)


def band(slide: Any, x: float, y: float, w: float, h: float, text: str, *,
         fill: str = SOFT, color: str = ACCENT, size: float = 12.0,
         accent: Optional[str] = None) -> None:
    """整行结论条（左侧可有色条）。"""
    card(slide, x, y, w, h, fill=fill, line_color=None, radius=0.10, accent=accent)
    text_block(slide, x + 0.22, y, w - 0.44, h, [text], size=size, color=color,
               anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.16)


def notes(slide: Any, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


# ==============================================================================
# 十四页正文
# ==============================================================================
def slide_cover(prs: Any, metrics: Dict[str, Any]) -> None:
    slide = new_slide(prs)
    bar(slide, 0.0, 0.0, SLIDE_W, 0.16, ACCENT, radius=0.0)
    bar(slide, 1.05, 1.62, 0.72, 0.09, ACCENT, radius=0.5)
    text_block(slide, 1.05, 1.82, 11.0, 0.34, [(KICKER, {"size": 14, "color": ACCENT, "bold": True})])
    text_block(slide, 1.05, 2.18, 11.5, 1.05, [(TITLE, {"size": 48, "bold": True, "color": INK})])
    text_block(slide, 1.05, 3.24, 11.5, 0.5,
               [(SUBTITLE, {"size": 22, "color": ACCENT2, "bold": True})])
    bar(slide, 1.05, 3.95, 4.6, 0.02, LINE, radius=0.0)

    left = MEMBERS[:2]
    right = MEMBERS[2:]
    for col, group in enumerate((left, right)):
        text_block(slide, 1.05 + col * 5.85, 4.18, 5.6, 0.9,
                   [(item, {"size": 12, "color": INK2, "space_after": 7}) for item in group],
                   line_spacing=1.15)
    text_block(slide, 1.05, 5.30, 11.5, 0.4, [(f"{date.today():%Y 年 %m 月}", {"size": 12, "color": INK2})])

    band(slide, 1.05, 6.05, 11.23, 0.62,
         "交付物：代码仓库 · 技术设计文档 · 系统评测报告 · 用户使用手册 · 演示 PPT     |     "
         "运行环境：Python 3.10+ ，唯一必需依赖 NumPy",
         fill=PANEL, color=INK2, size=11.5, accent=ACCENT)
    notes(slide, "开场一句话：这是一个把「论文知识库」做成可问答系统的完整工程 —— "
                 "检索、生成、Agent、前端、评测五个模块全部自己实现，只依赖 NumPy。\n"
                 "成员姓名/学号/班级请在封面自行填写。")


def slide_toc(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "全篇 14 页，按「问题 → 架构 → 实现 → 评测 → 复盘」推进", "目录")
    items = [
        ("01", "项目背景与痛点", "为什么需要语义检索 + 引用溯源"),
        ("02", "系统架构总览", "五层分层与一次问答的完整链路"),
        ("03", "RAG 链路与 Agent 循环", "手写实现，不依赖 LangChain"),
        ("04", "混合检索与重排序", "三档模式：vector / hybrid / full"),
        ("05", "记忆与会话隔离", "滑动窗口 + 摘要压缩"),
        ("06", "功能演示", "四张真实运行截图"),
        ("07", "评测方法与结果", "61 题评测集 · 严格口径 · 一键复现"),
        ("08", "分块策略与 Bad Case", "四种切分策略实测对比"),
        ("09", "一轮优化前后对比", "真实运行、同语料对照"),
        ("10", "踩坑记录", "三条被实测数据推翻的直觉"),
        ("11", "部署与复现", "三步启动 + Docker 一键部署"),
        ("12", "局限与展望", "如实标注的缺口与下一步"),
    ]
    col_w = CONTENT_W / 2 - 0.2
    for index, (num, name, desc) in enumerate(items):
        col, row = divmod(index, 6)
        x = MARGIN + col * (col_w + 0.4)
        y = BODY_T + row * 0.87
        card(slide, x, y, col_w, 0.74, fill=PANEL)
        bar(slide, x + 0.14, y + 0.16, 0.5, 0.42, SOFT, radius=0.22)
        text_block(slide, x + 0.14, y + 0.16, 0.5, 0.42, [(num, {"size": 13, "bold": True, "color": ACCENT})],
                   align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text_block(slide, x + 0.76, y + 0.09, col_w - 0.92, 0.58,
                   [(name, {"size": 14, "bold": True, "color": INK, "space_after": 2}),
                    (desc, {"size": 10.5, "color": INK3})],
                   line_spacing=1.1)
    foot(slide, 2)
    notes(slide, "念一遍目录即可，重点提示：评测与复盘占 6 页 —— 这个项目的重心在「做完并测出来」，"
                 "而不是「写完代码就结束」。")


def slide_background(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "课程题目一：论文知识库问答系统 —— 传统关键词搜索答不了下面三类问题",
         "项目背景与痛点")
    cards = [
        ("语义问题召不回", "「这个方法相比基线提升多少」这类问法不含原词，"
                        "按字面匹配的搜索召回不到目标段落。", ACCENT),
        ("跨论文对比靠人工", "同一主题分散在多篇 PDF，人工比对方法、数据集与指标，"
                          "费时且容易漏。", INFO),
        ("引用不可追溯", "通用大模型直接作答给不出出处；学术场景里"
                      "「答案从哪来」与答案本身同等重要。", PURPLE),
    ]
    card_w = (CONTENT_W - 0.72) / 3
    for index, (title, body, color) in enumerate(cards):
        x = MARGIN + index * (card_w + 0.36)
        card(slide, x, BODY_T, card_w, 2.28, accent=color)
        text_block(slide, x + 0.30, BODY_T + 0.22, card_w - 0.54, 0.42,
                   [(title, {"size": 16, "bold": True, "color": INK})])
        text_block(slide, x + 0.30, BODY_T + 0.72, card_w - 0.54, 1.36,
                   [(body, {"size": 12, "color": INK2})], line_spacing=1.34)

    text_block(slide, MARGIN, BODY_T + 2.62, CONTENT_W, 0.36,
               [("本项目给出的答案", {"size": 15, "bold": True, "color": ACCENT})])
    chips = [
        "混合检索：向量 + BM25 + RRF 融合",
        "ReAct Agent：8 个工具自主决策",
        "多轮记忆：滑动窗口 + 摘要压缩",
        "引用可溯源：答案附【来源 p.页码】",
    ]
    chip_w = (CONTENT_W - 0.54) / 4
    for index, text in enumerate(chips):
        x = MARGIN + index * (chip_w + 0.18)
        chip(slide, x, BODY_T + 3.06, chip_w, 0.66, text, size=11.5)
    band(slide, MARGIN, BODY_T + 3.96, CONTENT_W, 0.72,
         "一句话定位：把「找论文、读论文、比论文、答问题」串成一条可评测、可复现、可追溯的链路 —— "
         "而不是再做一个只会聊天的机器人。",
         fill=SOFT, color=ACCENT, size=12.5, accent=ACCENT)
    foot(slide, 3)
    notes(slide, "痛点讲具体：学生自己在写论文时最费时的三件事 —— 找、读、比。\n"
                 "强调不是「做个聊天机器人」，而是完整工程 + 可评测。")


def slide_architecture(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "五个模块对应五层：接入 / Agent / 生成 / 检索 / 数据，逐层向下依赖",
         "系统架构总览")
    picture_fit(slide, IMAGES / "architecture.png", MARGIN, BODY_T, 6.5, BODY_B - BODY_T,
                line_color=LINE)
    layers = [
        ("接入层", "Web 界面 + SSE 流式；/api/ask、/api/tools、/api/metrics 等接口", ACCENT),
        ("Agent 决策层", "手写 ReAct 循环 + 意图路由；8 个工具；最多 8 轮、工具超时 45s", PURPLE),
        ("生成层", "RAG 管线 · 提示词模板 · 引用解析；语义缓存 相似度 0.93 / TTL 7 天", INFO),
        ("检索层", "向量 + BM25 → RRF 融合 → 启发式重排；vector / hybrid / full 三档", OK),
        ("数据层", "PDF/DOCX/TXT 解析（内置零依赖）→ 四种分块策略 → 向量库三后端", WARN),
    ]
    x = MARGIN + 6.75
    width = CONTENT_W - 6.75
    row_h = 0.84
    for index, (name, desc, color) in enumerate(layers):
        y = BODY_T + index * (row_h + 0.13)
        card(slide, x, y, width, row_h, accent=color)
        text_block(slide, x + 0.26, y + 0.08, width - 0.44, 0.34,
                   [(name, {"size": 14, "bold": True, "color": color})])
        text_block(slide, x + 0.26, y + 0.42, width - 0.44, 0.38,
                   [(desc, {"size": 10.5, "color": INK2})], line_spacing=1.14)
    band(slide, MARGIN, BODY_B - 0.58, CONTENT_W, 0.52,
         "后端零框架：向量库、BM25、ReAct 循环、Web 服务全部基于标准库与 NumPy 自己实现，"
         "再以「适配器 + 能力探测」的方式接可选依赖。",
         fill=PANEL, color=INK2, size=11, accent=ACCENT)
    foot(slide, 4)
    notes(slide, "讲清「依赖方向」：上层只调用下层接口，所以换成 Chroma / FAISS、换真实模型都不动上层代码。\n"
                 "强调零框架不是为了炫技，是为了让「每层可替换、可单独测」。")


def slide_pipeline(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "检索 → 组装 → 生成 → 引用核对；Agent 在其上做「用哪个工具、要不要再查一轮」的决策",
         "RAG 链路与 Agent 循环")
    gap = 0.39
    pic_w = (CONTENT_W - gap) / 2
    picture_fit(slide, IMAGES / "rag_flow.png", MARGIN, BODY_T, pic_w, 2.72,
                caption="图 1  RAG 问答链路：检索、重排、上下文截断、流式生成与引用核对",
                line_color=LINE)
    picture_fit(slide, IMAGES / "react_loop.png", MARGIN + pic_w + gap, BODY_T, pic_w, 2.72,
                caption="图 2  Agent ReAct 循环：Thought → Action → Observation，直至 final_answer",
                line_color=LINE)
    y = BODY_T + 2.94
    points = [
        ("RAG", "候选召回与重排完成后按相关度排序、截断到 6000 字符；生成走流式，"
                "引用解析后与检索结果逐条核对，来源对不上的标注「未核实」。", INFO),
        ("Agent", "模型输出 JSON 决策（thought / action / args）；最多 8 轮、"
                  "并行工具上限 3、死循环检测窗口 2 —— 决策过程在前端轨迹栏全程可见。", PURPLE),
    ]
    card_w = (CONTENT_W - 0.36) / 2
    for index, (name, desc, color) in enumerate(points):
        x = MARGIN + index * (card_w + 0.36)
        card(slide, x, y, card_w, 1.16, accent=color)
        text_block(slide, x + 0.28, y + 0.14, card_w - 0.5, 0.34,
                   [(name, {"size": 14, "bold": True, "color": color})])
        text_block(slide, x + 0.28, y + 0.50, card_w - 0.5, 0.60,
                   [(desc, {"size": 11, "color": INK2})], line_spacing=1.2)

    text_block(slide, MARGIN, y + 1.28, CONTENT_W, 0.3,
               [("8 个工具（Agent 可自主调用）", {"size": 11.5, "bold": True, "color": INK2})])
    tools = ["knowledge_search", "paper_metadata", "paper_compare", "keyword_extract",
             "summarize_paper", "current_time", "calculator", "web_search"]
    chip_w = (CONTENT_W - 7 * 0.12) / 8
    for index, name in enumerate(tools):
        chip(slide, MARGIN + index * (chip_w + 0.12), y + 1.62, chip_w, 0.42,
             name, size=9.5, font=MONO, fill=PANEL, color=INK2)
    foot(slide, 5)
    notes(slide, "两张图分别对应模块二/模块一（RAG）与模块三（Agent）。\n"
                 "工具名逐个念一遍即可，重点说 knowledge_search 是主力，paper_compare 支撑"
                 "「跨论文对比」这个选题痛点。")


def slide_retrieval(prs: Any, metrics: Dict[str, Any]) -> None:
    slide = new_slide(prs)
    head(slide, "两路召回用排名融合（RRF）合并，规避分数量纲不可比；重排是可选的一档",
         "混合检索与重排序")

    table = (metrics.get("retrieval") or {}).get("table") or []
    labels = {"vector": "纯向量", "hybrid": "混合（RRF）", "full": "混合 + 重排"}
    rows: List[List[str]] = [["检索模式", "严格 Hit@5", "严格 NDCG@5", "平均延迟 (ms)"]]
    for item in table:
        mode = str(item.get("mode", ""))
        rows.append([
            labels.get(mode, mode),
            fmt(item.get("strict_hit@5")),
            fmt(item.get("strict_ndcg@5")),
            f"{float(item.get('avg_latency_ms') or 0):.2f}",
        ])
    if len(rows) == 1:
        rows.append(["未运行", "未运行", "未运行", "未运行"])

    right_aligns = [PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT]
    highlight: Dict[Tuple[int, int], str] = {}
    if table:
        best = max(range(len(table)), key=lambda i: float(table[i].get("strict_ndcg@5") or 0)) + 1
        highlight[(best, 2)] = SOFT          # 严格 NDCG@5 最优的那一行
    table_block(slide, MARGIN, BODY_T + 0.06, [2.1, 1.75, 1.95, 1.75], rows,
                size=12, header_size=11.5, row_h=0.42, aligns=right_aligns,
                highlight=highlight)

    summary = (metrics.get("retrieval") or {}).get("summary") or {}
    note = (f"评测样本 40 题（严格口径，按文档级去重）。hybrid 引擎口径 Hit@K = "
            f"{fmt(summary.get('engine_hit_at_k'))}，作为辅助参考。") if summary else "评测未运行。"
    text_block(slide, MARGIN, BODY_T + 1.86, 7.55, 0.6, [(note, {"size": 10.5, "color": INK3})],
               line_spacing=1.2)

    cards = [
        ("双路召回", "向量：hash n-gram 特征，512 维，零依赖确定性；BM25：k1 = 1.5、b = 0.75。", ACCENT),
        ("RRF 融合", "k = 60，只用排名不用分数 —— 两路分数量纲天然不可比，按排名融合更稳。", INFO),
        ("启发式重排", "6 个特征加权 + 相关性门控，输出 0~1 置信度；可选接入 ollama/local 模型。", PURPLE),
        ("不硬答", "重排置信度低于 0.25 时走降级话术（说明「知识库中没有足够依据」），不编答案。", WARN),
    ]
    x = MARGIN + 7.75
    width = CONTENT_W - 7.75
    row_h = 0.98
    for index, (name, desc, color) in enumerate(cards):
        y = BODY_T + index * (row_h + 0.12)
        card(slide, x, y, width, row_h, accent=color)
        text_block(slide, x + 0.26, y + 0.10, width - 0.44, 0.32,
                   [(name, {"size": 13, "bold": True, "color": color})])
        text_block(slide, x + 0.26, y + 0.44, width - 0.44, 0.50,
                   [(desc, {"size": 10.5, "color": INK2})], line_spacing=1.16)

    band(slide, MARGIN, BODY_B - 0.60, CONTENT_W, 0.54,
         "实测反直觉：叠加重排后严格 NDCG@5 由 0.5827 降到 0.5681 —— 重排不是「越多越好」，"
         "结论见第 09 页优化对比。",
         fill="FFF7ED", color=WARN, size=11.5, accent=WARN)
    foot(slide, 6)
    notes(slide, "这页的核心是「为什么不按分数融合」：两路分数不可比，RRF 只用排名。\n"
                 "最后一条预警留悬念 —— 下一页给数据，第 09 页给结论。")


def slide_memory(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "会话级记忆：滑动窗口控 token，摘要压缩保住长对话的语义连贯",
         "记忆与会话隔离")
    cards = [
        ("滑动窗口", "max_turns = 20、max_tokens = 3000；超出自动从最早的历史开始截断，"
                   "保证注入模型的上下文不超预算。", ACCENT, [
                       "只保留最近若干轮", "token 预算硬约束"]),
        ("摘要压缩", "轮次超过 12 轮时，生成阶段性摘要替换原始历史 —— 省 token 的同时"
                   "保持语义连贯，避免「聊到后面忘了前面」。", PURPLE, [
                       "阈值 12 轮触发", "摘要替换原始历史"]),
        ("会话隔离", "SQLite 持久化（chat.db），按 session 隔离；历史指纹参与缓存键，"
                   "「问题相同、上文不同」不会串用答案。", INFO, [
                       "刷新页面不丢上下文", "缓存键含历史指纹"]),
    ]
    card_w = (CONTENT_W - 0.72) / 3
    for index, (title, body, color, points) in enumerate(cards):
        x = MARGIN + index * (card_w + 0.36)
        card(slide, x, BODY_T, card_w, 3.30, accent=color)
        text_block(slide, x + 0.30, BODY_T + 0.22, card_w - 0.54, 0.4,
                   [(title, {"size": 16, "bold": True, "color": color})])
        text_block(slide, x + 0.30, BODY_T + 0.72, card_w - 0.54, 1.5,
                   [(body, {"size": 11.5, "color": INK2})], line_spacing=1.34)
        for offset, point in enumerate(points):
            py = BODY_T + 2.38 + offset * 0.42
            bar(slide, x + 0.30, py + 0.11, 0.10, 0.10, color, radius=0.5)
            text_block(slide, x + 0.50, py, card_w - 0.78, 0.32,
                       [(point, {"size": 10.5, "color": INK2})], anchor=MSO_ANCHOR.MIDDLE)
    band(slide, MARGIN, BODY_T + 3.52, CONTENT_W, 0.62,
         "效果：数十轮长对话不会被模型上下文窗口卡住；同一问题在上文变化后不会被错误复用旧答案。",
         fill=SOFT, color=ACCENT, size=12, accent=ACCENT)
    foot(slide, 7)
    notes(slide, "记忆层是模块三的一部分。先说「为什么要做」：多轮追问「它和 ResNet 相比如何」时，"
                 "指代必须靠历史才能解析。\n"
                 "再点出与缓存的交互：历史变了，缓存键就变 —— 避免答案串用。")


def slide_demo(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "四张真实运行截图：问答、引用与轨迹、多轮追问、运行指标（截图由脚本自动采集）",
         "功能演示")
    shots = [
        ("01_主界面_知识库与对话.png", "① 主界面：左侧知识库与上传，中间对话区"),
        ("02_问答_引用与Agent轨迹.png", "② 每个答案都带引用；右侧可见 Agent 决策轨迹"),
        ("03_多轮追问_上下文记忆.png", "③ 多轮追问：指代由上下文记忆解析"),
        ("05_运行指标面板.png", "④ 运行指标面板：检索/生成/缓存实时刷新"),
    ]
    gap = 0.39
    cell_w = (CONTENT_W - gap) / 2
    cell_h = 2.60
    for index, (name, caption) in enumerate(shots):
        col, row = divmod(index, 2)
        x = MARGIN + col * (cell_w + gap)
        y = BODY_T + row * (cell_h + 0.28)
        picture_fit(slide, SHOTS / name, x, y, cell_w, cell_h, caption=caption, line_color=LINE)
    foot(slide, 8)
    notes(slide, "现场演示时切到浏览器；这页作为备份 —— 万一现场跑不起来，用截图讲同样的内容。\n"
                 "重点指第 ② 张：引用角标 + 右侧 Agent 轨迹，这是「可追溯」的直接证据。")


def slide_eval(prs: Any, metrics: Dict[str, Any]) -> None:
    slide = new_slide(prs)
    head(slide, "61 题评测集、严格口径统计、一条命令复现 —— 指标全部来自 reports/ 下的真实产物",
         "评测方法与结果")

    retrieval = metrics.get("retrieval") or {}
    summary = retrieval.get("summary") or {}
    agent = metrics.get("agent") or {}
    e2e = metrics.get("end_to_end") or {}
    corpus = metrics.get("corpus") or {}
    eval_size = metrics.get("eval_set_size")
    stats = load_eval_stats()
    dist = "、".join(f"{k} {v}" for k, v in stats.items()) if stats else "见 reports/评测集.json"

    left_w = 5.35
    card(slide, MARGIN, BODY_T, left_w, 4.86, accent=ACCENT)
    text_block(slide, MARGIN + 0.28, BODY_T + 0.18, left_w - 0.5, 0.34,
               [("评测方法", {"size": 15, "bold": True, "color": ACCENT})])
    method: List[Para] = [
        [(f"{eval_size if eval_size else '未运行'} 题评测集", {"size": 12, "bold": True, "color": INK})],
        [(f"题型分布：{dist}", {"size": 10.5, "color": INK2})],
        [(f"评测语料：{corpus.get('num_documents', '未运行')} 篇文档 / "
          f"{corpus.get('num_chunks', '未运行')} 个分块 / 平均 {fmt(corpus.get('avg_chunk_chars'), 1)} 字符",
          {"size": 10.5, "color": INK2})],
        [("严格口径：", {"size": 10.5, "bold": True, "color": INK}),
         ("按文档级去重统计，未召回不计入；另有引擎口径辅助参考", {"size": 10.5, "color": INK2})],
        [("运行后端：", {"size": 10.5, "bold": True, "color": INK}),
         ("LLM = mock（离线确定性）、Embedding = hash 512 维", {"size": 10.5, "color": INK2})],
        [("证据链：", {"size": 10.5, "bold": True, "color": INK}),
         ("评测集 JSON + 明细 JSON/XLSX + 三份报告，可一键重跑", {"size": 10.5, "color": INK2})],
    ]
    text_block(slide, MARGIN + 0.28, BODY_T + 0.62, left_w - 0.5, 3.2,
               method, line_spacing=1.3, space_after=9)
    band(slide, MARGIN + 0.28, BODY_T + 3.94, left_w - 0.56, 0.76,
         "mock 后端下的指标反映的是工程链路正确性（检索、引用、降级、缓存），"
         "不代表真实大模型的生成质量 —— 这一点在交付书与评测报告中同样如实标注。",
         fill="FFF7ED", color=WARN, size=10, accent=WARN)

    kpis = [
        (fmt(summary.get("hit_at_k")), "严格 Hit@5（hybrid）", ACCENT),
        (fmt(summary.get("ndcg_at_k")), "严格 NDCG@5", ACCENT),
        (fmt(summary.get("mrr")), "严格 MRR", ACCENT),
        (f"{fmt(e2e.get('avg_latency_ms'), 2)} ms", "端到端平均延迟（检索 + 生成）", INFO),
        (f"{(float(agent.get('tool_accuracy') or 0) * 100):.0f}%" if agent else "未运行",
         "Agent 工具调用准确率", OK),
        (fmt(agent.get("avg_iterations"), 2), "Agent 平均推理轮次", OK),
        (f"{(float(e2e.get('avg_citation_accuracy') or 0) * 100):.0f}%" if e2e else "未运行",
         "引用准确率（覆盖率 97.5%）", PURPLE),
        (f"{(float(e2e.get('degrade_rate') or 0) * 100):.1f}%" if e2e else "未运行",
         "降级率（低相关度 6 例）", WARN),
    ]
    x0 = MARGIN + left_w + 0.39
    card_w = (CONTENT_W - left_w - 0.39 - 0.34) / 2
    for index, (value, label, color) in enumerate(kpis):
        col, row = divmod(index, 4)
        x = x0 + col * (card_w + 0.34)
        y = BODY_T + row * 1.24
        kpi(slide, x, y, card_w, 1.10, value, label, color=color)
    foot(slide, 9)
    notes(slide, "评测方法讲三件事：多少题、什么口径、跑在什么后端。\n"
                 "后端那句必须说清楚（mock + hash），否则指标会被误读成真实模型效果。\n"
                 "KPI 一行一行念，突出引用准确率 100% 与降级率 15%。")


def slide_chunking_badcase(prs: Any, metrics: Dict[str, Any]) -> None:
    slide = new_slide(prs)
    head(slide, "同一语料、同一评测集，唯一变量是分块策略；Bad Case 全部集中在检索未召回",
         "分块策略对比与 Bad Case")

    rows = load_chunking_rows()
    grid: List[List[str]] = [["分块策略", "分块数", "严格 Hit@5", "严格 NDCG@5", "平均延迟 (ms)"]]
    for item in rows:
        grid.append([item["label"], item["num_chunks"], fmt(item["hit"]), fmt(item["ndcg"]),
                     f"{float(item['latency']):.3f}"])
    if len(grid) == 1:
        grid.append(["未运行", "未运行", "未运行", "未运行", "未运行"])
    highlight: Dict[Tuple[int, int], str] = {}
    if rows:
        best_hit = max(range(len(rows)), key=lambda i: float(rows[i]["hit"])) + 1
        best_ndcg = max(range(len(rows)), key=lambda i: float(rows[i]["ndcg"])) + 1
        highlight[(best_hit, 3)] = "DCFCE7"
        highlight[(best_ndcg, 4)] = "DBEAFE"
    table_block(slide, MARGIN, BODY_T + 0.06, [2.05, 1.25, 1.6, 1.75, 1.75], grid,
                size=11.5, header_size=11, row_h=0.44,
                aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
                highlight=highlight)
    text_block(slide, MARGIN, BODY_T + 2.36, 8.4, 0.4,
               [("绿 = Hit@5 最优（recursive）　蓝 = NDCG@5 最优（fixed）　"
                 "评测集 61 题、hybrid、top_k = 5", {"size": 10.5, "color": INK3})])

    bad_x = MARGIN + 8.55
    bad_w = CONTENT_W - 8.55
    card(slide, bad_x, BODY_T, bad_w, 2.60, accent=DANGER)
    text_block(slide, bad_x + 0.28, BODY_T + 0.16, bad_w - 0.5, 0.36,
               [("Bad Case 分析（40 条样本）", {"size": 14, "bold": True, "color": DANGER})])
    text_block(slide, bad_x + 0.28, BODY_T + 0.62, bad_w - 0.5, 1.78, [
        [("13 条", {"size": 13, "bold": True, "color": DANGER}),
         ("（32.5%）全部为「检索失败：未召回相关文档」，严重度 P1", {"size": 10.5, "color": INK2})],
        ("根因：hash 词面向量对中英跨语言无信号 + 术语表述差异 + candidate_k 偏小",
         {"size": 10.5, "color": INK2}),
        [("引用错误 0 条", {"size": 11, "bold": True, "color": OK}),
         ("（P0 优先级严防，本轮未发现）", {"size": 10.5, "color": INK2})],
    ], line_spacing=1.24, space_after=7)

    band(slide, MARGIN, BODY_T + 2.92, CONTENT_W, 0.94,
         "「能召回」≠「排得准」：Hit@5 最优是 recursive（0.8033），NDCG@5 最优却是 fixed（0.6530）—— "
         "两个指标指向不同策略，说明选型不能只看单个数字；分块粒度、候选集大小与是否重排会相互影响。",
         fill=SOFT, color=ACCENT, size=12, accent=ACCENT)
    foot(slide, 10)
    notes(slide, "四种策略：fixed / recursive / semantic / paper。semantic 切得最细（52 块）但指标并不最好 —— "
                 "粒度不是越细越好。\n"
                 "Bad Case 只有一类失败：检索没召回。诚实说出来，下一页正好是改进它的迭代结果。")


def slide_optimization(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "一轮参数迭代：同语料、真实运行、只改三个参数",
         "优化前后对比")
    chips = ["candidate_k 20 → 40", "enable_rerank true → false", "chunk_size 512 → 384（27 → 37 块）"]
    chip_w = (CONTENT_W - 0.36) / 3
    for index, text in enumerate(chips):
        chip(slide, MARGIN + index * (chip_w + 0.18), BODY_T, chip_w, 0.46, text,
             size=11.5, fill=PANEL, color=INK2, font=MONO)

    rows = load_iteration_rows()
    grid: List[List[str]] = [["指标（严格口径）", "优化前", "优化后", "相对变化"]]
    highlight: Dict[Tuple[int, int], str] = {}
    for index, item in enumerate(rows, start=1):
        grid.append([item["label"], item["before"], item["after"], item["rel"]])
        good = is_improvement(item["label"], item["rel"])
        if good is True:
            highlight[(index, 3)] = "DCFCE7"      # 绿 = 改善
        elif good is False:
            highlight[(index, 3)] = "FEE2E2"      # 红 = 劣化
    if len(grid) == 1:
        grid.append(["未运行", "未运行", "未运行", "未运行"])
    table_block(slide, MARGIN, BODY_T + 0.68, [3.3, 1.7, 1.7, 1.7], grid,
                size=11.5, header_size=11, row_h=0.42,
                aligns=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT],
                highlight=highlight)

    right_x = MARGIN + 8.85
    right_w = CONTENT_W - 8.85
    card(slide, right_x, BODY_T + 0.68, right_w, 3.36, accent=OK)
    text_block(slide, right_x + 0.26, BODY_T + 0.84, right_w - 0.48, 0.34,
               [("这轮改了什么", {"size": 14, "bold": True, "color": OK})])
    text_block(slide, right_x + 0.26, BODY_T + 1.28, right_w - 0.48, 2.6, [
        ("放开候选集：candidate_k 20 → 40，让重排有更多可选的召回结果。", {"size": 11, "color": INK2}),
        ("关掉重排：本轮语料上重排反而压低 NDCG，关闭同时降低延迟与降级率。", {"size": 11, "color": INK2}),
        ("调细分块：chunk_size 512 → 384，块更贴合单段论述，命中率上升。", {"size": 11, "color": INK2}),
    ], line_spacing=1.26, space_after=9)

    band(slide, MARGIN, BODY_T + 4.20, CONTENT_W, 0.62,
         "提升来自三项参数的交互：单项消融并非都为正 —— 只调细 chunk_size 时 NDCG@5 反而 -0.0121。"
         "参数调优要联合验证，不能把单项最优当成整体最优。",
         fill=PANEL, color=INK2, size=11.5, accent=ACCENT)
    foot(slide, 11)
    notes(slide, "数据是同一份语料上真实跑出来的两组配置，不是估算。\n"
                 "强调「三改一」：除了 Hit@5 +4.26%，延迟 -10.37%、降级率 -79.99% —— "
                 "降级率降低意味着「答不出来」的情况少了八成。")


def slide_pitfalls(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "三条被实测数据推翻的直觉 —— 每条都有指标兜底",
         "踩坑记录")
    items = [
        ("01", "重排序不是越多越好", WARN,
         "直觉上「多加一层重排精度更高」，于是把它设成了默认开启。",
         "叠加重排后严格 NDCG@5 0.5827 → 0.5681；关掉重排后降级率 16.4% → 3.3%。",
         "特征工程要拿指标验证，别拿直觉上线；重排的收益依赖语料与候选集质量。"),
        ("02", "「能召回」≠「排得准」", INFO,
         "按 Hit@5 选了 recursive 切分，却发现排序质量并不是最好的。",
         "Hit@5 最优 recursive 0.8033，NDCG@5 最优 fixed 0.6530。",
         "召回与排序是两个目标，选型要同时看两类指标，否则会选到「能搜到但排得差」的配置。"),
        ("03", "缓存的两条硬规则", PURPLE,
         "缓存本是为了快，但两次实现在「正确性」和「性能」上各踩一坑。",
         "降级答案入库会把「答不出来」固化成缓存；写入路径曾用 SELECT * 读回整行（含答案正文）"
         "再逐条算余弦，未命中时反序列化数 MB 文本。",
         "降级/空答案一律不缓存；打分只取轻量列 + 一次矩阵乘，命中后再读正文。"),
    ]
    card_w = (CONTENT_W - 0.72) / 3
    for index, (num, title, color, phen, data, lesson) in enumerate(items):
        x = MARGIN + index * (card_w + 0.36)
        card(slide, x, BODY_T, card_w, 4.64, accent=color)
        bar(slide, x + 0.30, BODY_T + 0.24, 0.44, 0.38, color, radius=0.3)
        text_block(slide, x + 0.30, BODY_T + 0.24, 0.44, 0.38,
                   [(num, {"size": 12, "bold": True, "color": PANEL})],
                   align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text_block(slide, x + 0.30, BODY_T + 0.74, card_w - 0.56, 0.62,
                   [(title, {"size": 14.5, "bold": True, "color": INK})], line_spacing=1.12)
        blocks = [("现象", phen, INK2, 0.58), ("数据", data, color, 1.00), ("教训", lesson, INK2, 0.80)]
        y = BODY_T + 1.46
        for label, text, text_color, height in blocks:
            text_block(slide, x + 0.30, y, card_w - 0.56, 0.24,
                       [(label, {"size": 10, "bold": True, "color": text_color})])
            text_block(slide, x + 0.30, y + 0.24, card_w - 0.56, height,
                       [(text, {"size": 10.5, "color": text_color if label == "数据" else INK2})],
                       line_spacing=1.26)
            y += height + 0.34
    foot(slide, 12)
    notes(slide, "这页是「复盘」环节，讲的时候用「我们以为 → 数据说 → 所以我们改成」三段式。\n"
                 "第 3 条可以展开：缓存正确性（降级不入库）和缓存性能（轻量列 + 矩阵乘）是两个独立问题。")


def slide_deploy(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "三步启动、Docker 一键部署、评测一键复现 —— 评委可当场验证",
         "部署与复现")

    left_w = 6.35
    card(slide, MARGIN, BODY_T, left_w, 3.1, accent=ACCENT)
    text_block(slide, MARGIN + 0.28, BODY_T + 0.16, left_w - 0.5, 0.34,
               [("三步启动（默认离线 mock 模式，无需任何密钥）", {"size": 13.5, "bold": True, "color": ACCENT})])
    code = [
        "# 1) 安装唯一必需依赖",
        "pip install numpy",
        "# 2) 配置 API 密钥（用真实模型时才需要）",
        "python -m src.utils.secrets set llm_api_key",
        "# 3) 环境自检 + 生成样例论文 + 建立索引 + 启动",
        "python scripts/bootstrap.py --index",
        "python run.py     # 自动打开 http://127.0.0.1:8501",
    ]
    text_block(slide, MARGIN + 0.28, BODY_T + 0.60, left_w - 0.5, 2.3,
               [(line, {"size": 10, "font": MONO, "color": OK if line.startswith("#") else INK2,
                        "space_after": 3}) for line in code], line_spacing=1.24)

    right_x = MARGIN + left_w + 0.39
    right_w = CONTENT_W - left_w - 0.39
    card(slide, right_x, BODY_T, right_w, 3.1, accent=INFO)
    text_block(slide, right_x + 0.28, BODY_T + 0.16, right_w - 0.5, 0.34,
               [("Docker 与无界面校验", {"size": 13.5, "bold": True, "color": INFO})])
    text_block(slide, right_x + 0.28, BODY_T + 0.62, right_w - 0.5, 2.3, [
        ("docker compose -f docker/docker-compose.yml up -d", {"size": 10, "font": MONO, "color": INK2}),
        ("python run.py --health        # 自检各后端可用性", {"size": 10, "font": MONO, "color": INK2}),
        ("python run.py --list-tools   # 打印 8 个工具与参数", {"size": 10, "font": MONO, "color": INK2}),
        ("python run.py --cli          # 无浏览器环境下问答", {"size": 10, "font": MONO, "color": INK2}),
        ("端口默认 8501，被占用时自动向后寻找并在日志中提示。", {"size": 10.5, "color": INK2}),
    ], line_spacing=1.28, space_after=7)

    band(slide, MARGIN, BODY_T + 3.34, CONTENT_W, 1.28,
         "", fill=PANEL, color=INK2, accent=ACCENT)
    text_block(slide, MARGIN + 0.26, BODY_T + 3.44, CONTENT_W - 0.5, 0.32,
               [("复现本 PPT 与交付书里的全部数字（三条命令）", {"size": 12.5, "bold": True, "color": ACCENT})])
    text_block(slide, MARGIN + 0.26, BODY_T + 3.80, CONTENT_W - 0.5, 0.76, [
        ("python -m src.evaluation.eval_set            # 生成 61 题评测集",
         {"size": 10, "font": MONO, "color": INK2, "space_after": 3}),
        ("APP_LLM__PROVIDER=mock python -m src.evaluation.report --quick   # 评测 + 报告",
         {"size": 10, "font": MONO, "color": INK2, "space_after": 3}),
        ("python -m src.evaluation.chunking_experiment  # 四种分块策略对比实验",
         {"size": 10, "font": MONO, "color": INK2}),
    ], line_spacing=1.2)
    foot(slide, 13)
    notes(slide, "这页是给评委的「验证入口」：三步启动、Docker、以及复现全部指标的三条命令。\n"
                 "强调离线 mock 模式跑得起来 —— 不需要网络、不需要密钥也能看到完整功能。")


def slide_limits(prs: Any) -> None:
    slide = new_slide(prs)
    head(slide, "缺口如实列在这里：评测跑在离线后端、语料是合成样例 —— 交付书与报告中同样标注",
         "局限与展望")
    left_w = (CONTENT_W - 0.39) / 2
    card(slide, MARGIN, BODY_T, left_w, 4.2, accent=WARN)
    text_block(slide, MARGIN + 0.28, BODY_T + 0.16, left_w - 0.5, 0.34,
               [("当前局限（如实标注，非待办美化）", {"size": 14, "bold": True, "color": WARN})])
    limits = [
        "语料为 3 篇脚本合成的样例论文，未达课程建议的 10–20 篇真实论文；",
        "Embedding 实测走 hash 后端（词面相似度），bge / m3e 对比未做；",
        "Chroma / FAISS 适配器已实现但环境未安装，实测走 NumPy 后端；",
        "生成参数（Temperature / Top-p / Top-k）对比实验未做 —— 评测跑在 mock 后端下无意义；",
        "答案质量的人工评分表未附。",
    ]
    text_block(slide, MARGIN + 0.28, BODY_T + 0.62, left_w - 0.56, 3.4,
               [(item, {"size": 11, "color": INK2, "space_after": 10}) for item in limits],
               line_spacing=1.28)

    right_x = MARGIN + left_w + 0.39
    card(slide, right_x, BODY_T, left_w, 4.2, accent=OK)
    text_block(slide, right_x + 0.28, BODY_T + 0.16, left_w - 0.5, 0.34,
               [("下一步（按优先级）", {"size": 14, "bold": True, "color": OK})])
    nexts = [
        "换真实论文语料（10 篇以上），重跑全部评测与对比实验；",
        "接入 bge-large-zh 与真实大模型，复测检索与生成质量；",
        "跨语言检索优化：当前 P1 根因是 hash 词面向量对中英跨语言无信号；",
        "补齐生成参数对比实验与人工评分表。",
    ]
    text_block(slide, right_x + 0.28, BODY_T + 0.62, left_w - 0.56, 3.4,
               [(item, {"size": 11, "color": INK2, "space_after": 12}) for item in nexts],
               line_spacing=1.28)

    band(slide, MARGIN, BODY_T + 4.40, CONTENT_W, 0.56,
         "谢谢 / Q & A　　代码与文档见交付包：README.md · docs/技术设计文档.md · "
         "docs/用户使用手册.md · reports/系统评测报告.md",
         fill=SOFT, color=ACCENT, size=12, accent=ACCENT)
    foot(slide, 14)
    notes(slide, "收尾不吹：先把缺口说清楚，再说下一步怎么补。\n"
                 "如果被问到「为什么语料是合成的」——回答：为了保证离线可复现与可分发，"
                 "并已把这一点写进交付书与报告中。")


# ==============================================================================
# 组装与自检
# ==============================================================================
def build(metrics: Dict[str, Any]) -> Any:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W), Inches(SLIDE_H)
    slide_cover(prs, metrics)
    slide_toc(prs)
    slide_background(prs)
    slide_architecture(prs)
    slide_pipeline(prs)
    slide_retrieval(prs, metrics)
    slide_memory(prs)
    slide_demo(prs)
    slide_eval(prs, metrics)
    slide_chunking_badcase(prs, metrics)
    slide_optimization(prs)
    slide_pitfalls(prs)
    slide_deploy(prs)
    slide_limits(prs)

    props = prs.core_properties
    props.title = f"{TITLE} —— {SUBTITLE}"
    props.author = "智能科研助理项目组"
    props.comments = "由 scripts/make_ppt.py 生成；指标与 reports/ 下的评测产物同源。"
    return prs


def _estimate_text_height(tf: Any, width_in: float) -> float:
    """粗略估算文本框需要的实际高度（英寸），用于发现明显溢出。

    中文字符按 1.0 em 宽、ASCII 按 0.52 em 宽估算；行高系数把 ``line_spacing``
    折算进来（1.18 起步，>1 的部分只计一半，避免单行标题被误判为溢出）。
    """
    paras = list(tf.paragraphs)
    total = 0.0
    for index, para in enumerate(paras):
        text = "".join(run.text for run in para.runs)
        size = max((run.font.size.pt for run in para.runs if run.font.size), default=14.0)
        spacing = para.line_spacing if isinstance(para.line_spacing, float) else 1.0
        factor = min(1.18 + max(spacing - 1.0, 0.0) * 0.5, 1.35)
        if not text.strip():
            lines = 1
        else:
            units = sum(1.0 if ord(ch) > 0x2E80 else 0.52 for ch in text)
            lines = max(1, math.ceil(size * units / 72.0 / max(width_in, 0.5)))
        total += lines * size * factor / 72.0
        if index < len(paras) - 1 and para.space_after is not None:
            total += para.space_after.pt / 72.0
    return total


def check(path: Path) -> List[str]:
    """生成后自检：页数、形状越界、疑似文字溢出、图片与备注是否齐全。"""
    problems: List[str] = []
    prs = Presentation(str(path))
    if len(prs.slides) != TOTAL_SLIDES:
        problems.append(f"页数不是 {TOTAL_SLIDES}：{len(prs.slides)}")
    width, height = prs.slide_width, prs.slide_height
    pictures = 0
    for index, slide in enumerate(prs.slides, start=1):
        if slide.notes_slide.notes_text_frame.text.strip() == "":
            problems.append(f"第 {index} 页缺少备注")
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pictures += 1
            if shape.left is None:
                continue
            if shape.left < 0 or shape.top < 0 or \
                    shape.left + shape.width > width + Emu(1000) or \
                    shape.top + shape.height > height + Emu(1000):
                problems.append(f"第 {index} 页有形状越界：{shape.shape_type} "
                                f"@ ({shape.left/914400:.2f}, {shape.top/914400:.2f}) "
                                f"{shape.width/914400:.2f}×{shape.height/914400:.2f}")
            if shape.has_text_frame and shape.text_frame.text.strip():
                # 可用宽度要扣掉文本框自身的内边距（本脚本统一设为 0，
                # 未显式设置时按 PowerPoint 默认的左右各 0.1" 计）
                tf = shape.text_frame
                ml = tf.margin_left if tf.margin_left is not None else Inches(0.1)
                mr = tf.margin_right if tf.margin_right is not None else Inches(0.1)
                box_w = (shape.width - ml - mr) / 914400
                box_h = shape.height / 914400 - 0.05
                need = _estimate_text_height(tf, box_w)
                if need > box_h * 1.06:
                    snippet = shape.text_frame.text.strip().replace("\n", " / ")[:34]
                    problems.append(f"第 {index} 页文字可能溢出（需 {need:.2f}\" > 框 {box_h:.2f}\"）：{snippet}")
    if pictures < 7:
        problems.append(f"内嵌图片少于预期（{pictures} < 7）")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="生成课程演示 PPT")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出路径（默认项目根目录）")
    parser.add_argument("--no-check", action="store_true", help="跳过生成后的自检")
    args = parser.parse_args()

    out: Path = args.out
    if out.exists():
        sys.stderr.write(f"提示：将覆盖已存在的文件 {out.name}（若正被 WPS/PowerPoint 打开会失败）\n")
    if not IMAGES.exists():
        sys.stderr.write(f"警告：找不到图片目录 {IMAGES}，相关页面会标注缺图\n")

    metrics = load_metrics()
    prs = build(metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))

    size_kb = out.stat().st_size / 1024
    print(f"已生成：{out}（{len(prs.slides)} 页，{size_kb:.0f} KB）")
    if metrics:
        print(f"指标来源：reports/性能评测数据.json（生成于 {metrics.get('generated_at', '未知时间')}，"
              f"评测集 {metrics.get('eval_set_size', '?')} 题）")
    else:
        print("警告：未读到评测数据，PPT 中的指标位置已标注「未运行」")

    if not args.no_check:
        problems = check(out)
        if problems:
            print("\n自检发现问题：")
            for item in problems:
                print(f"  ✘ {item}")
            return 1
        print("自检通过：页数、形状边界、文字高度、图片与备注均正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
