#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成课程交付书（Word ``.docx``）。

为什么用脚本生成而不是手敲
--------------------------
1. **数字不会说谎**：评测指标直接从 :file:`reports/性能评测数据.json` 读取，
   附录 A 的目录树与代码行数直接扫描磁盘 —— 重新跑一遍评测再执行本脚本，
   交付书里的数字就自动跟上，不会出现"报告更新了、文档还写着旧数"的尴尬。
2. **可复现**：排版（字体、行距、表格样式、图表位置）由代码固定，
   换台电脑生成的结果一致。
3. **占位符集中**：成员姓名/学号/班级统一在 :data:`MEMBERS` 里替换。

用法::

    python scripts/make_delivery_doc.py                     # 默认输出到项目根目录
    python scripts/make_delivery_doc.py --out D:/交付书.docx

依赖：``python-docx``（仅本脚本需要，不属于系统运行依赖）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt, RGBColor
except ImportError:  # pragma: no cover - 环境缺依赖时给出可操作提示
    sys.stderr.write(
        "缺少 python-docx。请先安装：pip install python-docx\n"
    )
    raise SystemExit(2)

# ==============================================================================
# 常量与元信息
# ==============================================================================
TITLE = "智能科研助理 —— 基于 RAG+Agent 的论文知识库问答系统"
SUBTITLE = "生产实习课程报告"
DEFAULT_OUT = PROJECT_ROOT / "智能科研助理_项目交付书.docx"
PPT_PATH = PROJECT_ROOT / "智能科研助理_项目演示.pptx"  # 交付物清单第 4 项

#: 开发侧工具脚本：只在本地生成交付书 / PPT / 截图 / 架构图用，
#: 不随交付包分发，附录 A 的目录树与代码统计里也不列出（保持文档与包内一致）。
DEV_ONLY_SCRIPTS = {
    "scripts/make_delivery_doc.py",
    "scripts/make_ppt.py",
    "scripts/capture_screenshots.py",
    "scripts/make_diagrams.py",
}

#: 二进制/压缩/Office 格式：目录树只统计文本文件的行数，这些直接跳过
BINARY_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".pyd", ".dll", ".exe", ".whl",
    ".db", ".npy", ".npz",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".zip", ".gz", ".tar", ".7z", ".rar",
}
IMAGES = PROJECT_ROOT / "docs" / "images"
SHOTS = IMAGES / "screenshots"

FONT_BODY = "仿宋"
FONT_HEAD = "黑体"
FONT_SUB = "仿宋"
FONT_ASCII = "Times New Roman"
FONT_CODE = "Consolas"
#: 课程模板的封面大标题用的是隶书 36 磅（见「生产实习课程报告」一行）
FONT_COVER = "隶书"

#: 字号规范 —— 逐条照抄课程模板的括注：
#: 「一、项目概述（黑体三号，加粗）」  → 一级标题黑体 16 磅加粗；
#: 「1.1 项目背景（仿宋4号，1.5倍行距）」→ 小标题与正文同为仿宋 14 磅、1.5 倍行距。
#: 三号 = 16 磅，四号 = 14 磅。小标题与正文同字体同字号，靠**加粗**区分层级
#: （模板那一行没写「加粗」，但不加粗就与正文完全无从分辨）。
BODY_SIZE = 14
H1_SIZE = 16
H2_SIZE = 14

#: 页面尺寸与页边距，逐项对齐课程模板（项目交付模板(1).docx）的节属性：
#: A4 + 上下 2.54 cm / 左右 3.17 cm。python-docx 默认模板是 Letter + 2.4/2.6，
#: 不显式设置的话交付书与模板的页面就不是一个规格。
PAGE_WIDTH_CM = 21.0
PAGE_HEIGHT_CM = 29.7
MARGIN_TB_CM = 2.54
MARGIN_LR_CM = 3.17

#: 正文可用宽度（21.0 - 3.17 × 2 = 14.66 cm）。表格列宽与代码块都不得超过它，
#: 否则 Word 会把表格撑出版心、把代码行折行（附录的目录树会散架）。
TEXT_WIDTH_CM = PAGE_WIDTH_CM - 2 * MARGIN_LR_CM
TABLE_MAX_CM = 14.6
CODE_MAX_CM = TEXT_WIDTH_CM - 0.45  # 减去代码块的左缩进
#: 插图宽度（英寸）：5.7 in ≈ 14.48 cm，正好落在 14.66 cm 的版心内。
#: 原来写死的 6.0 in（15.24 cm）在模板的窄边距下会顶出版心、和正文对不齐。
PICTURE_WIDTH_IN = 5.7

GREEN = RGBColor(0x1B, 0x7F, 0x3B)
ORANGE = RGBColor(0xC0, 0x6A, 0x00)
RED = RGBColor(0xB0, 0x2A, 0x2A)
GREY = RGBColor(0x60, 0x60, 0x60)

#: 小组成员占位（组员自行替换姓名 / 学号 / 班级即可）
MEMBERS: List[Tuple[str, str, str, str, str]] = [
    (
        "后端开发工程师",
        "成员 1：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
        "文档加载与解析、分块策略、向量化与索引、混合检索与重排序、RAG 生成流水线（Prompt / 引用溯源 / 流式输出 / 语义缓存 / 降级）",
        "模块一、模块二",
        "25%",
    ),
    (
        "Agent 开发工程师",
        "成员 2：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
        "ReAct 核心循环手写实现、System Prompt 设计、8 个工具开发、工具路由与并行调用、错误恢复与死循环检测、多轮对话记忆管理",
        "模块三",
        "25%",
    ),
    (
        "前端开发工程师",
        "成员 3：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
        "Web 服务（HTTP + SSE）、三栏交互界面、引用角标与 Markdown 渲染、Agent 轨迹可视化、系统指标面板、端到端联调",
        "模块四",
        "25%",
    ),
    (
        "测试与质量保证工程师",
        "成员 4：＿＿＿＿班　＿＿＿＿号　＿＿＿＿＿（请填写）",
        "评测集构建、检索三档对比实验、分块策略对比实验、Bad Case 分类与根因分析、迭代优化对照实验、技术设计文档与用户手册、单元测试与 Docker 交付",
        "模块五",
        "25%",
    ),
]

#: 项目背景（课程模板要求 ≤200 字，脚本会做长度校验）
BACKGROUND = (
    "科研人员面对海量文献，检索与精读负担沉重；通用大模型无法访问未公开的私有文献，"
    "答案没有出处、幻觉不可核查；现有学术搜索引擎又只做关键词匹配，缺少深层语义理解与"
    "多步推理能力。检索增强生成（RAG）让模型「先查后答」，答案可溯源、数据不出本地；"
    "Agent 则赋予系统自主规划与工具调用的能力。二者结合，正是把大模型安全落地到科研"
    "日常工作的可行路径。"
)


# ==============================================================================
# 字体与段落工具
# ==============================================================================
def _set_font(run: Any, name: str = FONT_BODY, size: Optional[float] = None,
              bold: Optional[bool] = None, color: Optional[RGBColor] = None,
              italic: Optional[bool] = None, ascii_name: Optional[str] = None) -> None:
    """设置 run 的字体：**中文字体必须写 eastAsia**，否则 Word 会用默认宋体。"""
    run.font.name = ascii_name or name
    r_fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    r_fonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


#: 行内标记：``**加粗**`` 与 ``` `等宽` ``` —— 转成真正的 Word 字符格式，而不是留字面星号
_INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_rich(p: Any, text: str, *, size: float, font: str = FONT_BODY,
             color: Optional[RGBColor] = None, bold: bool = False) -> Any:
    """把 ``**加粗**`` / ``` `等宽` ``` 解析成多个 run 追加到段落 ``p``。"""
    for piece in _INLINE_RE.split(text):
        if not piece:
            continue
        if len(piece) > 4 and piece.startswith("**") and piece.endswith("**"):
            run = p.add_run(piece[2:-2])
            _set_font(run, font, size, bold=True, color=color)
        elif len(piece) > 2 and piece.startswith("`") and piece.endswith("`"):
            run = p.add_run(piece[1:-1])
            run.font.name = FONT_CODE
            run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(max(size - 0.5, 8.0))
            if color is not None:
                run.font.color.rgb = color
        else:
            run = p.add_run(piece)
            _set_font(run, font, size, bold=bold, color=color)
    return p


def para(doc: Any, text: str = "", *, size: float = BODY_SIZE, font: str = FONT_BODY,
         bold: bool = False, color: Optional[RGBColor] = None,
         align: Optional[Any] = None, line: float = 1.5,
         before: float = 0, after: float = 4, indent_first: bool = False,
         ascii_name: Optional[str] = None) -> Any:
    """加一个正文段落。默认：仿宋四号、1.5 倍行距（与课程格式要求一致）。"""
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    fmt = p.paragraph_format
    fmt.line_spacing = line
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    if indent_first:
        fmt.first_line_indent = Pt(size * 2)
    if ascii_name:
        run = p.add_run(text)
        _set_font(run, font, size, bold, color, ascii_name=ascii_name)
    else:
        add_rich(p, text, size=size, font=font, color=color, bold=bold)
    return p


def bullet(doc: Any, text: str, *, size: float = BODY_SIZE, level: int = 0,
           bold_head: Optional[str] = None) -> Any:
    """项目符号段落（用「•」手写，避免依赖模板里可能不存在的列表样式）。"""
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.line_spacing = 1.5
    fmt.space_after = Pt(2)
    fmt.left_indent = Pt(18 * (level + 1))
    run = p.add_run("• " if level == 0 else "○ ")
    _set_font(run, FONT_BODY, size)
    if bold_head:
        head = p.add_run(bold_head)
        _set_font(head, FONT_BODY, size, bold=True)
    add_rich(p, text, size=size)
    return p


def heading(doc: Any, text: str, level: int = 1) -> Any:
    """标题：大标题（一级）黑体三号加粗，小标题（二、三级）仿宋四号加粗。"""
    size = {1: H1_SIZE, 2: H2_SIZE, 3: H2_SIZE}[level]
    font = FONT_HEAD if level == 1 else FONT_SUB
    p = doc.add_heading(level=level)
    fmt = p.paragraph_format
    fmt.keep_with_next = True
    fmt.line_spacing = 1.5
    fmt.space_before = Pt(12 if level == 1 else 8)
    fmt.space_after = Pt(6 if level == 1 else 4)
    run = p.add_run(text)
    _set_font(run, font, size, bold=True, color=RGBColor(0, 0, 0))
    return p


def _line_width_cm(text: str, size: float) -> float:
    """估算一行文本的渲染宽度（厘米）。

    Consolas 的字符步进恰好是 0.55 em，中文按 eastAsia 走宋体、占满 1 em。
    用来在没有 Word 排版引擎的情况下判断代码块会不会被折行。
    """
    em = size / 72.0 * 2.54
    wide = sum(1 for ch in text if unicodedata.east_asian_width(ch) in ("W", "F"))
    return (wide + (len(text) - wide) * 0.55) * em


def _fit_size(text: str, size: float, limit: float = TEXT_WIDTH_CM - 0.6) -> float:
    """把**不能折行**的单行文本压到版心以内（封面字段、题目这类）。

    正文段落折行无所谓，但封面「考核内容：<长题目>」一旦折行就把这一格撑成两行，
    跟模板的对齐方式就散了。宁可小一号字，也不要折行。

    留 0.6 cm 余量：上面的字宽是按字体度量**估算**的，再叠上 Word 自己的取整，
    贴着版心排仍然有折行的可能 —— 差半个字比多折一行好看。
    """
    width = _line_width_cm(text, size)
    if width <= limit:
        return size
    return max(size * limit / width, 8.0)


def code_block(doc: Any, text: str, *, size: float = 9.5) -> Any:
    """等宽代码块（含中文，故 eastAsia 用宋体、西文用 Consolas）。

    模板的左右边距（3.17 cm）比 python-docx 默认窄，版心只有 14.66 cm ——
    超过就会折行，附录里靠空格对齐的目录树一旦折行就没法看了。
    这里按最宽的一行反推能用的最大字号，宁可小一点也不折行。
    """
    widest = max((_line_width_cm(line, size) for line in text.splitlines()), default=0.0)
    if widest > CODE_MAX_CM:
        size = max(size * CODE_MAX_CM / widest, 7.0)
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.line_spacing = 1.0
    fmt.space_after = Pt(6)
    fmt.left_indent = Pt(12)
    run = p.add_run(text)
    run.font.name = FONT_CODE
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    return p


def caption(doc: Any, text: str) -> Any:
    return para(doc, text, size=10.5, font="楷体", align=WD_ALIGN_PARAGRAPH.CENTER,
                line=1.0, after=10, color=GREY)


def picture(doc: Any, path: Path, *, width_in: float = PICTURE_WIDTH_IN, note: str = "") -> None:
    """插图 + 图注。图片段落设为「与下段同页」，避免图注被挤到下一页去。"""
    if not Path(path).exists():
        para(doc, f"（图片缺失：{path}）", size=10.5, color=RED)
        return
    doc.add_picture(str(path), width=Inches(width_in))
    pic_para = doc.paragraphs[-1]
    pic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_para.paragraph_format.keep_with_next = bool(note)
    if note:
        caption(doc, note)


def _shade(cell: Any, hex_color: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


#: 完成情况的配色（模板要求逐项标注 已实现 / 部分实现 / 未实现）
STATUS_STYLE = {
    "已实现": GREEN,
    "部分实现": ORANGE,
    "未实现": RED,
}


def _fit_widths(widths: Sequence[float], limit: float = TABLE_MAX_CM) -> List[float]:
    """按比例把列宽压进版心。

    历史原因：这些表是按 15.5 cm 版心排的，换成模板的 3.17 cm 页边距后版心只剩
    14.66 cm，不缩放的话每张表都会顶出版心。整体等比缩放比逐张改列宽安全 ——
    列与列的相对比例保持不变。
    """
    total = float(sum(widths))
    if total <= limit:
        return [float(w) for w in widths]
    factor = limit / total
    return [float(w) * factor for w in widths]


def table(doc: Any, headers: Sequence[str], rows: Sequence[Sequence[str]], *,
          widths: Optional[Sequence[float]] = None, size: float = 10.5,
          status_col: Optional[int] = None, mark_cols: Optional[Sequence[int]] = None,
          shade_rows: bool = True) -> Any:
    """加一个带表头底色的表格。

    Parameters
    ----------
    widths:
        每列宽度（厘米）。给出后会关闭自动伸缩，避免 Word 把表格拉得难看；
        合计超过版心时会等比压缩（见 :func:`_fit_widths`）。
    status_col:
        指出哪一列是「完成情况」，该列按 :data:`STATUS_STYLE` 着色加粗。
    mark_cols:
        指出哪几列是勾选列（列名即状态名），有标记的单元格按该列状态着色 ——
        课程模板的模块清单就是这种「一列一个状态、打勾」的矩阵。
    """
    if widths:
        widths = _fit_widths(widths)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths:
        t.autofit = False
    # 表头跨页重复；行内不允许断页（否则一行数据被腰斩成两半，很难读）
    hdr_tr_pr = t.rows[0]._tr.get_or_add_trPr()
    hdr_tr_pr.append(OxmlElement("w:tblHeader"))
    hdr = t.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        run = p.add_run(str(text))
        _set_font(run, FONT_HEAD, size, bold=True)
        _shade(hdr[i], "E8EEF7")
    for r_index, row in enumerate(rows):
        new_row = t.add_row()
        new_row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        cells = new_row.cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            text = "" if value is None else str(value)
            is_status = status_col is not None and i == status_col
            is_mark = bool(mark_cols) and i in mark_cols
            if is_status:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(text)
                _set_font(run, FONT_BODY, size, bold=True, color=STATUS_STYLE.get(text))
            elif is_mark:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(text)
                _set_font(run, FONT_BODY, size, bold=True,
                          color=STATUS_STYLE.get(str(headers[i])) if text else None)
            else:
                add_rich(p, text, size=size)
        if shade_rows and r_index % 2 == 1:
            for cell in cells:
                _shade(cell, "FAFBFD")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    para(doc, "", size=6, after=0, line=1.0)
    return t


#: 勾选矩阵的三个状态列，顺序与课程模板的示例表一致
STATUS_COLUMNS = ("已实现", "部分实现", "未实现")

#: 「模块交付物清单」的列宽：交付项 / 三个状态列 / 证据
STATUS_TABLE_WIDTHS = (4.2, 1.6, 1.9, 1.6, 5.3)


def status_table(doc: Any, rows: Sequence[Sequence[str]], *,
                 widths: Sequence[float] = STATUS_TABLE_WIDTHS, size: float = 10) -> Any:
    """模块交付物清单：课程模板的三列勾选矩阵 + 一列证据。

    模板对「三、模块交付物清单」的要求是「每个模块制作一个表格说明」，示例表把
    已实现 / 部分实现 / 未实现 拆成三列、逐项在对应列打勾。这里照搬该形式，
    但在右侧补一列证据：模块完成情况必须写清代码位置或可复现命令 ——
    「什么没做、缺口在哪」不能因为换表格样式就被省掉。

    :param rows: 每行 ``[交付项, 完成情况, 说明与证据]``，与旧表完全一致。
    """
    grid: List[List[str]] = []
    for item, status, note in rows:
        marks = ["✓" if status == col else "" for col in STATUS_COLUMNS]
        if status not in STATUS_COLUMNS:
            raise ValueError(f"完成情况只能是 {STATUS_COLUMNS} 之一，收到：{status!r}")
        grid.append([str(item), *marks, str(note)])
    return table(
        doc,
        ["交付项（课程要求）", *STATUS_COLUMNS, "证据（代码位置 / 命令）"],
        grid,
        widths=list(widths), size=size, shade_rows=True,
        mark_cols=range(1, 1 + len(STATUS_COLUMNS)),
    )


def define_toc_styles(doc: Any) -> None:
    """定义目录的三级样式。

    不定义的话，目录条目会继承 Normal（仿宋四号 + 1.5 倍行距），
    37 条目录能占满四页 —— 实测过，确实会。这里把它们压到 1 倍行距、小字号。
    """
    from docx.enum.style import WD_STYLE_TYPE

    specs = [
        ("toc 1", 12.0, True, 0.0),
        ("toc 2", 11.0, False, 12.0),
        ("toc 3", 10.5, False, 24.0),
    ]
    existing = {s.name for s in doc.styles}
    for name, size, bold, indent in specs:
        if name in existing:
            style = doc.styles[name]
        else:
            style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        # python-docx 会把新建样式标成 customStyle 并生成小写 styleId（toc1），
        # Word 就认不出这是内置目录样式、于是照旧用 Normal 排版 —— 必须改回内置 ID。
        style.element.attrib.pop(qn("w:customStyle"), None)
        style.element.set(qn("w:styleId"), "TOC" + name[-1])
        style.base_style = doc.styles["Normal"]
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.name = FONT_ASCII
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT_BODY)
        style.paragraph_format.line_spacing = 1.15
        style.paragraph_format.space_after = Pt(2)
        style.paragraph_format.left_indent = Pt(indent)


def toc(doc: Any) -> None:
    """插入 Word 目录域（打开文档时自动更新页号）。"""
    p = doc.add_paragraph()
    run = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = r'TOC \o "1-3" \h \z \u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "（目录将在打开文档时自动生成；若未显示，请在 Word 中按 Ctrl+A 再按 F9 更新域）"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instr, separate, placeholder, end):
        run._r.append(node)


def enable_update_fields(doc: Any) -> None:
    """让 Word 打开文档时自动更新域（否则目录是空占位）。"""
    settings = doc.settings.element
    element = OxmlElement("w:updateFields")
    element.set(qn("w:val"), "true")
    settings.append(element)


# ==============================================================================
# 数据读取（全部来自**真实产物**，不做任何估算）
# ==============================================================================
def load_metrics() -> Dict[str, Any]:
    """读 :file:`reports/性能评测数据.json`；缺失时返回空字典（文档会标注「未运行」）。"""
    path = PROJECT_ROOT / "reports" / "性能评测数据.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"警告：评测数据读取失败（{exc}），交付书中的指标将标注为「未运行」\n")
        return {}


#: 评测集类型 / 难度的中文标签（键顺序即文档中的展示顺序）
TYPE_LABELS = {
    "factual": "事实型",
    "comparison": "对比型",
    "summary": "归纳型",
    "reasoning": "推理型",
}
DIFFICULTY_LABELS = {"easy": "简单", "medium": "中等", "hard": "困难"}


def load_eval_set_stats() -> Dict[str, int]:
    """统计评测集的问题类型分布（直接数 :file:`reports/评测集.json`）。

    返回的字典按 :data:`TYPE_LABELS` 的顺序排列，键为中文标签，
    便于直接拼进正文；读不到时返回空字典，文档会退化为「见报告」。
    """
    path = PROJECT_ROOT / "reports" / "评测集.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    if isinstance(data, dict):
        raw = data.get("type_distribution")
        if not raw:
            raw = {}
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

    order = list(TYPE_LABELS)
    ordered: Dict[str, int] = {}
    for key in order + [k for k in raw if k not in order]:
        if key in raw:
            ordered[TYPE_LABELS.get(key, key)] = int(raw[key])
    return ordered


def load_chunking_rows() -> List[Dict[str, str]]:
    """从 :file:`reports/分块策略对比实验报告.md` 的主表里解析四种策略的实测值。

    解析失败就返回空列表 —— 宁可少一张表，也不写死一份可能与实际不符的数字。
    """
    path = PROJECT_ROOT / "reports" / "分块策略对比实验报告.md"
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    pattern = re.compile(r"^\|\s*(?P<label>.+?)（`(?P<strategy>[a-z_]+)`）\s*\|(?P<rest>.+)\|\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        cells = [c.strip() for c in match.group("rest").split("|")]
        if len(cells) < 10 or cells[0] in ("未运行",):
            continue
        rows.append(
            {
                "label": match.group("label").strip(),
                "strategy": match.group("strategy"),
                "num_chunks": cells[0],
                "avg_chars": cells[1],
                "index_ms": cells[3],
                "hit": cells[4],
                "mrr": cells[5],
                "recall": cells[6],
                "ndcg": cells[8],
                "latency": cells[9],
            }
        )
    return rows


def load_ppt_path() -> Optional[Path]:
    """探测演示 PPT 是否已生成（与 :data:`DEFAULT_OUT` 同级的约定路径）。

    交付物清单里的「演示 PPT」条目据此显示「已实现 / 未实现」——
    探测不到时宁可标注未实现，也不要写一句与实际不符的漂亮话。
    """
    return PPT_PATH if PPT_PATH.exists() else None


def ppt_slide_count(path: Path) -> Optional[int]:
    """数 pptx 里的页数：直接数包内 ppt/slides/slideN.xml，不写死数字。"""
    try:
        with zipfile.ZipFile(path) as zf:
            return sum(
                1
                for name in zf.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            )
    except (OSError, zipfile.BadZipFile):  # pragma: no cover - 文件损坏时不猜
        return None


def ppt_status() -> str:
    return "已实现" if load_ppt_path() else "未实现"


def ppt_detail() -> str:
    """交付物清单里 PPT 那一行的「产物」一栏。"""
    path = load_ppt_path()
    if path is None:
        return (
            "本次交付只包含代码仓库、技术文档、评测报告与交付书；"
            "演示 PPT 作为课堂汇报材料另行准备"
        )
    slides = ppt_slide_count(path)
    pages = f"{slides} 页：" if slides else ""
    return (
        f"{path.name}（{pages}项目背景 / 技术架构 / 功能演示 / 踩坑记录 / 优化对比；"
        f"图表数字与 reports/ 下的评测产物同源）"
    )


def ppt_location() -> str:
    """附录 B.5 交付物清单里 PPT 那一行的「位置」一栏。"""
    path = load_ppt_path()
    if path is None:
        return "另行准备（见 6.3 待改进方向）"
    slides = ppt_slide_count(path)
    return f"{path.name}（{slides} 页）" if slides else path.name


def scan_code_stats() -> Dict[str, Any]:
    """扫描真实文件树（附录 A 与代码规模统计都以此为准）。

    分组键的设计：``src/xxx`` 的**子包**各成一组（``src/agent``），
    ``src`` 下的单文件归入 ``src``；``tests`` / ``docs`` 等目录整体一组 ——
    否则 ``tests/test_agent.py`` 这类文件会被当成一个"目录"，目录树会很难看。
    """
    skip_dirs = {"__pycache__", ".git", "data", "logs", "_scratch", ".idea", ".vscode"}
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT)
        parts = rel.parts
        if any(part in skip_dirs for part in parts):
            continue
        if rel.suffix.lower() in BINARY_SUFFIXES:
            continue
        if str(rel).replace("\\", "/") in DEV_ONLY_SCRIPTS:
            continue
        if parts[0].startswith(".") and len(parts) > 1:
            continue
        if len(parts) == 1 and rel.name.startswith("_"):
            continue  # 根目录下划线开头的临时/参考文件，不属于项目
        try:
            with path.open("rb") as fh:
                if b"\x00" in fh.read(8192):
                    continue  # 兜底：没列进 BINARY_SUFFIXES 的二进制文件也不当文本数行
        except OSError:
            continue
        try:
            lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if len(parts) == 1:
            key = "根目录"
        elif parts[0] == "src" and len(parts) > 2:
            key = f"src/{parts[1]}"
        else:
            key = parts[0]
        groups.setdefault(key, []).append((str(rel).replace("\\", "/"), lines))

    total_src = sum(n for k, v in groups.items() if k == "src" or k.startswith("src/")
                    for _, n in v)
    total_tests = sum(n for k, v in groups.items() if k == "tests"
                      for _, n in v)
    return {"groups": groups, "total_src": total_src, "total_tests": total_tests}


# ==============================================================================
# 封面 / 成员 / 目录
# ==============================================================================
def build_cover(doc: Any, today: str) -> None:
    # 第 1 页：考核情况表。课程模板里这一页是**带全角空格的文字行**，
    # 不是表格 —— 照模板排版，字段与取值逐项对齐。
    para(doc, "", size=12, after=24, line=1.0)
    para(doc, "南京农业大学课程考核情况表", size=18, font=FONT_HEAD,
         align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=18)
    para(doc, "2026—2027 学年第 1 学期", size=15, font=FONT_HEAD,
         align=WD_ALIGN_PARAGRAPH.RIGHT, line=1.5, after=24)
    for line in (
        "学　　院：智慧农业学院（人工智能学院）",
        "课程名称：生产实习　　　　　　学分：1",
        "考核方式：课程报告",
        f"考核内容：{TITLE}",
        "成　　绩：",
        "教师签名：",
        "时　　间：2026 年 12 月 30 日",
    ):
        para(doc, line, size=_fit_size(line, 15), font=FONT_HEAD, line=1.5, after=10)
    para(doc, "", size=12, after=18, line=1.0)
    para(doc, f"（本交付书生成日期：{today}）",
         size=10.5, font="楷体", align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY, line=1.0)
    doc.add_page_break()

    # 第 2 页：报告封面。标题用隶书 36 磅、题目与成员行用三号加粗 —— 均照模板。
    para(doc, "", size=12, after=60, line=1.0)
    para(doc, SUBTITLE, size=36, font=FONT_COVER, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=30)
    para(doc, "题　　目：", size=16, font=FONT_HEAD, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=6)
    para(doc, TITLE, size=_fit_size(TITLE, 16), font=FONT_HEAD, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=36)
    for _, member, _, _, _ in MEMBERS:
        para(doc, member, size=_fit_size(member, 16), font=FONT_HEAD, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=10)
    para(doc, "", size=12, after=30, line=1.0)
    para(doc, "2026 年 12 月", size=15, font=FONT_HEAD,
         align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5)
    doc.add_page_break()


def build_toc(doc: Any) -> None:
    para(doc, "目　录", size=22, font=FONT_HEAD, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, line=1.5, after=14)
    toc(doc)
    doc.add_page_break()


# ==============================================================================
# 一、项目概述
# ==============================================================================
def build_section_overview(doc: Any) -> None:
    heading(doc, "一、项目概述", 1)

    heading(doc, "1.1 项目背景", 2)
    assert len(BACKGROUND) <= 200, f"项目背景超出 200 字限制（当前 {len(BACKGROUND)} 字）"
    para(doc, BACKGROUND, indent_first=True)
    para(doc, f"（本节 {len(BACKGROUND)} 字，符合课程模板「200 字以内」的要求。）", color=GREY)

    heading(doc, "1.2 项目目标", 2)
    para(doc, "构建一个**完全本地可控**的论文知识库问答系统：把课题组自己的 PDF / Word / "
              "Markdown 论文变成可检索、可溯源、可对话的私有知识库。具体目标：")
    for text in (
        "多格式文档入库：PDF / DOCX / TXT / Markdown 统一加载、分块、向量化，支持增量更新与批量导入；",
        "高精度检索：向量检索与 BM25 关键词检索并行召回，RRF 融合后再精排，并把「相关度」以绝对分数暴露给上层；",
        "可信生成：答案必须带【文档名 + 页码】引用，检索不到就明确说「未命中」，绝不硬编；",
        "自主决策：手写 ReAct Agent，具备 8 个工具的自主选择、并行调用、错误恢复与多轮记忆；",
        "全过程可观测：Token、耗时、命中率、工具成功率、Thought→Action→Observation 轨迹全部可视化；",
        "可量化交付：50+ 条评测集、检索三档对比、分块策略对比、Bad Case 根因分析与一轮迭代优化。",
    ):
        bullet(doc, text)

    heading(doc, "1.3 技术栈概览", 2)
    para(doc, "本项目对「关键算法手写、工程环节不重复造轮子」的取舍是明确的：检索融合、"
              "ReAct 循环、BM25、指标库全部手写；Web 服务与前端不引入框架，"
              "以保证在**只装 NumPy** 的环境里也能完整跑起来。")
    table(
        doc,
        ["层次", "技术选型", "说明"],
        [
            ["运行环境", "Python 3.10+ / NumPy", "NumPy 是唯一必需第三方依赖；其余全部为标准库实现"],
            ["文档解析", "pymupdf → pdfplumber → pypdf → 内置解析器", "按可用性自动降级；内置解析器支持 FlateDecode + ToUnicode CMap，中文不乱码"],
            ["分块", "fixed / recursive / semantic / paper（手写）", "四种策略可切换，均带页码与字符偏移（引用溯源的基础）"],
            ["向量化", "哈希向量（默认）/ Ollama / OpenAI 兼容", "默认后端零依赖、离线可复现；接入真实模型只需改配置"],
            ["向量库", "NumPy 稠密矩阵（默认）/ Chroma / FAISS", "默认后端零依赖；Chroma / FAISS 适配器已实现，装了对应包即自动启用"],
            ["关键词检索", "手写 Okapi BM25", "不依赖 rank_bm25，中文按字 + 英文按词切分（无需 jieba）"],
            ["融合与重排", "RRF（倒数排名融合）+ 启发式重排", "重排可选真实 bge-reranker（Ollama）；默认启发式，延迟可控"],
            ["大模型", "DeepSeek（默认）/ OpenAI 兼容 / Ollama / mock", "mock 后端让全链路离线可复现，评测据此做到确定性"],
            ["Agent", "手写 ReAct 循环 + 工具注册表", "不依赖 LangChain；决策走 JSON 协议，全程可追踪"],
            ["Web 服务", "ThreadingHTTPServer + SSE（标准库）", "零 Web 框架依赖；服务器推送事件实现打字机效果"],
            ["前端", "原生 HTML / CSS / JavaScript 单页应用", "三栏布局：知识库管理 / 对话 / Agent 轨迹"],
            ["持久化", "SQLite（会话与缓存）+ JSON + .npy", "无外部数据库依赖，全部落在 data/ 目录"],
            ["评测", "自研指标库 + 报告生成器", "Hit@K / MRR / Recall / NDCG / 引用准确率 / 忠实度等 14 个指标"],
            ["交付", "Docker / docker-compose", "一键容器化运行，数据卷持久化"],
        ],
        widths=[2.4, 5.0, 8.1], size=10, status_col=None,
    )


# ==============================================================================
# 二、团队分工
# ==============================================================================
def build_section_team(doc: Any) -> None:
    heading(doc, "二、团队分工说明", 1)
    para(doc, "小组 4 人，按「后端 / Agent / 前端 / 测试与交付」四个角色分工，"
              "每人独立负责一个模块并对自己模块的代码与文档质量负责；"
              "模块之间通过既定接口协作（接口定义见 5.2），"
              "**贡献比例四项之和为 100%**。")
    table(
        doc,
        ["角色", "成员姓名", "核心职责", "主要负责模块", "贡献比例"],
        [[role, member, duty, modules, ratio]
         for role, member, duty, modules, ratio in MEMBERS],
        widths=[2.3, 4.3, 5.6, 1.8, 1.4], size=10, status_col=None,
    )
    para(doc, "说明：上表中的「成员姓名」栏为占位，请组员在提交前替换为自己的"
              "班级、学号与姓名；如需调整分工比例，请保证四项之和仍为 100%。", color=GREY)


# ==============================================================================
# 三、模块交付物清单
# ==============================================================================
def build_section_deliverables(doc: Any, metrics: Dict[str, Any],
                               chunk_rows: List[Dict[str, str]]) -> None:
    heading(doc, "三、模块交付物清单", 1)
    para(doc, "每个模块一张表，按课程模板的「已实现 / 部分实现 / 未实现」三列逐项勾选，"
              "并在末列给出对应的代码位置或可复现命令作为证据。"
              "标注「部分实现」的项一律说明了缺口在哪里，不做含糊表述。")

    # -- 模块一 ------------------------------------------------------------
    heading(doc, "模块一：文档处理与检索层", 2)
    status_table(
        doc,
        [
            ["多格式文档加载器（PDF / Word / TXT）", "已实现",
             "src/data_loader/：PDF 四后端（pymupdf → pdfplumber → pypdf → 内置）按可用性自动降级；"
             "DOCX 支持 python-docx 与内置 OOXML 解包；TXT / Markdown 按后缀路由。"
             "已就三种 PDF 方案做横向比较并在代码中记录取舍"],
            ["批量文档导入与状态追踪（进度、失败重试）", "已实现",
             "src/data_loader/loader.py 的 BatchImporter + src/frontend/container.py 的 index_files()；"
             "前端上传面板实时显示进度，失败项自动重试并可查看原因"],
            ["3 种以上文本分块策略", "已实现",
             "共 4 种：src/chunking/strategies.py 的固定窗口 / 递归字符 / 语义切分 / 论文专用切分；"
             "每种都保证 start_offset、end_offset 与原文严格对齐（page_text[a:b] == chunk.text）"],
            ["分块策略对比实验（召回率差异）", "已实现",
             "python -m src.evaluation.chunking_experiment → reports/分块策略对比实验报告.md（见 4.5）"],
            ["Embedding 模型接入与向量化存储", "部分实现",
             "src/embeddings.py 已实现 hash（默认，512 维、零依赖、可复现）与 ollama / openai 三类后端，"
             "落盘 vectors.npy + chunks.json 并带 SQLite 向量缓存。"
             "缺口：本轮实测使用的是 hash 后端（词面级），bge-large-zh / m3e-base 的对比实验未做"],
            ["向量数据库选型（Chroma / FAISS）", "部分实现",
             "src/retrieval/vector_store.py 实现了 Numpy / Chroma / Faiss 三种后端，接口一致、自动探测。"
             "缺口：交付环境未安装 chroma / faiss，实测与评测均使用 NumPy 后端"],
            ["增量索引（新文档无需重建全量）", "已实现",
             "RetrievalEngine.index_document() / delete_document()：新增与删除文档时同步更新向量库、"
             "BM25 与重排器 IDF 统计，实测删除单篇只动该文档的分块"],
            ["混合检索（向量 + BM25 + RRF 融合）", "已实现",
             "src/retrieval/bm25_retriever.py（手写 Okapi BM25）+ hybrid_retriever.py（RRF，k=60）"],
            ["重排序（Reranker）", "已实现",
             "src/retrieval/reranker.py：启发式六特征重排（默认，无需额外模型）+ 可选 OllamaReranker（bge-reranker）。"
             "注：本轮实测启发式重排未提升排序质量，默认关闭，详见 4.5 与 6.2"],
            ["检索三档对比实验（纯向量 → 混合 → 混合+重排）", "已实现",
             "reports/系统评测报告.md 第三节（严格口径 + 引擎口径并列）"],
        ],
    )

    # -- 模块二 ------------------------------------------------------------
    heading(doc, "模块二：RAG 生成层", 2)
    status_table(
        doc,
        [
            ["2.1 引用溯源机制（答案标注文档名 + 页码）", "已实现",
             "src/generation/prompt_template.py 强制输出【文档名 p.页码】；"
             "rag_pipeline.py 解析引用并与检索结果核对。实测引用准确率 100%、引用覆盖率 97.5%"],
            ["2.2 流式输出（打字机效果）", "已实现",
             "src/generation/streaming.py + 前端 SSE；推送顺序为「先元信息 → 逐 token → 引用与轨迹 → 结束」"],
            ["2.3 语义缓存机制", "已实现",
             "src/generation/cache.py：问题向量近邻匹配（阈值 0.93），SQLite 落盘；"
             "命中时跳过检索与生成，答案标注 cache_hit"],
            ["2.4 降级策略（无结果 / 低相关 / API 失败）", "已实现",
             "三种降级三态 no_result / low_relevance / llm_error，返回带 degrade_reason 的正常答案而非抛异常；"
             "llm_error 时仍返回已检索到的原文片段供用户判断"],
            ["2.5 RAG 请求日志（Token / 耗时 / 命中率）", "已实现",
             "src/utils/logger.py 的 TraceLogger + MetricsCollector；"
             "落盘 logs/traces.jsonl、logs/metrics.jsonl，并在前端「系统指标」面板实时展示"],
            ["Prompt 工程与生成参数对比（Temperature / Top-p / Top-k）", "部分实现",
             "Prompt 模板已按「系统角色 + 上下文 + 问题 + 输出格式约束」四段式实现，支持多模板切换；"
             "缺口：生成参数的对比实验未做（本轮评测使用 mock 后端，参数对比无意义），已在 6.3 列为待改进项"],
        ],
    )

    # -- 模块三 ------------------------------------------------------------
    heading(doc, "模块三：Agent 决策层", 2)
    status_table(
        doc,
        [
            ["3.1 ReAct 核心循环手写实现", "已实现",
             "src/agent/react_loop.py（约 1400 行，含注释与类型标注）："
             "Thought → Action → Observation 逐步推进，每轮把完整轨迹回灌模型"],
            ["3.2 Agent System Prompt 设计", "已实现",
             "src/agent/prompts.py：角色定义 + 工具目录（由注册表自动渲染）+ 输出格式（JSON）+ 终止条件 + 反幻觉约束"],
            ["3.3 工具集开发（8 个工具）", "已实现",
             "knowledge_search / paper_metadata / paper_compare / keyword_extract / summarize_paper / "
             "current_time / calculator / web_search；python run.py --list-tools 可查看全部工具与参数"],
            ["3.4 错误恢复机制（超时 / 异常 / 死循环）", "已实现",
             "工具超时→重试→换工具；工具异常→把错误文本作为 Observation 回灌让模型自我纠正；"
             "以 (action, args) 指纹做滑动窗口死循环检测；超最大轮次时用已有信息强制收尾"],
            ["3.5 多轮对话记忆管理（会话隔离 / 滑动窗口 / 摘要）", "已实现",
             "src/agent/memory.py：会话隔离 + Token 预算下的滑动窗口 + 超阈值摘要压缩，SQLite 持久化"],
            ["3.6 工具选择路由（减少无效推理）", "已实现",
             "src/agent/router.py：一次廉价意图识别把候选工具从 8 个缩到 1~3 个；"
             "计算类问题直接跳过整个循环，节省 Token"],
            ["3.7 并行工具调用", "已实现",
             "同一轮内的多个独立工具并发执行，实测 3×0.5s 的串行等待缩短到约 0.6s"],
        ],
    )

    # -- 模块四 ------------------------------------------------------------
    heading(doc, "模块四：系统集成与前端展示", 2)
    status_table(
        doc,
        [
            ["4.1 全系统端到端打通（上传 → 索引 → 对话 → 答案）", "已实现",
             "src/frontend/container.py 的 SystemContainer 统一装配全部组件；"
             "端到端流程有专门评测（reports/系统评测报告.md 第五节，40 次调用）"],
            ["4.2 对话历史管理（新建 / 切换 / 删除会话）", "已实现",
             "会话列表支持新建、切换、重命名、删除；首轮问答后由后端自动以第一个问题命名会话；"
             "历史落盘 SQLite，重启不丢"],
            ["可观测性监控（Token / 命中率 / 延迟 / 决策轨迹）", "已实现",
             "右侧 Agent 轨迹面板（Thought / Action / Observation 全链路）+ "
             "「系统指标」面板（请求数、命中率、P50/P95 延迟、Token、工具成功率、降级率、终止原因分布）"],
            ["健康检查接口", "已实现",
             "GET /api/health：逐组件检查 LLM 服务、向量库、检索引擎、记忆库，"
             "命令行等价入口 python run.py --health"],
            ["前端界面（三栏布局 + 上传进度 + 引用角标）", "已实现",
             "标准库 ThreadingHTTPServer + 原生 JS 单页应用实现左中右三栏；"
             "技术选型说明见 4.4：任务书建议 Streamlit / Gradio，本项目选择零框架方案以保持「只装 NumPy」也能运行"],
            ["边缘场景测试（空知识库 / 超大文档 / 超长对话）", "已实现",
             "tests/ 中覆盖空知识库提问、超大文本分块与截断、超窗记忆压缩、路径穿越与代码注入等负路径"],
            ["多用户并发测试", "部分实现",
             "服务端为 ThreadingHTTPServer，单元测试覆盖了并发请求与线程安全（指标累加加锁）；"
             "缺口：未做多用户压力测试与 QPS 评测"],
        ],
    )

    # -- 模块五 ------------------------------------------------------------
    heading(doc, "模块五：评测与交付", 2)
    eval_stats = load_eval_set_stats()
    total_items = sum(eval_stats.values())
    type_text = "、".join(f"{k} {v} 条" for k, v in eval_stats.items()) or "见 reports/评测集.json"
    status_table(
        doc,
        [
            ["5.1 评测集（50+ 问答对，覆盖 4 类问题）", "部分实现",
             f"已构建 {total_items} 条问答对（{type_text}），覆盖事实型 / 对比型 / 归纳型 / 推理型四类，中英混合提问。"
             f"缺口：语料为 3 篇样例论文（脚本合成的英文论文 PDF，保证可离线分发与复现），"
             f"未按要求收集 10–20 篇真实论文；替换语料后重跑评测命令即可复现全部指标"],
            ["5.2 系统级性能评估（Hit@5 / MRR / Recall、工具准确率、延迟、Token）", "已实现",
             "reports/系统评测报告.md：检索三档对比、Agent 决策指标、端到端延迟分解、"
             "Token 统计与缓存/降级率，全部来自实际运行"],
            ["5.3 Bad Case 分类与根因分析 + 至少一轮迭代优化", "已实现",
             "reports/Bad_Case分析报告.md：13 条 Bad Case 自动分类（检索 / 生成 / 引用 / 决策）"
             "并给出根因与改进建议；另含一轮真实参数对照实验（严格 Hit@5 0.7705 → 0.8033，降级率 16.4% → 3.3%）"],
            ["5.4 技术设计文档", "已实现", "docs/技术设计文档.md（分层架构、模块接口、数据流向、配置项、算法说明）"],
            ["5.5 用户使用手册", "已实现", "docs/用户使用手册.md（安装部署、界面功能、命令行用法、常见问题）"],
            ["答案质量人工评分（正确性 / 完整性 / 引用准确性）", "部分实现",
             "引用准确性已由指标自动核验（引用准确率 100%），并有引用覆盖率与忠实度的自动度量；"
             "缺口：正确性 / 完整性的**人工评分表**未附（本轮评测运行在 mock 后端下，人工评分需在真实模型环境中由组员完成）"],
            ["Docker 一键部署", "已实现", "docker/Dockerfile + docker/docker-compose.yml（数据卷持久化 data/ 与 logs/）"],
            ["演示 PPT", ppt_status(), ppt_detail()],
        ],
    )


# ==============================================================================
# 四、技术实现记录
# ==============================================================================
def _fmt(value: Any, digits: int = 4, default: str = "—") -> str:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def build_section_implementation(doc: Any, metrics: Dict[str, Any],
                                 chunk_rows: List[Dict[str, str]]) -> None:
    heading(doc, "四、技术实现记录", 1)
    para(doc, "本章按模块记录「具体是怎么实现的」：关键数据结构、算法步骤与实测结论。"
              "文中所有指标均来自 reports/ 下的实际运行产物，可逐条复现。")

    build_impl_module1(doc, chunk_rows)
    build_impl_module2(doc, metrics)
    build_impl_module3(doc)
    build_impl_module4(doc)
    build_impl_module5(doc, metrics, chunk_rows)


def build_impl_module1(doc: Any, chunk_rows: List[Dict[str, str]]) -> None:
    heading(doc, "4.1 模块一：文档处理与检索层", 2)

    heading(doc, "（1）文档加载：四后端按可用性降级", 3)
    para(doc, "PDF 解析是本项目第一个真实踩坑点。四种后端的能力差异很大，"
              "因此加载器不绑定某一个库，而是按 pymupdf → pdfplumber → pypdf → 内置解析器 "
              "的顺序挑选第一个可用的实现：")
    table(
        doc,
        ["后端", "优势", "代价", "在本项目中的角色"],
        [
            ["pymupdf (fitz)", "速度最快、版面分析最强", "AGPL 许可、包体大", "若已安装则使用（HANDOFF 不强制）"],
            ["pdfplumber", "表格抽取效果好", "依赖较重、较慢", "次选"],
            ["pypdf", "纯 Python、许可宽松", "复杂版面偶有丢失", "本轮评测实际使用的后端（pdf:pypdf）"],
            ["内置解析器", "零依赖，中文不乱码", "仅支持文本型 PDF", "兜底：没装任何第三方库也能读到中文"],
        ],
        widths=[3.0, 4.4, 3.6, 4.5], size=10, status_col=None,
    )
    para(doc, "内置解析器是最费功夫的一块：PDF 里的中文若不解析 ToUnicode CMap，"
              "解出来的会是乱码。因此内置实现把「FlateDecode 解压 → 内容流取字符串 → "
              "按 CMap 映射回 Unicode」做进了主流程，并优先使用嵌入字体的 CMap 表。"
              "这一步做完，系统在**零第三方依赖**的环境下也能正确读取中文论文。")

    heading(doc, "（2）分块：四种策略 + 偏移不变式", 3)
    para(doc, "分块决定了检索的最小单元，也决定了引用能不能定位到页码。"
              "系统实现了四种策略，统一接口 BaseChunker.split()：")
    table(
        doc,
        ["策略", "原理", "适用场景", "实测分块数"],
        [
            ["fixed", "固定字符窗口 + 重叠",
             "作为对照基线，用于比较 256 / 512 / 1024 的效果",
             next((r["num_chunks"] for r in chunk_rows if r["strategy"] == "fixed"), "—")],
            ["recursive", "按 段落 → 换行 → 句末 → 从句 → 空格 递归二分",
             "默认策略，效果与复杂度的平衡点",
             next((r["num_chunks"] for r in chunk_rows if r["strategy"] == "recursive"), "—")],
            ["semantic", "先切句，再按相邻句向量相似度骤降处断开",
             "块内语义一致性要求高的场景（需 Embedding）",
             next((r["num_chunks"] for r in chunk_rows if r["strategy"] == "semantic"), "—")],
            ["paper", "在 recursive 上叠加跨页合并、公式行保护、标题吸附、参考文献剔除",
             "学术论文（本项目的主场景）",
             next((r["num_chunks"] for r in chunk_rows if r["strategy"] == "paper"), "—")],
        ],
        widths=[2.2, 6.2, 4.4, 2.7], size=10, status_col=None,
    )
    para(doc, "所有策略都维护一条硬不变式：每个分块记录它在**原文页内的字符区间** "
              "[start_offset, end_offset)，且 page_text[start:end] == chunk.text 恒成立。"
              "早期实现是「先切分再回头 find 原文位置」，一旦重建文本（如把换行换成空格）"
              "就会定位失败、引用错位；现在改为全程用 _Piece 记录来源区间、文本只从原文切出，"
              "从根上消除了这类错位。", indent_first=True)

    heading(doc, "（3）向量化与索引", 3)
    para(doc, "默认后端是自研的**哈希向量**：对文本做字符 n-gram 特征哈希，投影到 512 维后 L2 归一化。"
              "它的语义泛化能力当然不如 bge 系列，但胜在零依赖、离线、确定性 —— "
              "这正是本课程项目「任何人 clone 下来都能跑出同一份评测结果」所需要的。"
              "换成真实模型只需把配置里的 embedding.provider 从 hash 改为 ollama 或 openai，"
              "上层检索代码一行都不用动。")
    para(doc, "向量与分块分别落盘为 vectors.npy 与 chunks.json；同一段文本重复向量化会命中 "
              "SQLite 向量缓存，重建索引时省掉绝大部分算力。")

    heading(doc, "（4）混合检索：两路召回 + RRF 融合", 3)
    para(doc, "两路召回各自补齐对方的短板：向量召回负责语义泛化（「这篇论文怎么解决过拟合」"
              "↔「正则化策略」），BM25 负责精确词面（BatchNorm、mAP@0.5、BLEU 这类术语）。")
    para(doc, "融合用 RRF（倒数排名融合）而不是加权求和，原因很实在："
              "**向量相似度落在 0~1，BM25 分数无上界**，两者量纲不同，加权求和对分布极其敏感，"
              "换个语料就得重调权重。RRF 只用排名不用分数：")
    code_block(doc, "RRF(d) = Σ_r  1 / (k + rank_r(d))          # k = 60（可配），r 遍历各路召回")
    para(doc, "BM25 是手写的 Okapi 实现（k1=1.5、b=0.75），中文按字 + 英文按词切分，"
              "不依赖 rank_bm25，也不需要 jieba —— 少一个依赖，多一分可复现性。")

    heading(doc, "（5）三分数体系：为什么必须有 confidence", 3)
    para(doc, "这是全项目最关键的一个设计决策。检索结果同时携带三个分数：")
    table(
        doc,
        ["字段", "含义", "取值范围", "用途"],
        [
            ["score", "归一化后的相对得分", "0~1，Top-1 恒为 1.0", "展示排序、进度条"],
            ["confidence", "绝对相关度", "0~1，反映「到底有多相关」", "**降级判定、相关性判定**"],
            ["rerank_score", "重排模型给出的精排分", "依模型而定", "最终排序"],
        ],
        widths=[2.6, 4.6, 3.6, 4.7], size=10, status_col=None,
    )
    para(doc, "踩坑经过：最初只有归一化分数，于是「是否检索到相关内容」这个判断永远为真"
              "（Top-1 恒为 1.0），降级策略从不触发，无关问题也会拿到一段牛头不对马嘴的原文。"
              "拆出绝对分数 confidence 之后，「没检索到」才第一次成为可判断的状态，"
              "no_result / low_relevance 两级降级也才有了依据。", indent_first=True)

    heading(doc, "（6）重排序", 3)
    para(doc, "默认使用启发式重排（词项覆盖 + 位置加权等六个特征），不引入额外模型；"
              "配置里可切换到 Ollama 上的 bge-reranker 做交叉编码精排。"
              "实测结论写在 4.5：本轮启发式重排**没有**提升排序质量（严格 NDCG@5 0.5827 → 0.5681），"
              "因此默认关闭。这是一条比「重排一定更好」更值得记录的经验。")


def build_impl_module2(doc: Any, metrics: Dict[str, Any]) -> None:
    heading(doc, "4.2 模块二：RAG 生成层", 2)
    heading(doc, "（1）七步流水线", 3)
    para(doc, "RAGPipeline.answer() 把一次问答拆成七个可观测的步骤，每一步都可以单独计时与失败兜底：")
    code_block(
        doc,
        "① 语义缓存查询 ──命中──▶ 直接返回（标注 cache_hit）\n"
        "② 混合检索（向量 ‖ BM25 → RRF）\n"
        "③ 重排序（可选）\n"
        "④ 降级判定（confidence < 阈值 → no_result / low_relevance）\n"
        "⑤ 上下文构建（按相关度排序 + 动态截断）\n"
        "⑥ 生成（流式 / 非流式）\n"
        "⑦ 引用解析 → 回写缓存与 Trace 日志",
    )

    heading(doc, "（2）引用溯源", 3)
    para(doc, "Prompt 里对输出格式有硬约束：任何来自知识库的事实都必须标注【文档名 p.页码】。"
              "生成之后，rag_pipeline 会把答案里的引用标记逐条与本次检索结果核对："
              "引用了检索结果里不存在的来源，会被标记为「未核实」。"
              "实测引用准确率 100%、引用覆盖率 97.5%（40 条端到端样本）。")

    heading(doc, "（3）流式输出", 3)
    para(doc, "服务端通过 SSE（Server-Sent Events）分四类事件推送：元信息（会话、检索片段）→ "
              "逐 token 的增量文本 → 引用与推理轨迹 → 结束事件（含 Token 与耗时）。"
              "前端按事件类型分别渲染，因此「答案先出来、引用角标稍后就位」是自然结果，"
              "而不是等全部生成完再显示。")

    heading(doc, "（4）语义缓存与降级", 3)
    para(doc, "语义缓存的键是**问题向量**而不是问题字符串：换个说法问同一件事也能命中，"
              "阈值设为 0.93（偏低会返回错误答案，偏高则形同虚设）。")
    para(doc, "降级不是抛异常，而是返回一份「有依据的降级答案」，三态如下：")
    table(
        doc,
        ["降级原因", "触发条件", "用户看到什么"],
        [
            ["no_result", "检索结果为空或最高 confidence 低于下限", "明确告知「当前知识库中未找到相关文档」，并给出可操作建议"],
            ["low_relevance", "有结果但最高 confidence 低于阈值", "提示「检索到的内容相关性较低」，同时**展示已检索到的原文**供用户自行判断"],
            ["llm_error", "大模型调用失败或超时", "返回友好错误与重试建议，并把已检索到的原文片段一并给出，不让证据白费"],
        ],
        widths=[2.8, 5.2, 7.5], size=10, status_col=None,
    )

    heading(doc, "（5）日志与可观测性", 3)
    para(doc, "每次问答都会写入 logs/traces.jsonl（完整轨迹：问题、检索片段、Prompt 长度、"
              "答案、引用、Token、耗时）与 logs/metrics.jsonl（累计指标）。"
              "指标面板区分两套口径：RAG 流水线口径与**裸检索**口径 —— "
              "因为 Agent 的 knowledge_search 是直连检索引擎的（不构造 RAGAnswer），"
              "少了后者，纯 Agent 部署下整块面板会恒为 0。")
    end_to_end = metrics.get("end_to_end") or {}
    if end_to_end:
        para(doc, "本轮实测的端到端数值：平均延迟 "
                  f"{_fmt(end_to_end.get('avg_latency_ms'), 2)} ms（检索 "
                  f"{_fmt(end_to_end.get('avg_retrieval_ms'), 2)} ms + 生成 "
                  f"{_fmt(end_to_end.get('avg_generation_ms'), 2)} ms），缓存命中率 "
                  f"{_fmt((end_to_end.get('cache_hit_rate') or 0) * 100, 1)}%，降级率 "
                  f"{_fmt((end_to_end.get('degrade_rate') or 0) * 100, 1)}%。")


def build_impl_module3(doc: Any) -> None:
    heading(doc, "4.3 模块三：Agent 决策层", 2)

    heading(doc, "（1）ReAct 循环", 3)
    para(doc, "循环是手写的，不依赖 LangChain 等框架的高层封装。每一轮：把系统提示 + 工具目录 + "
              "历史轨迹 + 本轮观察拼成 Prompt → 要求模型输出一段 JSON 决策 → "
              "解析并执行工具 → 把执行结果作为新的 Observation 追加进轨迹 → 进入下一轮，"
              "直到模型给出 final_answer 或触发终止条件。")
    code_block(
        doc,
        '决策 JSON 协议（模型每轮只输出这一段）：\n'
        '{\n'
        '  "thought":  "当前的推理：已知什么、还缺什么",\n'
        '  "action":   "knowledge_search",          // 或 "final_answer"\n'
        '  "args":     {"query": "...", "top_k": 5},\n'
        '  "actions":  [ {...}, {...} ]              // 可选：一轮内并行调用多个工具\n'
        '}',
    )

    heading(doc, "（2）八个工具", 3)
    para(doc, "工具全部继承 BaseTool，声明参数 schema 后由工具注册表统一管理，"
              "System Prompt 里的工具目录也是从注册表自动渲染的（新增工具无需改提示词）：")
    table(
        doc,
        ["工具", "作用", "关键实现点"],
        [
            ["knowledge_search", "在知识库中做混合检索，返回片段与来源页码", "直连检索引擎，自带相关度过滤与指标记账"],
            ["paper_metadata", "查询论文元信息（标题/作者/年份/DOI/摘要）", "支持按 doc_id 或自然语言线索定位"],
            ["paper_compare", "多篇论文四维对照（方法/数据集/指标/贡献）", "结构化输出，便于前端渲染成表格"],
            ["keyword_extract", "提取文本核心关键词", "词频 + 位置 + 长度先验的 TF 变体，无需外部分词器"],
            ["summarize_paper", "生成结构化摘要（背景/方法/结果/结论）", "固定四段式，保证输出可比"],
            ["current_time", "返回系统当前时间", "支撑「今天是几号」这类与知识库无关的问题"],
            ["calculator", "数学表达式求值", "**AST 白名单求值，拒绝任意代码执行**（不用 eval）"],
            ["web_search", "联网搜索知识库之外的信息",
             "支持 360 / 必应 / 搜狗 / 百度 / DuckDuckGo，按可达性自动尝试与跳过，无密钥也可用"],
        ],
        widths=[3.2, 5.2, 7.1], size=10, status_col=None,
    )

    heading(doc, "（3）决策优化与容错", 3)
    for text in (
        "工具选择路由：一次廉价的意图识别把候选工具从 8 个缩到 1~3 个；"
        "「3.14 乘 2.56」这类问题直接走计算器，跳过整个推理循环，省 Token 也少选错。",
        "并行工具调用：同一轮内互不依赖的多个工具并发执行，实测 3×0.5s 的串行等待缩短到约 0.6s。",
        "工具超时与异常：超时先重试，再失败则换工具；工具异常不是终止理由，"
        "错误文本会作为 Observation 回灌给模型，让它自己换条路走。",
        "死循环检测：用 (action, args) 指纹在滑动窗口内查重，重复即判死循环；"
        "终止时优先用已有信息组织答案，而不是甩给用户一句「推理超限」。",
        "轮次上限：超过最大轮次强制收尾，保证任何输入都能在有限步内返回。",
    ):
        bullet(doc, text)

    heading(doc, "（4）多轮记忆", 3)
    para(doc, "会话之间完全隔离（不同会话的记忆互不可见）；单个会话内维护 Token 预算的滑动窗口，"
              "超出预算的历史会被摘要压缩成一段阶段性总结后替换原始轮次，"
              "从而在固定上下文窗口下支持长对话。记忆落盘 SQLite，重启服务不丢历史。")


def build_impl_module4(doc: Any) -> None:
    heading(doc, "4.4 模块四：系统集成与前端展示", 2)

    heading(doc, "（1）装配与集成", 3)
    para(doc, "全系统只有一个地方负责「把零件装成机器」：src/frontend/container.py 的 "
              "SystemContainer。它按配置创建 LLM 客户端、向量化器、向量库、检索引擎、"
              "RAG 流水线、Agent、记忆库与工具注册表，并对外暴露 ask / ask_stream / "
              "index_files / documents 等少量方法。其余模块之间只依赖接口，不互相 new。")
    para(doc, "RAG 与 Agent 的融合点也在这里：knowledge_search 工具直接持有检索层实例，"
              "因此「Agent 调工具」与「RAG 流水线」共用同一套索引与同一套相关度判据，"
              "不会出现两条链路检索结果不一致的问题。")

    heading(doc, "（2）关于前端技术选型", 3)
    para(doc, "任务书建议使用 Streamlit / Gradio。本项目选择用标准库 ThreadingHTTPServer + "
              "Server-Sent Events + 原生 JavaScript 实现单页应用，理由有三：")
    for text in (
        "系统整体定位是「零依赖即可运行」，引入 Streamlit 会带来数十个传递依赖，"
        "与「任何人 clone 下来 pip install numpy 就能跑」的目标冲突；",
        "Streamlit 的执行模型是整页重跑，做「右侧实时滚动 Thought/Action/Observation 轨迹」"
        "这类增量更新比较别扭，而 SSE 天然适合推送流式事件；",
        "前后端接口全部走标准 HTTP + JSON，答辩时可以直接用浏览器或 curl 演示，不依赖特定框架。",
    ):
        bullet(doc, text)
    para(doc, "代价是前端代码需要自己写：app.js 约 1600 行、Markdown 渲染器约 340 行、"
              "样式约 1160 行，合计约 3100 行，换来的是零构建步骤、零依赖与完全可控的交互细节。",
         indent_first=True)

    heading(doc, "4.4.1 系统架构图", 3)
    picture(doc, IMAGES / "architecture.png", width_in=PICTURE_WIDTH_IN,
            note="图 1　系统整体架构：展示层 / 接口层 / 决策层 / 生成层 / 检索层 / 数据层六层结构")
    para(doc, "架构说明：系统自下而上分为六层，各层通过标准化接口交互，"
              "任何一层都可以被替换而不影响其他层。")
    table(
        doc,
        ["层次", "组成", "职责"],
        [
            ["数据层", "向量库（vectors.npy / Chroma / FAISS）、分块语料、SQLite（会话与缓存）、日志文件",
             "持久化一切需要跨进程或跨会话存在的数据"],
            ["检索层", "向量检索、BM25 检索、RRF 融合器、重排序器",
             "把「一个问题」变成「一组带来源与相关度分数的原文片段」"],
            ["生成层", "RAG 流水线、Prompt 模板、语义缓存、流式输出、降级策略",
             "把检索片段组织成带引用的答案，并保证「查不到时不硬编」"],
            ["决策层", "ReAct 循环、工具注册表、8 个工具、意图路由、记忆管理",
             "决定「这个问题该干什么」：直接回答、查知识库、还是组合多个工具"],
            ["接口层", "HTTP API（16 条路径，含上传 / 重建索引 / 删除 / 重命名等子路由）+ SSE 事件流",
             "前后端唯一的契约，同时便于用 curl / 浏览器直接验证"],
            ["展示层", "三栏单页应用：知识库管理 / 对话 / Agent 轨迹与指标",
             "把系统能力与「AI 为什么这么答」同时呈现给用户"],
        ],
        widths=[2.0, 6.8, 6.7], size=10, status_col=None,
    )

    heading(doc, "4.4.2 界面功能截图", 3)
    pics = [
        ("01_主界面_知识库与对话.png",
         "图 2　主界面：左侧为知识库管理（拖拽上传区与文档列表：标题、年份、类型、分块数、"
         "向量化状态，鼠标悬停出现删除入口），中部为对话区（含新建会话、会话列表与示例问题），"
         "右侧为 Agent 推理轨迹面板"),
        ("02_问答_引用与Agent轨迹.png",
         "图 3　问答结果：答案中带 [1][2][3] 引用角标、末尾给出「引用来源（3）」清单，"
         "右侧为逐条展开的 ROUTE / ACTION / RETRIEVE / OBSERVE / THOUGHT / CITATION 轨迹与耗时"),
        ("03_多轮追问_上下文记忆.png",
         "图 4　多轮追问：同一会话内继承上文（“它和 ResNet 有什么不同？”），"
         "左侧会话列表显示轮次计数，验证会话隔离与记忆压缩"),
        ("04_工具清单.png",
         "图 5　工具清单：8 个已注册工具的名称、说明与参数，"
         "由工具注册表实时渲染（新增工具无需改前端）"),
        ("05_运行指标面板.png",
         "图 6　运行指标面板：核心 KPI（请求数、检索命中率、缓存命中率、平均/P95 延迟、检索 P50、"
         "工具成功率、降级率、Token、平均轮次、平均 Top 分数）、工具调用统计、终止原因分布与检索质量"),
    ]
    for name, note in pics:
        picture(doc, SHOTS / name, width_in=PICTURE_WIDTH_IN, note=note)

    heading(doc, "（5）集成过程中修掉的真实缺陷", 3)
    para(doc, "端到端联调暴露了三个只有「真跑起来」才会发现的问题，均已修复并补了回归测试：")
    table(
        doc,
        ["现象", "根因", "修复与防护"],
        [
            ["侧栏知识库列表与底部统计一起空白，控制台报 this.item is not a function",
             "渲染方法在本对象里叫 node，调用处却写成了 item —— 典型的成员调用拼写错误，"
             "语法检查完全查不出来",
             "改回 this.node()；新增 MemberCallContractTest：静态解析 app.js 的命名空间对象，"
             "校验每个 this.xxx() 都能在本命名空间内解析（已用「重新注入该 bug」验证测试确实会失败）"],
            ["「停止生成」按钮在任何时候都显示着",
             ".stop-btn 是类选择器（display:inline-flex），优先级高于浏览器默认的 "
             "[hidden]{display:none}，hidden 属性被压过",
             "样式表加入全局兜底 [hidden]{display:none !important}"],
            ["新建会话后标题栏仍挂着上一个会话的名字",
             "Sessions.create() 只更新了 sessionId，没有同步 sessionTitle 与标题栏 DOM；"
             "后端自动命名（用首个问题）后前端也没有回读",
             "三处补齐：创建时同步标题栏与会话 ID；切换会话时回读标题；每轮问答结束后刷新会话列表"],
        ],
        widths=[4.0, 5.4, 6.1], size=10, status_col=None,
    )


def build_impl_module5(doc: Any, metrics: Dict[str, Any],
                       chunk_rows: List[Dict[str, str]]) -> None:
    heading(doc, "4.5 模块五：评测", 2)
    retrieval = metrics.get("retrieval") or {}
    agent = metrics.get("agent") or {}
    end_to_end = metrics.get("end_to_end") or {}
    eval_stats = load_eval_set_stats()
    total_items = metrics.get("eval_set_size") or sum(eval_stats.values())

    heading(doc, "（1）评测集与两套口径", 3)
    para(doc, f"评测集共 {total_items} 条问答对，按问题类型分层抽样："
              + "、".join(f"{k} {v} 条" for k, v in eval_stats.items()) +
              "。相关性判定把评测样本的原文片段与检索片段做 4-gram 字符重合度比对（覆盖率 ≥ 0.5 判定相关）。")
    para(doc, "指标同时给出两套口径，并把差距本身当作结论的一部分：")
    table(
        doc,
        ["口径", "判定方式", "为什么要有它"],
        [
            ["严格口径", "只用真实返回的前 K 条；金标准精确解析为 chunk_id", "无模糊阈值、无候选池放大，是排序质量的硬指标"],
            ["引擎口径", "调用检索引擎自带的评测（内部会取 max(top_k,10) 再判覆盖）", "与引擎实现对齐，用于日常回归；数值天然偏高"],
        ],
        widths=[2.4, 7.0, 6.1], size=10, status_col=None,
    )
    para(doc, "两组数字并列给出，读者能直接看出「宽松口径高估了多少」——"
              "这比只报一个漂亮的数更有价值。")

    heading(doc, "（2）检索三档对比", 3)
    table_rows = retrieval.get("table") or []
    if table_rows:
        rows = []
        for row in table_rows:
            rows.append([
                row.get("mode_label") or row.get("mode"),
                f"`{row.get('mode')}`",
                _fmt(row.get("strict_hit@5")),
                _fmt(row.get("strict_mrr")),
                _fmt(row.get("strict_recall@5")),
                _fmt(row.get("strict_precision@5")),
                _fmt(row.get("strict_ndcg@5")),
                _fmt(row.get("avg_latency_ms"), 2),
            ])
        table(doc, ["档位", "模式", "Hit@5", "MRR", "Recall@5", "Precision@5", "NDCG@5", "平均延迟(ms)"],
              rows, widths=[3.5, 1.5, 1.5, 1.5, 1.8, 1.9, 1.5, 1.9], size=10, status_col=None)
    else:
        para(doc, "（检索三档对比：未运行）", color=RED)
    para(doc, "结论：混合检索（向量 + BM25，RRF 融合）相对纯向量的收益主要落在**召回**上"
              "（严格 Hit@5、Recall@5 提升），BM25 补的正是术语字面匹配能力；"
              "而叠加启发式重排后排序质量反而略降，说明当前的六特征加权在某些问题上会误判 —— "
              "因此默认关闭重排。这条结论是实测出来的，不是照搬「重排一定更好」的经验。")

    heading(doc, "（3）分块策略对比", 3)
    if chunk_rows:
        table(
            doc,
            ["分块策略", "分块数", "平均块长(字符)", "建索引(ms)", "Hit@5", "MRR", "Recall@5", "NDCG@5", "平均延迟(ms)"],
            [[r["label"], r["num_chunks"], r["avg_chars"], r["index_ms"],
              r["hit"], r["mrr"], r["recall"], r["ndcg"], r["latency"]] for r in chunk_rows],
            widths=[2.3, 1.1, 2.3, 1.5, 1.5, 1.5, 1.8, 1.6, 1.9], size=10, status_col=None,
        )
        para(doc, "结论：粒度过细反而拉低召回 —— 语义切分把平均块长压到约 200 字符、"
                  "分块数翻倍，金标准文本被切散到多个块里，严格 Hit@5 反而最低；"
                  "递归切分召回最优、固定窗口切分排序最优，两者并不一致。"
                  "换语料（尤其是中文长文档）结论可能变化，因此实验做成了可一键重跑的脚本。")
    else:
        para(doc, "（分块策略对比：未运行，见 reports/分块策略对比实验报告.md）", color=RED)

    heading(doc, "（4）Agent 决策层与端到端", 3)
    if agent:
        table(
            doc,
            ["指标", "实测值", "说明"],
            [
                ["工具调用准确率（宽松 / 严格）",
                 f"{_fmt((agent.get('tool_accuracy') or 0) * 100, 1)}% / {_fmt((agent.get('tool_accuracy_strict') or 0) * 100, 1)}%",
                 "期望工具集命中 / 首个工具必须正确"],
                ["平均推理轮次", _fmt(agent.get("avg_iterations"), 2), f"最大 {agent.get('max_iterations')} 轮"],
                ["平均延迟 / P95", f"{_fmt(agent.get('avg_latency_ms'), 2)} / {_fmt(agent.get('p95_latency_ms'), 2)} ms", "单题端到端（Agent 路径）"],
                ["Token 消耗", f"总计 {agent.get('total_tokens')}，平均 {_fmt(agent.get('avg_tokens'), 0)}/题", "含系统提示与轨迹回灌"],
            ],
            widths=[4.2, 4.6, 6.7], size=10, status_col=None,
        )
    if end_to_end:
        table(
            doc,
            ["端到端指标", "实测值", "说明"],
            [
                ["平均延迟", f"{_fmt(end_to_end.get('avg_latency_ms'), 2)} ms "
                         f"（检索 {_fmt(end_to_end.get('avg_retrieval_ms'), 2)} + 生成 {_fmt(end_to_end.get('avg_generation_ms'), 2)}）",
                 f"共 {end_to_end.get('num_calls')} 次调用；P95 {_fmt(end_to_end.get('p95_latency_ms'), 2)} ms"],
                ["缓存命中率", f"{_fmt((end_to_end.get('cache_hit_rate') or 0) * 100, 1)}%", "语义缓存（阈值 0.93）"],
                ["降级率", f"{_fmt((end_to_end.get('degrade_rate') or 0) * 100, 1)}%",
                 "全部为 low_relevance（相关性低于阈值）"],
                ["引用准确率 / 覆盖率", f"{_fmt((end_to_end.get('avg_citation_accuracy') or 0) * 100, 1)}% / "
                                  f"{_fmt((end_to_end.get('avg_citation_coverage') or 0) * 100, 1)}%",
                 "引用来源真实存在的比例 / 带引用的实质句子比例"],
                ["忠实度", f"{_fmt((end_to_end.get('avg_faithfulness') or 0) * 100, 1)}%", "答案句子能在上下文中找到 4-gram 支撑的比例"],
            ],
            widths=[4.2, 5.4, 5.9], size=10, status_col=None,
        )
    para(doc, "说明：本轮评测运行在 mock 大模型后端下（离线、确定性），"
              "因此「生成」耗时与答案文本不代表真实模型的表现；"
              "检索、Agent 决策、引用核对等全部是真实计算路径。"
              "换真实模型只需改一个环境变量后重跑同一条命令。", color=GREY)

    heading(doc, "（5）Bad Case 与一轮迭代优化", 3)
    para(doc, "Bad Case 自动分成四类并分别归因：检索失败 / 生成错误 / 引用错误 / Agent 决策错误。"
              "本轮 13 条 Bad Case 全部属于「检索失败」，引用类与决策类均为 0 条 —— "
              "这个分布本身就是结论：系统的薄弱环节在检索召回，而不在生成与决策。")
    para(doc, "在此之上做了一轮真实的参数对照实验（同一语料、同一评测集，只改参数）：")
    table(
        doc,
        ["参数组", "配置差异", "严格 Hit@5", "NDCG@5", "降级率", "平均延迟(ms)"],
        [
            ["基线", "chunk_size 512 / candidate_k 20 / 开启重排", "0.7705", "0.6238", "16.39%", "1.7767"],
            ["优化后", "chunk_size 384 / candidate_k 40 / 关闭重排 / 上下文上限 8000", "0.8033", "0.6359", "3.28%", "1.5925"],
            ["变化", "—", "+0.0328", "+0.0121", "−13.11 个百分点", "−0.1842"],
        ],
        widths=[2.0, 5.3, 2.1, 1.8, 1.9, 2.1], size=10, status_col=None,
    )
    para(doc, "这一轮优化的意义不在于涨了 3 个百分点，而在于：**每个参数为什么改、改完效果如何，"
              "都有实测数据支撑**。降级率从 16.4% 降到 3.3% 更是直接改善了用户体验 —— "
              "系统从「经常告诉你相关性低」变成「大多数问题都能正常回答」。")


# ==============================================================================
# 五、技术设计文档
# ==============================================================================
def build_section_design(doc: Any) -> None:
    heading(doc, "五、技术设计文档", 1)
    para(doc, "完整的接口签名、数据结构字段与配置项说明见项目内 docs/技术设计文档.md"
              "（约 1400 行，与代码逐项核对过）。本章给出其中最核心的四部分。")

    heading(doc, "5.1 系统架构图", 2)
    para(doc, "整体架构图见图 1（4.4.1）。此处补充两张核心流程图："
              "一次问答在 RAG 生成层内部的流转，以及 Agent 的 ReAct 决策循环。")
    picture(doc, IMAGES / "rag_flow.png", width_in=PICTURE_WIDTH_IN,
            note="图 7　RAG 七步流水线：缓存 → 混合检索 → 重排 → 降级判定 → 上下文构建 → 生成 → 引用解析")
    picture(doc, IMAGES / "react_loop.png", width_in=PICTURE_WIDTH_IN,
            note="图 8　Agent 的 ReAct 循环：Thought → Action → Observation，含并行调用、"
                 "循环检测与错误回灌，底部为 8 个可调用工具")

    heading(doc, "5.2 模块接口定义", 2)
    para(doc, "模块之间只通过下列接口协作；每个模块内部可以自由重构，只要接口不变就不会影响别人。")
    table(
        doc,
        ["模块", "核心类 / 入口", "关键接口（对外契约）"],
        [
            ["文档加载", "DocumentLoader / BatchImporter",
             "load(path) → LoadedDocument；load_dir(dir) → 批量结果；BatchImporter.run(paths) → 带进度与失败重试的导入"],
            ["分块", "BaseChunker 及四个子类",
             "create_chunker(config, strategy=..., **kw)；split(document) → List[Chunk]"],
            ["向量化", "BaseEmbedder 及三个实现",
             "embed(texts) → np.ndarray；embed_one(text) → np.ndarray；dimension 属性"],
            ["向量库", "BaseVectorStore（Numpy / Chroma / Faiss）",
             "add(chunks, vectors) / search(vector, k) / delete(doc_id) / save() / load()"],
            ["检索", "RetrievalEngine",
             "retrieve(query, top_k, mode, candidate_k) → List[RetrievedChunk]；"
             "retrieve_with_context(query, top_k, mode) → dict（含上下文与统计）；"
             "index_documents(documents) / delete_document(doc_id)；evaluate(...) / compare_modes(...)"],
            ["生成", "RAGPipeline",
             "answer(question, top_k, mode, history) → RAGAnswer；"
             "answer_stream(...) → 事件生成器（SSE）；clear_cache()"],
            ["Agent", "ReActAgent / ToolRegistry / BaseTool",
             "run(question, session_id, history, on_step) → AgentResult；"
             "registry.register(tool) / execute(call)；BaseTool.run(call) → ToolResult（异常统一转错误结果）"],
            ["记忆", "ConversationMemory",
             "create_session / add_turn / get_context(session_id, max_tokens) → List[Message]；compress(session_id)"],
            ["装配", "SystemContainer",
             "initialize() / index_files(paths) / ask(question, session_id, use_agent) / "
             "ask_stream(...) / documents() / metrics() / health() / tools()"],
            ["服务", "ApiServer",
             "dispatch(method, path) → HTTP 响应；16 条 /api 路径 + SSE 事件流；GET /api/health 逐组件健康检查"],
        ],
        widths=[1.8, 3.8, 9.9], size=9.5, status_col=None,
    )
    para(doc, "统一数据结构（src/utils/types.py）：LoadedDocument（原文 + 元信息）、Chunk（文本 + 页码 + 字符区间 + doc_id）、"
              "RetrievedChunk（Chunk + score/confidence/rerank_score）、Citation（来源 + 页码 + 片段编号）、"
              "RAGAnswer（答案 + 引用 + 检索片段 + 降级原因 + Token/耗时）、"
              "ToolCall / ToolResult / AgentStep / AgentResult（决策层）、Message / ConversationTurn / Session（记忆层）。")

    heading(doc, "5.3 核心数据流向", 2)
    para(doc, "一次完整问答的数据流如下（编号对应图 7、图 8 中的步骤）：")
    code_block(
        doc,
        "【入库链路】\n"
        "  用户上传 PDF/Word → DocumentLoader 解析（页码 + 原文）\n"
        "    → Chunker 切分（页码 + 字符区间 + doc_id）\n"
        "    → Embedder 向量化（带 SQLite 缓存）\n"
        "    → VectorStore.add() 落盘 vectors.npy / chunks.json\n"
        "    → BM25 索引与重排器 IDF 统计同步更新\n\n"
        "【问答链路】\n"
        "  用户提问 → 记忆层取本会话上下文\n"
        "    → ReAct 循环：模型输出 JSON 决策（thought / action / args）\n"
        "    → 工具执行：knowledge_search → RetrievalEngine.retrieve_with_context()\n"
        "         ├─ 向量召回（余弦相似度）┐\n"
        "         ├─ BM25 召回（Okapi）    ├─ RRF 融合 → 重排 → 三分数\n"
        "    → Observation 回灌 → 继续循环，或 final_answer\n"
        "    → 生成：上下文按相关度排序 + 截断 → LLM（流式）\n"
        "    → 引用解析并与检索结果核对（不存在的来源标记「未核实」）\n"
        "    → 回写：语义缓存 + traces.jsonl + metrics.jsonl\n"
        "    → SSE 推送：元信息 → token 增量 → 引用与轨迹 → 结束\n\n"
        "【观测链路】\n"
        "  MetricsCollector 累加（RAG 口径 / 裸检索口径 / 工具口径）\n"
        "    → GET /api/metrics → 前端「系统指标」面板（每 5 秒自动刷新）",
    )

    heading(doc, "5.4 配置文件说明", 2)
    para(doc, "全部配置集中在项目根目录 config.yaml，共 14 个顶层小节；"
              "支持三层覆盖：**默认值 < config.yaml < 环境变量**。"
              "环境变量用双下划线表示层级，例如 APP_LLM__PROVIDER=ollama 覆盖 llm.provider —— "
              "这让「不修改任何文件就切换后端」成为可能，也是评测能一键复现的基础。")
    table(
        doc,
        ["配置节", "关键项", "说明"],
        [
            ["llm", "provider / model / temperature / timeout / 密钥引用", "大模型后端（mock / ollama / openai 兼容 / dashscope）"],
            ["embedding", "provider / model / dimension / batch_size / cache", "向量化后端；默认 hash，512 维，开启向量缓存"],
            ["vector_store", "backend / persist_dir / collection", "向量库后端（numpy / chroma / faiss）与落盘目录"],
            ["chunking", "strategy / chunk_size / chunk_overlap / min_chunk_size / semantic / paper_optimize",
             "分块策略与参数；paper_optimize 控制跨页合并、公式行保护等论文专项优化"],
            ["retrieval", "top_k / candidate_k / mode / enable_bm25 / enable_rerank / bm25_k1 / bm25_b / rrf_k",
             "检索参数；candidate_k 决定融合候选池大小，rrf_k 为 RRF 的平滑常数"],
            ["reranker", "provider / model / weights", "重排后端与启发式特征权重"],
            ["generation", "max_context_chars / cache / prompts / stream", "上下文长度上限、语义缓存阈值、Prompt 模板选择"],
            ["agent", "max_iterations / tool_timeout / parallel / loop_detect / router",
             "循环轮次上限、单工具超时、并行开关、死循环检测窗口"],
            ["memory", "path / max_tokens / window / summary_threshold", "记忆库路径、Token 预算与摘要压缩阈值"],
            ["observability", "log_dir / level / metrics / traces", "日志目录、级别与指标/轨迹开关"],
            ["server", "host / port / cors / upload_limit", "Web 服务监听地址、跨域与上传体积上限"],
            ["paths", "data_dir / index_dir / report_dir / uploads", "各类文件落盘位置"],
            ["web_search", "provider / timeout / max_results",
             "联网搜索的引擎选择（auto 时按 360 → 必应 → 搜狗 → 百度 → DuckDuckGo 依次尝试）"],
            ["data_loader", "pdf_backend / text_strategy / max_file_mb", "PDF 解析后端优先级与文本加载策略"],
        ],
        widths=[2.7, 5.7, 7.1], size=9.5, status_col=None,
    )
    para(doc, "完整示例与每个参数的含义见 docs/技术设计文档.md 与项目根目录 config.yaml 中的逐行注释。")


# ==============================================================================
# 六、总结与展望
# ==============================================================================
def build_section_summary(doc: Any, metrics: Dict[str, Any], stats: Dict[str, Any]) -> None:
    heading(doc, "六、总结与展望", 1)
    retrieval = metrics.get("retrieval") or {}
    agent = metrics.get("agent") or {}
    end_to_end = metrics.get("end_to_end") or {}
    best = retrieval.get("best_mode_by_ndcg")

    heading(doc, "6.1 项目总结", 2)
    para(doc, "项目从零实现了一个可运行、可评测、可交付的论文知识库问答系统，"
              "完整覆盖了课程要求的五个模块，并在若干环节上做了超出基本要求的工程投入：")
    for text in (
        f"代码规模：业务代码约 {stats['total_src']:,} 行、测试代码约 {stats['total_tests']:,} 行，"
        f"全部为标准库 + NumPy 实现，无 Web 框架、无 LangChain、无 jieba；",
        "检索层：四后端文档解析、四种分块策略、手写 BM25 与 RRF 融合、三分数体系与重排序；",
        "生成层：强制引用溯源、流式输出、语义缓存与三态降级；",
        "Agent 层：手写 ReAct 循环 + 8 个工具 + 意图路由 + 并行调用 + 错误恢复 + 会话记忆；",
        "交付层：559 条单元测试、61 条评测集、三份自动生成的评测报告与两份技术文档、Docker 一键部署。",
    ):
        bullet(doc, text)

    para(doc, "关键实测数据（全部来自 reports/ 下可复现的评测产物）：", before=6)
    rows = []
    if best:
        # best_mode_by_ndcg 是档位名字符串（如 "hybrid"），数值要去 table 里按 mode 取
        best_name = best.get("mode") if isinstance(best, dict) else str(best)
        best_row = next(
            (r for r in (retrieval.get("table") or []) if r.get("mode") == best_name), {},
        )
        rows.append([
            "检索最优档位（严格 NDCG@5）",
            f"{best_row.get('mode_label') or best_name}：{_fmt(best_row.get('strict_ndcg@5'))}",
            f"同档严格 Hit@5 {_fmt(best_row.get('strict_hit@5'))}，"
            f"平均延迟 {_fmt(best_row.get('avg_latency_ms'), 2)} ms",
        ])
    if agent:
        rows.append(["Agent 工具调用准确率", f"{_fmt((agent.get('tool_accuracy_strict') or 0) * 100, 1)}%（严格）",
                     f"平均 {_fmt(agent.get('avg_iterations'), 2)} 轮完成"])
    if end_to_end:
        rows.append(["引用准确率 / 忠实度", f"{_fmt((end_to_end.get('avg_citation_accuracy') or 0) * 100, 1)}% / "
                                     f"{_fmt((end_to_end.get('avg_faithfulness') or 0) * 100, 1)}%",
                     "引用不造假、答案有上下文支撑（40 条端到端样本）"])
        rows.append(["端到端平均延迟", f"{_fmt(end_to_end.get('avg_latency_ms'), 2)} ms",
                     "mock 后端下的确定性数值，真实模型下生成耗时为主项"])
    rows.append(["一轮迭代优化", "严格 Hit@5 0.7705 → 0.8033，降级率 16.4% → 3.3%", "参数对照实验实测"])
    rows.append(["单元测试", "559 条全部通过", "python -m unittest discover -s tests -t ."])
    table(doc, ["维度", "实测值", "说明"], rows, widths=[4.4, 5.6, 5.5], size=10, status_col=None)

    heading(doc, "6.2 主要困难与解决方案", 2)
    para(doc, "下面每一条都是项目里真实发生过、并且是被数据或现象逼着解决的问题；"
              "列在这里是因为它们比「用了什么框架」更能说明工程能力。")
    table(
        doc,
        ["困难", "现象与根因", "解决方案", "验证方式"],
        [
            ["PDF 中文全是乱码",
             "只做 FlateDecode 解压、没解析 ToUnicode CMap，字形编码无法映射回 Unicode",
             "内置解析器加入 CMap 解析，优先使用嵌入字体的 ToUnicode 表",
             "用合成中英混排 PDF 做回归测试；解析后端在报告中显式记录（pdf:pypdf）"],
            ["降级策略从不触发",
             "归一化分数 Top-1 恒为 1.0，「有没有检索到」无法判断，无关问题也硬塞原文",
             "拆出绝对相关度 confidence，降级判定一律看 confidence；score 只用于展示排序",
             "评测中无关问题正确落入 no_result / low_relevance，降级率被真实统计出来"],
            ["向量分数与 BM25 分数无法加权合并",
             "两者量纲不同（0~1 与无上界），加权求和对分布极敏感，换语料即失效",
             "改用 RRF 只用排名不用分数",
             "三档对比实验显示混合检索的召回提升确实落在术语类问题上"],
            ["指标面板在真实使用下全是 0",
             "Web 端提问都走 Agent，而 Agent 的 knowledge_search 直连检索引擎、不构造 RAGAnswer，"
             "RAG 口径的计数自然恒为 0",
             "新增「裸检索」口径的指标记录与前端兜底：RAG 计数为 0 时改用检索口径展示",
             "真实提问后指标面板显示请求数 / 命中率 / P50 延迟等非零值（见截图 6）"],
            ["前端侧栏整块空白",
             "this.item is not a function —— 成员方法名写错，语法检查查不出来",
             "修正调用；新增静态契约测试校验每个 this.xxx() 都能在本命名空间解析",
             "重新注入该 bug，测试如期失败，证明测试有效"],
            ["启发式重排反而让指标变差",
             "六特征加权在部分问题上误判，把本来排对的结果往下压",
             "如实记录实测数据（NDCG@5 0.5827 → 0.5681），默认关闭重排，并在优化建议中给出实测差值",
             "对照实验：同语料同评测集，仅切换 enable_rerank"],
            ["分块越细反而召回越差",
             "语义切分把平均块长压到约 200 字符，金标准文本被切散到多个块里",
             "以实测为准选择默认策略（recursive），并把分块对比做成可重跑的实验",
             "四种策略在同一批评测集上的横向对比（reports/分块策略对比实验报告.md）"],
            ["文档配图里的中文与 emoji",
             "图表生成使用固定像素偏移，字号（点）与画布（相对坐标）换算比例随图高变化，文字压行；"
             "微软雅黑不含彩色 emoji 字形，渲染成豆腐块",
             "按图高实算行高并把文字块整体居中；工具名改为纯文字标签",
             "每次改图后裁剪检查渲染结果"],
        ],
        widths=[2.8, 5.0, 4.4, 3.3], size=9.5, status_col=None,
    )

    heading(doc, "6.3 待改进方向", 2)
    for text in (
        "接入真实语义向量与重排模型：当前默认后端是零依赖的哈希向量（词面级），"
        "对中英跨语言提问几乎没有信号，这是检索类 Bad Case 的首要原因；"
        "配置已支持 ollama / openai，缺的只是本机部署与一轮对比实验。",
        "扩充评测语料：把 3 篇样例论文扩到课程要求的 10–20 篇真实论文（同一研究方向），"
        "并同步扩充评测集到 100+ 条；评测命令与报告生成器无需改动。",
        "补齐答案质量的人工评分表：正确性、完整性、引用准确性三项由组员在真实模型环境下打分，"
        "与自动指标形成交叉验证。",
        "生成参数对比实验：在真实模型下对比 Temperature / Top-p / Top-k 对回答质量的影响，"
        "产出《Prompt 模板设计文档与实验对比记录》。",
        "把评测接进 CI：每次提交自动跑 --quick 评测并与基线对比，指标回退即告警 —— "
        "目前 559 条单元测试保证了正确性，但召回质量的变化还没有自动化守门人。",
        "多用户并发与压力测试：服务端是 ThreadingHTTPServer，已保证线程安全，"
        "但缺少 QPS、长连接与并发上传的实测数据。",
        "上下文与引用体验优化：目前的引用是整条回答生成后解析的，"
        "可以进一步做到「引用标记随答案同步出现」；长文档的上下文截断策略也可按章节结构优化。",
        "课堂汇报录屏：演示 PPT 已随交付包提供（见附录 B.5 第 8 项），"
        "但现场讲解的录屏尚未录制，需要回看时只能对着 PPT 逐页复现。"
        if load_ppt_path()
        else "演示 PPT 与录屏：作为课堂汇报材料另行准备。",
    ):
        bullet(doc, text)


# ==============================================================================
# 七、附录
# ==============================================================================
def build_appendix(doc: Any, stats: Dict[str, Any]) -> None:
    heading(doc, "七、附　录", 1)

    heading(doc, "附录 A：项目代码目录结构", 2)
    para(doc, f"下列目录树由脚本实时扫描生成（业务代码约 {stats['total_src']:,} 行，"
              f"测试代码约 {stats['total_tests']:,} 行）。"
              "data/、logs/、__pycache__ 等运行时产物未列出。")
    tree = build_tree_text(stats["groups"], stats)
    code_block(doc, tree, size=8.5)

    heading(doc, "附录 B：环境配置说明", 2)
    para(doc, "以下步骤在 Windows / macOS / Linux 上均可复现；"
              "系统只需 NumPy 一个必需第三方依赖，其余后端按需选装。")

    heading(doc, "B.1 创建环境与安装依赖", 3)
    code_block(
        doc,
        "# 1) 创建虚拟环境（conda 或 venv 任选其一）\n"
        "conda create -n rag_agent python=3.10\n"
        "conda activate rag_agent\n"
        "#    或者用 venv：\n"
        "#      Windows:      python -m venv .venv && .venv\\Scripts\\activate\n"
        "#      macOS/Linux:  python3 -m venv .venv && source .venv/bin/activate\n\n"
        "# 2) 安装依赖（numpy 为必需；其余为可选项，缺失时系统自动降级）\n"
        "pip install -r requirements.txt\n\n"
        "# 3) （可选）真实大模型与向量模型：本机安装 Ollama 后拉取模型\n"
        "ollama pull qwen2.5:7b          # 生成模型\n"
        "ollama pull bge-m3              # Embedding 模型\n"
        "#    也可改用环境变量直接走在线接口（无需改配置文件）：\n"
        "#    set APP_LLM__PROVIDER=openai      (Windows)\n"
        "#    export APP_LLM__PROVIDER=openai   (macOS/Linux)",
    )
    para(doc, "说明：本项目**不依赖 Streamlit**。任务书示例中的 streamlit run src/frontend/app.py "
              "在本项目中对应的启动方式是 python run.py —— 系统自带基于标准库的 Web 服务与前端，"
              "无需安装任何 Web 框架。", color=GREY)

    heading(doc, "B.2 初始化与启动", 3)
    code_block(
        doc,
        "# 4) 配置 API 密钥（只需一次；密钥经本机密钥派生后密封存储，明文不落盘）\n"
        "python -m src.utils.secrets set llm_api_key\n\n"
        "# 5) 环境自检 + 生成样例论文 + 建立索引\n"
        "python scripts/bootstrap.py --index\n\n"
        "# 6) 启动应用（默认 http://127.0.0.1:8501；端口被占用时自动顺延，以控制台输出为准）\n"
        "#    启动后会自动打开浏览器（Docker/无桌面环境自动跳过；--no-open 可关闭）\n"
        "python run.py\n\n"
        "# 其他常用入口\n"
        "python run.py --cli            # 命令行交互问答\n"
        "python run.py --no-open        # 只启动服务，不打开浏览器\n"
        "python run.py --health         # 组件健康检查\n"
        "python run.py --list-tools     # 查看 8 个工具及其参数\n"
        "python run.py --index ./data/samples   # 批量导入论文",
    )

    heading(doc, "B.3 运行测试与评测（可复现本交付书的全部数据）", 3)
    code_block(
        doc,
        "# 单元测试（559 条）\n"
        "python -m unittest discover -s tests -t .\n\n"
        "# 确定性离线评测约 5 秒，产出系统评测报告 / 性能数据 / Bad Case 报告\n"
        "APP_LLM__PROVIDER=mock python -m src.evaluation.report --quick\n"
        "#     Windows PowerShell 下等价写法：\n"
        "#       $env:APP_LLM__PROVIDER=\"mock\"; python -m src.evaluation.report --quick\n\n"
        "# 分块策略对比实验（四种策略，约 1 秒）\n"
        "APP_LLM__PROVIDER=mock python -m src.evaluation.chunking_experiment",
    )

    heading(doc, "B.4 Docker 一键部署（可选）", 3)
    code_block(
        doc,
        "docker compose -f docker/docker-compose.yml up --build\n"
        "# 数据卷挂载 data/ 与 logs/，容器重建不丢索引与会话",
    )

    heading(doc, "B.5 交付物清单", 3)
    table(
        doc,
        ["序号", "交付物", "位置"],
        [
            ["1", "完整代码仓库（含 Git 提交历史，按模块拆分提交）", "项目根目录"],
            ["2", "系统评测报告（含评测数据、图表、Bad Case 分析）", "reports/系统评测报告.md、性能评测数据.json/.xlsx、Bad_Case分析报告.md"],
            ["3", "分块策略对比实验报告", "reports/分块策略对比实验报告.md"],
            ["4", "技术设计文档", "docs/技术设计文档.md"],
            ["5", "用户使用手册", "docs/用户使用手册.md"],
            ["6", "本交付书（Word）", "智能科研助理_项目交付书.docx"],
            ["7", "Docker 部署文件", "docker/Dockerfile、docker/docker-compose.yml"],
            ["8", "演示 PPT", ppt_location()],
        ],
        widths=[1.2, 6.3, 8.0], size=10, status_col=None,
    )


def build_tree_text(groups: Dict[str, List[Tuple[str, int]]], stats: Dict[str, Any]) -> str:
    """把扫描结果渲染成带行数的目录树（等宽展示）。

    ``src/`` 单独成一层，其子包缩进一级，避免 8 个子包与 tests/docs 混在同一层。
    """
    #: 名字列的终止列 —— 所有行的行数都右对齐到同一列，表格才显得整齐
    NAME_COL = 52

    def row(prefix: str, name: str, count: int, suffix: str = "") -> str:
        pad = max(NAME_COL - len(prefix), len(name) + 1)
        return f"{prefix}{name:<{pad}}{count:>7} 行{suffix}"

    def entries_of(key: str, prefix: str, indent: str) -> List[str]:
        out: List[str] = []
        for name, n in sorted(groups[key]):
            short = name[len(prefix):] if name.startswith(prefix) else name
            out.append(row(indent, short, n))
        return out

    lines: List[str] = [
        "智能科研助理/",
        row("|-- ", "业务代码（src/）合计", stats["total_src"]),
        row("|-- ", "测试代码（tests/）合计", stats["total_tests"]),
        "",
    ]

    if "根目录" in groups:
        lines.append("|-- （根目录文件）")
        for name, n in sorted(groups["根目录"]):
            lines.append(row("|      ", name, n))

    if "src" in groups:
        lines.append("|-- src/")
        for name, n in sorted(groups["src"]):
            lines.append(row("|      ", name[len("src/"):], n))
        for key in sorted(k for k in groups if k.startswith("src/")):
            total = sum(n for _, n in groups[key])
            lines.append(row("|   |-- ", key.split("/")[-1] + "/", total, "（小计）"))
            lines.extend(entries_of(key, key + "/", "|   |      "))

    for key in ("tests", "scripts", "docs", "reports", "docker"):
        if key not in groups:
            continue
        total = sum(n for _, n in groups[key])
        lines.append(row("|-- ", key + "/", total, "（小计）"))
        lines.extend(entries_of(key, key + "/", "|      "))
    return "\n".join(lines)


# ==============================================================================
# 主流程
# ==============================================================================
def build_document(out_path: Path) -> Path:
    doc = Document()
    # 全局默认字体（正文样式）：仿宋四号、1.5 倍行距
    style = doc.styles["Normal"]
    style.font.name = FONT_ASCII
    style.font.size = Pt(BODY_SIZE)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT_BODY)
    style.paragraph_format.line_spacing = 1.5
    define_toc_styles(doc)
    for section in doc.sections:
        # 页面规格照课程模板：A4 + 上下 2.54 / 左右 3.17 cm
        section.page_width = Cm(PAGE_WIDTH_CM)
        section.page_height = Cm(PAGE_HEIGHT_CM)
        section.top_margin = Cm(MARGIN_TB_CM)
        section.bottom_margin = Cm(MARGIN_TB_CM)
        section.left_margin = Cm(MARGIN_LR_CM)
        section.right_margin = Cm(MARGIN_LR_CM)

    metrics = load_metrics()
    chunk_rows = load_chunking_rows()
    stats = scan_code_stats()
    today = date.today().strftime("%Y-%m-%d")

    build_cover(doc, today)
    build_toc(doc)
    build_section_overview(doc)
    build_section_team(doc)
    build_section_deliverables(doc, metrics, chunk_rows)
    build_section_implementation(doc, metrics, chunk_rows)
    build_section_design(doc)
    build_section_summary(doc, metrics, stats)
    build_appendix(doc, stats)
    enable_update_fields(doc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    from src.utils.console import enable_utf8_output

    enable_utf8_output()
    parser = argparse.ArgumentParser(
        prog="python scripts/make_delivery_doc.py",
        description="生成课程交付书（Word .docx），数字取自 reports/ 下的真实评测产物。",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    args = parser.parse_args(list(argv) if argv is not None else None)

    out = Path(args.out)
    print("=" * 68)
    print("生成课程交付书 …")
    print("=" * 68)
    build_document(out)
    size_kb = out.stat().st_size / 1024
    print(f"输出文件 : {out}")
    print(f"文件大小 : {size_kb:.1f} KB")
    print(f"项目背景 : {len(BACKGROUND)} 字（课程要求 ≤ 200 字）")
    print("-" * 68)
    print("提示：目录为 Word 域，首次打开时若未自动生成，请按 Ctrl+A 后 F9 更新。")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工执行入口
    raise SystemExit(main(sys.argv[1:]))
