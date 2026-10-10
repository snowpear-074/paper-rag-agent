#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成交付文档所需的架构图（确定性、可重复生成）。

图的**内容来源是代码与设计文档**，不是手绘：
::

    docs/技术设计文档.md  第二章「系统总体架构」的 mermaid 源
    src/ 各模块的实际类与职责

输出（默认 ``docs/images/``）::

    architecture.png   五层系统架构
    rag_flow.png       RAG 问答链路（含降级分支）
    react_loop.png     ReAct 循环与工具集

依赖：``matplotlib``（仅文档插图需要，**不是系统运行依赖**）。
字体优先使用系统中文字体，缺失时给出提示而不是画出方块。

用法::

    python scripts/make_diagrams.py
    python scripts/make_diagrams.py --out docs/images --dpi 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 配色（与前端主题一致：深蓝主色 + 浅灰底）
INK = "#1f2933"
MUTED = "#52606d"
LAYER_COLORS = {
    "show": ("#e8f1fb", "#2f6fb0"),
    "orchestrate": ("#eef7ee", "#3f8f4f"),
    "generate": ("#fdf3e3", "#b8761f"),
    "retrieve": ("#f3ecfb", "#6b4fa0"),
    "data": ("#eceff1", "#4a5a66"),
    "cross": ("#fdeaea", "#b04a4a"),
}


def setup_matplotlib(dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = dpi
    return plt


def check_cjk_font(plt) -> None:
    """中文字体缺失时提前告知（否则图上全是方块，白费功夫）。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    if not available.intersection(set(plt.rcParams["font.sans-serif"])):
        raise SystemExit(
            "未找到中文字体，图中中文会显示为方块。请安装「微软雅黑 / SimHei / "
            "Noto Sans CJK」后重试，或用 --font 指定已装字体名。"
        )


def new_canvas(plt, width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: Sequence[str] = (),
    colors: Tuple[str, str] = ("#f5f7fa", "#4a5a66"),
    title_size: float = 10.5,
    body_size: float = 8.2,
    radius: float = 1.6,
    title_top: bool = False,
) -> None:
    """画一个圆角卡片（标题 + 若干说明行）。

    文字块整体**垂直居中**：字号是点（point），画布是 0~100 的相对坐标，
    两者的换算比例取决于图高，所以早期的固定偏移量（``y + h - 1.9``）在
    6.6in 的图上会让标题压住第一行说明。这里按 ``figheight`` 实算行高。
    """
    from matplotlib.patches import FancyBboxPatch

    fill, edge = colors
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.35,rounding_size={radius}",
        linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(patch)

    # 1 点 = (100 / figheight_inches / 72) 个坐标单位
    unit_per_pt = 100.0 / max(ax.figure.get_figheight(), 1e-6) / 72.0
    title_h = title_size * 1.15 * unit_per_pt
    body_h = body_size * 1.30 * unit_per_pt
    head_gap = 4.0 * unit_per_pt          # 标题与正文之间留白
    lead = 1.2 * unit_per_pt              # 正文行距

    block = title_h + (head_gap + len(lines) * body_h + max(len(lines) - 1, 0) * lead
                       if lines else 0.0)
    # ``title_top``：容器型卡片（如"工具集"外框）标题贴顶，正文由调用方自己摆
    cursor = (y + h - 1.2) if title_top else (y + (h + block) / 2.0)

    ax.text(x + w / 2, cursor, title, ha="center", va="top",
            fontsize=title_size, color=INK, fontweight="bold", zorder=3)
    cursor -= title_h + head_gap
    for line in lines:
        ax.text(x + w / 2, cursor, line, ha="center", va="top",
                fontsize=body_size, color=MUTED, zorder=3)
        cursor -= body_h + lead


def band(ax, y: float, h: float, label: str, color: str) -> None:
    """左侧的层标签。"""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (1.6, y), 9.4, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=0, facecolor=color, zorder=1,
        )
    )
    ax.text(
        6.3, y + h / 2, label, ha="center", va="center",
        fontsize=10, color="white", fontweight="bold", rotation=90, zorder=3,
    )


def arrow(ax, x1: float, y1: float, x2: float, y2: float, text: str = "", color: str = MUTED,
          style: str = "-|>", dashed: bool = False, size: float = 8.0) -> None:
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
            linewidth=1.4, color=color, zorder=4,
            linestyle="--" if dashed else "-",
            shrinkA=1.0, shrinkB=1.0,
        )
    )
    if text:
        ax.text(
            (x1 + x2) / 2 + 1.2, (y1 + y2) / 2, text,
            ha="left", va="center", fontsize=size, color=color, zorder=5,
        )


def title(ax, text: str, subtitle: str = "") -> None:
    ax.text(50, 97.5, text, ha="center", va="top", fontsize=15, color=INK, fontweight="bold")
    if subtitle:
        ax.text(50, 94.0, subtitle, ha="center", va="top", fontsize=9, color=MUTED)


# ---------------------------------------------------------------------------
# 图 1：五层架构
# ---------------------------------------------------------------------------
def draw_architecture(plt, path: Path) -> Path:
    fig, ax = new_canvas(plt, 15.0, 10.6)
    title(ax, "智能科研助理 · 系统总体架构",
          "五层分层架构：展示 / 调度 / 生成 / 检索 / 数据（模块间只通过 src/utils/types.py 的数据契约耦合）")

    rows: List[Tuple[str, str, str, List[Tuple[str, List[str]]]]] = [
        ("① 展示层", "show", "LAYER L1", [
            ("Web 界面", ["三栏布局", "会话 · 对话 · 轨迹"]),
            ("REST API + SSE", ["/api/chat", "流式事件推送"]),
            ("CLI 交互", ["run.py --cli"]),
            ("指标面板", ["Token · 命中率 · 延迟"]),
        ]),
        ("② 调度层", "orchestrate", "LAYER L2", [
            ("ReAct Agent", ["手写 Thought→Action", "→Observation 循环"]),
            ("工具注册表", ["8 个工具", "参数校验 + 超时"]),
            ("意图路由", ["规则 / LLM 双实现", "候选工具裁剪"]),
            ("对话记忆", ["滑动窗口 + 摘要", "会话隔离"]),
        ]),
        ("③ 生成层", "generate", "LAYER L3", [
            ("RAG Pipeline", ["缓存→检索→降级", "→生成→引用"]),
            ("提示词模板", ["rag_qa / rag_strict", "/ rag_summary"]),
            ("语义缓存", ["向量近邻匹配", "阈值 0.93"]),
            ("LLM 接入层", ["openai / ollama", "/ dashscope / mock"]),
        ]),
        ("④ 检索层", "retrieve", "LAYER L4", [
            ("混合检索", ["向量 ‖ BM25", "RRF 融合"]),
            ("向量召回", ["numpy / chroma", "/ faiss"]),
            ("BM25 召回", ["k1=1.5 b=0.75"]),
            ("重排序", ["启发式 6 特征", "/ CrossEncoder"]),
        ]),
        ("⑤ 数据层", "data", "LAYER L5", [
            ("向量库持久化", ["JSON + numpy"]),
            ("索引清单", ["chunks.json", "documents.json"]),
            ("SQLite", ["记忆 · 缓存", "轨迹日志"]),
            ("上传目录", ["data/uploads"]),
        ]),
    ]

    top = 91.0
    band_h = 15.4
    gap = 2.3
    for index, (label, key, _sub, items) in enumerate(rows):
        y = top - band_h - index * (band_h + gap)
        fill, edge = LAYER_COLORS[key]
        band(ax, y, band_h, label, edge)
        left, right = 13.5, 77.5
        width = (right - left - 1.6 * (len(items) - 1)) / len(items)
        for col, (name, lines) in enumerate(items):
            box(ax, left + col * (width + 1.6), y, width, band_h, name, lines, (fill, edge))

    # 右侧横切关注点
    cross_x, cross_w = 79.5, 19.0
    box(ax, cross_x, 74.0, cross_w, 17.0, "配置中心",
        ["config.yaml 分层覆盖", "默认值 → 文件 → 环境变量", "→ 显式传参"], LAYER_COLORS["cross"])
    box(ax, cross_x, 53.0, cross_w, 17.0, "可观测性",
        ["logs/traces.jsonl", "logs/metrics.jsonl", "每次问答可回溯"], LAYER_COLORS["cross"])
    box(ax, cross_x, 32.0, cross_w, 17.0, "评测体系",
        ["评测集 61 题 · 检索/生成", "两套口径 · Bad Case", "自动生成报告"], LAYER_COLORS["cross"])
    box(ax, cross_x, 11.0, cross_w, 17.0, "降级策略",
        ["LLM 不可用 → 提示词兜底", "向量库缺失 → BM25", "绝不抛异常给用户"], LAYER_COLORS["cross"])

    # 层间主链路
    for index in range(len(rows) - 1):
        y_from = top - band_h - index * (band_h + gap)
        y_to = y_from - gap
        arrow(ax, 30, y_from, 30, y_to, color="#8a94a0")
    ax.text(31.5, top - band_h - 0.6, "请求自顶向下，结果自底向上", fontsize=8.2, color=MUTED)

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 2：RAG 问答链路
# ---------------------------------------------------------------------------
def draw_rag_flow(plt, path: Path) -> Path:
    fig, ax = new_canvas(plt, 15.0, 6.6)
    title(ax, "RAG 问答链路（含降级分支）",
          "src/generation/rag_pipeline.py：七步流水线，任何一步失败都走「有依据的降级」，而不是抛异常")

    generate = LAYER_COLORS["generate"]
    retrieve = LAYER_COLORS["retrieve"]
    data = LAYER_COLORS["data"]

    steps = [
        ("① 语义缓存查询", ["问题向量化", "命中 → 直接返回"], generate, 2.0),
        ("② 混合检索", ["向量 ‖ BM25 → RRF", "→ 重排序"], retrieve, 20.4),
        ("③ 降级判定", ["无结果 / 低相关", "confidence < 阈值"], retrieve, 38.8),
        ("④ 上下文构建", ["按相关度排序", "按预算截断"], retrieve, 57.2),
        ("⑤ LLM 生成", ["同步 / 流式", "失败 → 友好提示"], generate, 75.6),
    ]
    width = 16.6
    row_y, row_h = 70.0, 16.5
    for name, lines, colors, x in steps:
        box(ax, x, row_y, width, row_h, name, lines, colors, title_size=9.8)
    for index in range(len(steps) - 1):
        x_from = steps[index][3] + width
        arrow(ax, x_from, row_y + row_h / 2, steps[index + 1][3], row_y + row_h / 2)

    # 缓存命中的短路
    arrow(ax, 10.3, row_y, 10.3, 60.0, "命中", color="#b04a4a")
    box(ax, 2.0, 42.0, 34.0, 17.0, "缓存命中：直接返回",
        ["cache_hit=True，跳过检索与生成", "键 = 问题 + 模型 + 提示词模板",
         "+ 历史指纹 + top_k / max_tokens"],
        LAYER_COLORS["cross"], title_size=9.8)

    # 降级分支
    arrow(ax, 47.1, row_y, 47.1, 60.0, "no_result / low_relevance", color="#b8761f", size=7.6)
    box(ax, 40.0, 42.0, 32.0, 17.0, "降级分支（不是异常）",
        ["no_result → NO_CONTEXT 模板", "low_relevance → 提示并列出弱相关片段",
         "llm_error → 返回可读提示 + 重试建议"],
        LAYER_COLORS["generate"], title_size=9.8)

    # 生成后的两步
    box(ax, 76.0, 42.0, 22.0, 17.0, "⑥ 引用解析",
        ["只保留答案中真正", "引用到的来源", "【论文 p.3】→ 引用卡片"], generate, title_size=9.8)
    arrow(ax, 92.2, row_y, 92.2, 60.0)
    box(ax, 76.0, 17.0, 22.0, 17.0, "⑦ 写缓存 + Trace",
        ["降级答案不入缓存", "traces.jsonl / metrics.jsonl"], data, title_size=9.8)
    arrow(ax, 87.0, 42.0, 87.0, 34.0)
    arrow(ax, 87.0, 17.0, 87.0, 10.0)
    ax.text(87.0, 8.5, "RAGAnswer", ha="center", va="top", fontsize=10.0,
            color=INK, fontweight="bold")

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 3：ReAct 循环与工具集
# ---------------------------------------------------------------------------
def draw_react_loop(plt, path: Path) -> Path:
    fig, ax = new_canvas(plt, 15.0, 6.6)
    title(ax, "ReAct 推理循环与工具集",
          "src/agent/react_loop.py：手写 Thought → Action → Observation 循环，不依赖 LangChain")

    orchestrate = LAYER_COLORS["orchestrate"]
    show = LAYER_COLORS["show"]

    loop_y, loop_h = 75.0, 15.0
    mid_y = loop_y + loop_h / 2
    box(ax, 3.0, loop_y, 20.0, loop_h, "用户提问", ["+ 会话历史", "+ 候选工具"], show, title_size=10)
    box(ax, 27.0, loop_y, 20.0, loop_h, "Thought", ["模型输出决策 JSON", "thought / actions"], orchestrate, title_size=10)
    box(ax, 51.0, loop_y, 20.0, loop_h, "Action", ["单工具串行", "多工具并行执行"], orchestrate, title_size=10)
    box(ax, 75.0, loop_y, 22.0, loop_h, "Observation", ["工具结果回填", "Scratchpad（带预算）"], orchestrate, title_size=10)

    arrow(ax, 23.0, mid_y, 27.0, mid_y)
    arrow(ax, 47.0, mid_y, 51.0, mid_y)
    arrow(ax, 71.0, mid_y, 75.0, mid_y)
    arrow(ax, 86.0, 75.0, 86.0, 63.0, "未完成", color="#3f8f4f")
    arrow(ax, 37.0, 63.0, 37.0, 75.0, "回灌下一轮", color="#3f8f4f")
    ax.plot([86.0, 37.0], [63.0, 63.0], color="#3f8f4f", linewidth=1.4, zorder=4)

    box(ax, 3.0, 44.0, 30.0, 17.0, "循环控制",
        ["指纹去重 (action, args)", "语义近似（近期历史）",
         "一次警告 → 终止 loop_detected", "全局迭代上限 max_iterations"],
        LAYER_COLORS["cross"], title_size=10)
    box(ax, 35.0, 44.0, 28.0, 17.0, "错误恢复",
        ["工具超时（守护线程 + join）", "可重试：429 / 5xx / 超时",
         "不可重试：参数错 / 未知工具", "失败提示回灌模型自纠"],
        LAYER_COLORS["generate"], title_size=10)
    box(ax, 65.0, 44.0, 32.0, 17.0, "终止条件",
        ["全部动作都是 final_answer → 收敛", "达到迭代上限 → max_iterations",
         "检测到重复 → loop_detected", "模型输出不可解析 → 一轮格式纠正"],
        LAYER_COLORS["show"], title_size=10)

    # 工具名与 app.js 的 TOOL_ICON 一一对应（图标在图中省略：字体不含彩色 emoji）
    tools = [
        ("knowledge_search", "混合检索知识库"),
        ("paper_metadata", "论文元信息"),
        ("paper_compare", "多篇论文对比"),
        ("keyword_extract", "关键词抽取"),
        ("summarize_paper", "单篇摘要"),
        ("current_time", "当前时间"),
        ("calculator", "表达式计算（AST 白名单）"),
        ("web_search", "联网检索（可关闭）"),
    ]
    box(ax, 3.0, 3.0, 94.0, 33.0, "工具集（8 个，注册表按配置装配）", [],
        orchestrate, title_size=10, title_top=True)
    col_w = 92.0 / 4
    for index, (name, desc) in enumerate(tools):
        col, row = index % 4, index // 4
        x = 4.5 + col * col_w
        y = 27.0 - row * 15.0
        ax.text(x, y, name, ha="left", va="center", fontsize=8.8, color=INK, fontweight="bold")
        ax.text(x, y - 2.8, desc, ha="left", va="center", fontsize=7.8, color=MUTED)

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="生成交付文档用的架构图（PNG）")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "images"),
                        help="输出目录（默认 docs/images）")
    parser.add_argument("--dpi", type=int, default=160, help="分辨率（默认 160）")
    parser.add_argument("--font", default="", help="指定中文字体名（可选）")
    parser.add_argument("--only", default="", help="只生成某一张：architecture / rag_flow / react_loop")
    args = parser.parse_args(argv)

    plt = setup_matplotlib(args.dpi)
    if args.font:
        plt.rcParams["font.sans-serif"] = [args.font] + list(plt.rcParams["font.sans-serif"])
    check_cjk_font(plt)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders: Dict[str, Tuple[str, object]] = {
        "architecture": ("architecture.png", draw_architecture),
        "rag_flow": ("rag_flow.png", draw_rag_flow),
        "react_loop": ("react_loop.png", draw_react_loop),
    }
    selected = [args.only] if args.only else list(builders)
    for key in selected:
        if key not in builders:
            raise SystemExit(f"未知的图名：{key}（可选：{', '.join(builders)}）")
        filename, builder = builders[key]
        path = builder(plt, out_dir / filename)
        print(f"[diagram] {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
