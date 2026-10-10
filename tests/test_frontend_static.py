# -*- coding: utf-8 -*-
"""前端静态资源与契约一致性测试。

运行方式（项目根目录）::

    python -m unittest tests.test_frontend_static -v

设计说明
--------
本测试只依赖 Python 标准库 ``unittest`` / ``re`` / ``pathlib``，零第三方依赖，
不启动任何服务、不依赖网络，属于纯静态源码断言：

1. 静态文件存在且非空；
2. ``index.html`` 具备关键容器 id 并正确引用 CSS / JS；
3. ``app.js`` 覆盖全部 HTTP API 路径字面量与全部 SSE 事件名；
4. ``markdown.js`` 的「转义优先」安全检查（escapeHtml 必须先于标签插入）；
5. 用纯 Python 复现 markdown.js 的引用角标正则，独立验证 ``【来源 p.3】`` 的解析。
"""

import re
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径定位：允许从任意工作目录运行，向上回溯找到含 src/frontend/static 的根目录
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
STATIC_DIR = PROJECT_ROOT / "src" / "frontend" / "static"

INDEX_HTML = STATIC_DIR / "index.html"
STYLE_CSS = STATIC_DIR / "css" / "style.css"
APP_JS = STATIC_DIR / "js" / "app.js"
MARKDOWN_JS = STATIC_DIR / "js" / "markdown.js"


def read_text(path: Path) -> str:
    """以 UTF-8 读取文本；文件缺失时抛出可读性更好的断言错误。"""
    if not path.is_file():
        raise AssertionError(f"缺少必需文件：{path}")
    return path.read_text(encoding="utf-8")


# ===========================================================================
# 1. 静态文件存在性
# ===========================================================================
class StaticFilesExistTest(unittest.TestCase):
    """四个前端静态文件必须存在且非空。"""

    FILES = (
        ("index.html", INDEX_HTML),
        ("css/style.css", STYLE_CSS),
        ("js/app.js", APP_JS),
        ("js/markdown.js", MARKDOWN_JS),
    )

    def test_static_dir_exists(self):
        self.assertTrue(STATIC_DIR.is_dir(), f"静态目录不存在：{STATIC_DIR}")

    def test_all_files_exist_and_not_empty(self):
        for label, path in self.FILES:
            with self.subTest(file=label):
                self.assertTrue(path.is_file(), f"缺少文件：{label}（{path}）")
                content = path.read_text(encoding="utf-8")
                self.assertGreater(len(content.strip()), 0, f"{label} 内容为空")
                # 粗略但有效的规模下限，防止占位文件蒙混过关
                self.assertGreater(len(content), 400, f"{label} 体量过小，疑似占位文件")


# ===========================================================================
# 2. index.html 结构
# ===========================================================================
class IndexHtmlTest(unittest.TestCase):
    """index.html 的容器 id 与资源引用。"""

    @classmethod
    def setUpClass(cls):
        cls.html = read_text(INDEX_HTML)

    def test_required_container_ids(self):
        required = [
            "app", "sidebar", "chat-area", "trace-panel", "metrics-modal",
            "file-input", "message-input",
        ]
        for element_id in required:
            with self.subTest(element_id=element_id):
                pattern = r'id\s*=\s*["\']%s["\']' % re.escape(element_id)
                self.assertRegex(
                    self.html, pattern,
                    f"index.html 未找到 id=\"{element_id}\" 的容器",
                )

    def test_additional_containers(self):
        """界面要求中提到的其它关键容器（会话列表 / 文档列表 / 轨迹时间线等）。"""
        for element_id in ("session-list", "doc-list", "messages", "trace-timeline",
                           "upload-list", "toast-stack", "tools-modal"):
            with self.subTest(element_id=element_id):
                self.assertRegex(self.html, r'id\s*=\s*["\']%s["\']' % re.escape(element_id))

    def test_references_stylesheet(self):
        self.assertRegex(
            self.html, r'<link[^>]+href\s*=\s*["\']css/style\.css["\']',
            "index.html 未正确引用 css/style.css",
        )

    def test_references_scripts(self):
        self.assertRegex(
            self.html, r'<script[^>]+src\s*=\s*["\']js/markdown\.js["\']',
            "index.html 未正确引用 js/markdown.js",
        )
        self.assertRegex(
            self.html, r'<script[^>]+src\s*=\s*["\']js/app\.js["\']',
            "index.html 未正确引用 js/app.js",
        )

    def test_markdown_loaded_before_app(self):
        """markdown.js 必须先于 app.js 加载，保证 window.Markdown 可用。"""
        idx_md = self.html.find("js/markdown.js")
        idx_app = self.html.find("js/app.js")
        self.assertNotEqual(idx_md, -1)
        self.assertNotEqual(idx_app, -1)
        self.assertLess(idx_md, idx_app, "js/markdown.js 应在 js/app.js 之前加载")

    def test_no_external_cdn_dependency(self):
        """零 CDN 依赖：不得引用外部 http(s) 资源。"""
        external = re.findall(
            r'(?:src|href)\s*=\s*["\'](https?:)?//[^"\']+["\']', self.html, flags=re.IGNORECASE
        )
        self.assertEqual(external, [], f"index.html 出现外部资源引用：{external}")

    def test_file_input_accepts_required_formats(self):
        match = re.search(r'<input[^>]*id\s*=\s*["\']file-input["\'][^>]*>', self.html)
        self.assertIsNotNone(match, "未找到 file-input 输入框")
        tag = match.group(0)
        self.assertIn("multiple", tag, "file-input 应支持多选")
        for ext in (".pdf", ".docx", ".txt", ".md"):
            with self.subTest(ext=ext):
                self.assertIn(ext, tag, f"file-input 的 accept 缺少 {ext}")


# ===========================================================================
# 3. app.js 契约覆盖
# ===========================================================================
class AppJsContractTest(unittest.TestCase):
    """app.js 必须覆盖全部 API 路径与 SSE 事件名。"""

    @classmethod
    def setUpClass(cls):
        cls.js = read_text(APP_JS)

    def test_api_path_literals(self):
        paths = [
            "/api/health", "/api/chat", "/api/sessions", "/api/documents",
            "/api/metrics", "/api/tools", "/api/traces", "/api/config",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertIn(path, self.js, f"app.js 未使用 API 路径 {path}")

    def test_session_scoped_endpoints_built_by_template(self):
        """会话 / 文档的按 id 接口使用模板字符串拼接。"""
        self.assertRegex(self.js, r"/api/sessions/\$\{")
        self.assertRegex(self.js, r"/api/documents/\$\{")
        self.assertIn("/api/documents/upload", self.js)
        self.assertIn("/api/documents/reindex", self.js)

    def test_sse_event_names(self):
        events = [
            "start", "route", "retrieval", "thought", "action",
            "observation", "token", "citation", "final", "error", "done",
        ]
        for event in events:
            with self.subTest(event=event):
                # 必须作为字符串字面量出现（事件表或 case 分支）
                self.assertRegex(
                    self.js, r'["\']%s["\']' % re.escape(event),
                    f"app.js 未处理 SSE 事件 {event}",
                )

    def test_upload_sse_events(self):
        for event in ("progress", "indexed"):
            with self.subTest(event=event):
                self.assertRegex(self.js, r'["\']%s["\']' % re.escape(event))

    def test_manual_sse_parsing_not_eventsource(self):
        """必须用 fetch + ReadableStream 手工解析，不得使用 EventSource（需要 POST）。"""
        self.assertIn("ReadableStream", self.js + read_text(MARKDOWN_JS) + self.js)
        self.assertIn("getReader", self.js)
        self.assertNotIn("new EventSource", self.js)

    def test_abort_controller_present(self):
        self.assertIn("AbortController", self.js, "缺少 AbortController，无法中断 SSE 生成")

    def test_modular_namespaces(self):
        """代码需组织为清晰的模块化对象。"""
        for name in ("API", "SSE", "Store", "UI", "Chat", "Docs", "Trace", "Metrics", "Toast"):
            with self.subTest(module=name):
                self.assertRegex(
                    self.js, r"const\s+%s\s*=\s*\{" % re.escape(name),
                    f"app.js 缺少模块化对象 {name}",
                )

    def test_uses_markdown_renderer(self):
        self.assertIn("Markdown.render", self.js, "app.js 未调用 markdown.js 的渲染器")

    def test_base64_upload_via_filereader(self):
        self.assertIn("FileReader", self.js, "上传需使用 FileReader")
        self.assertIn("readAsDataURL", self.js, "上传需使用 readAsDataURL 转 base64")
        self.assertIn("content_base64", self.js, "上传请求体需包含 content_base64 字段")


# ===========================================================================
# 4. markdown.js 安全检查（转义优先）
# ===========================================================================
class MarkdownEscapeFirstTest(unittest.TestCase):
    """复现「先转义、后插标签」的安全约束。"""

    @classmethod
    def setUpClass(cls):
        cls.js = read_text(MARKDOWN_JS)

    def test_escape_html_exists(self):
        self.assertRegex(
            self.js, r"function\s+escapeHtml\s*\(",
            "markdown.js 未定义 escapeHtml 函数",
        )

    def test_escape_html_called_before_innerhtml(self):
        """源码顺序断言：escapeHtml( 首次出现必须早于 innerHTML 首次出现。"""
        first_escape_call = self.js.find("escapeHtml(")
        first_inner_html = self.js.find("innerHTML")
        self.assertNotEqual(first_escape_call, -1, "markdown.js 中未调用 escapeHtml(")
        if first_inner_html != -1:
            self.assertLess(
                first_escape_call, first_inner_html,
                "escapeHtml( 首次出现位置必须早于 innerHTML，先转义后插标签的约束被破坏",
            )

    def test_escape_called_inside_inline_pipeline(self):
        """parseInline 内部必须真正调用 escapeHtml（而不是只定义不调用）。"""
        match = re.search(r"function\s+parseInline\s*\([^)]*\)\s*\{", self.js)
        self.assertIsNotNone(match, "markdown.js 未定义 parseInline 函数")
        # 截取 parseInline 函数体（到下一个顶层函数定义为界，足够覆盖实现体）
        body = self.js[match.end():]
        end = body.find("\n  function ")
        if end != -1:
            body = body[:end]
        self.assertIn("escapeHtml(", body, "parseInline 渲染流程中未调用 escapeHtml")
        # 转义调用必须早于该函数体内任何 HTML 标签拼接
        self.assertLess(
            body.find("escapeHtml("), body.find("<"),
            "parseInline 中 escapeHtml 必须早于 HTML 标签插入",
        )

    def test_escapes_dangerous_characters(self):
        for char in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
            with self.subTest(entity=char):
                self.assertIn(char, self.js, f"escapeHtml 未处理实体 {char}")

    def test_required_markdown_features_present(self):
        """标题 / 粗体 / 斜体 / 行内代码 / 代码块 / 列表 / 引用 / 表格 / 水平线 / 链接。"""
        patterns = {
            "标题": r"#\{1,4\}",
            "粗体": r"\\\*\\\*",
            "行内代码": r"`\(\[\^`",
            "代码块围栏": r"`\{3,\}",
            "无序列表": r"\\s\*\)\(\[-\*\+\]\)",
            "引用": r"\\s\*>\\s\?",
            "水平线": r"-\{3,\}",
            "表格分隔": r"-\{2,\}",
        }
        for name, pattern in patterns.items():
            with self.subTest(feature=name):
                self.assertRegex(self.js, pattern, f"markdown.js 缺少 {name} 支持")
        for keyword in ('target="_blank"', 'rel="noopener', "code-block", "cb-copy", "table-wrap"):
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, self.js)

    def test_citation_ref_class_present(self):
        self.assertIn("citation-ref", self.js, "markdown.js 未输出 citation-ref 角标")
        self.assertIn("data-index", self.js, "角标缺少 data-index 属性")


# ===========================================================================
# 5. 引用角标正则的 Python 端独立验证
# ===========================================================================
# 下面的正则与 markdown.js 中的 CITATION_SOURCE_RE / CITATION_STANDALONE_RE
# 逐字符保持一致，用于在没有 JS 运行时的情况下独立验证解析规则。
CITATION_SOURCE_RE = re.compile(
    r"【\s*(?:\[?(\d{1,3})\]?\s*)?([^】\n]{1,120}?)\s+(?:[pP]\.|第)\s*(\d{1,4})\s*页?\s*】"
)
CITATION_STANDALONE_RE = re.compile(r"(^|[^\w\]])\[(\d{1,3})\](?![(（\w])")


class CitationRegexTest(unittest.TestCase):
    """引用角标规则：源文档名 + 页码 → index/source/page。"""

    def test_parse_source_citation_with_page(self):
        text = "【Attention Is All You Need p.3】"
        match = CITATION_SOURCE_RE.search(text)
        self.assertIsNotNone(match, "未能匹配 【Attention Is All You Need p.3】")
        index, source, page = match.group(1), match.group(2), match.group(3)
        self.assertEqual(source, "Attention Is All You Need")
        self.assertEqual(page, "3")
        self.assertIsNone(index, "无编号时 index 应为 None，由前端兜底分配")
        self.assertEqual(match.group(0), text, "应完整匹配整段引用标记")

    def test_parse_source_citation_with_chinese_page(self):
        match = CITATION_SOURCE_RE.search("结论见【论文A 第12页】")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(2), "论文A")
        self.assertEqual(match.group(3), "12")

    def test_parse_source_citation_with_explicit_index(self):
        match = CITATION_SOURCE_RE.search("【[2] Attention Is All You Need p.3】")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "2")
        self.assertEqual(match.group(2), "Attention Is All You Need")
        self.assertEqual(match.group(3), "3")

    def test_standalone_reference(self):
        text = "自注意力机制显著降低了计算复杂度，并提升了并行度 [1]"
        match = CITATION_STANDALONE_RE.search(text)
        self.assertIsNotNone(match, "未能匹配独立的 [1] 引用标记")
        self.assertEqual(match.group(2), "1", "index 应为 1")
        # 紧跟汉字的 [1] 视为普通方括号（与 markdown 链接引用的判定保持一致）
        self.assertIsNone(CITATION_STANDALONE_RE.search("该结论见复杂度[1]"))

    def test_standalone_reference_after_fullwidth_bracket(self):
        match = CITATION_STANDALONE_RE.search("该方法效果最佳（见【论文A p.5】）[3]")
        self.assertIsNotNone(match, "】后的 [3] 应被识别为引用角标")
        self.assertEqual(match.group(2), "3")

    def test_markdown_link_not_treated_as_citation(self):
        """[1](https://example.com) 是链接，不能被误判为引用角标。"""
        text = "参见 [1](https://example.com/paper.pdf) 的说明"
        match = CITATION_STANDALONE_RE.search(text)
        self.assertIsNone(match, "markdown 链接不应被识别为引用角标")

    def test_plain_brackets_without_citation_shape(self):
        """【...】内没有页码时不是引用标记，保持原样。"""
        self.assertIsNone(CITATION_SOURCE_RE.search("【这是一段普通说明文字】"))

    def test_multiple_citations_in_one_answer(self):
        text = (
            "Transformer 完全基于注意力机制【Attention Is All You Need p.3】[1]，"
            "并在机器翻译任务上取得最优结果【Neural Machine Translation p.7】[2]。"
        )
        sources = CITATION_SOURCE_RE.findall(text)
        self.assertEqual(len(sources), 2, "应解析出 2 个带来源的引用")
        self.assertEqual(sources[0][1], "Attention Is All You Need")
        self.assertEqual(sources[0][2], "3")
        self.assertEqual(sources[1][1], "Neural Machine Translation")
        self.assertEqual(sources[1][2], "7")
        standalones = [m.group(2) for m in CITATION_STANDALONE_RE.finditer(text)]
        self.assertEqual(standalones, ["1", "2"], "应解析出 [1] 与 [2] 两个独立角标")

    def test_escape_before_citation_rewrite(self):
        """纯 Python 近似复现渲染顺序：先转义、再替换引用标记。"""
        raw = "<script>alert(1)</script> 见【论文A p.5】"

        def escape_html(text: str) -> str:
            return (text.replace("&", "&amp;").replace("<", "&lt;")
                        .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;"))

        escaped = escape_html(raw)
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;script&gt;", escaped)

        rendered = CITATION_SOURCE_RE.sub(
            lambda m: '<sup class="citation-ref" data-index="{0}">[{0}]</sup>'.format(m.group(1) or "1"),
            escaped,
        )
        self.assertIn('<sup class="citation-ref" data-index="1">[1]</sup>', rendered)
        self.assertIn("&lt;script&gt;", rendered, "转义结果必须被保留")
        self.assertNotIn("<script>", rendered, "渲染后不得出现可执行标签")


# ===========================================================================
# 6. style.css 主题与响应式
# ===========================================================================
class StyleCssTest(unittest.TestCase):
    """CSS 变量主题 / 三栏网格 / 响应式断点。"""

    @classmethod
    def setUpClass(cls):
        cls.css = read_text(STYLE_CSS)

    def test_css_variables_defined(self):
        for var in ("--bg", "--panel", "--text", "--accent", "--border"):
            with self.subTest(var=var):
                self.assertIn(f"{var}:", self.css, f"未定义主题变量 {var}")

    def test_dark_theme_support(self):
        self.assertIn("prefers-color-scheme: dark", self.css, "缺少系统深色主题适配")
        self.assertIn('data-theme="dark"', self.css, "缺少手动深色主题")
        self.assertIn('data-theme="light"', self.css, "缺少手动浅色主题")

    def test_grid_layout_three_columns(self):
        self.assertRegex(
            self.css, r"grid-template-columns:\s*var\(--sidebar-w\)[^;]*var\(--trace-w\)",
            "缺少三栏网格布局（sidebar / 主区 / trace）",
        )
        for size in ("300px", "380px"):
            with self.subTest(size=size):
                self.assertIn(size, self.css)

    def test_responsive_breakpoints(self):
        self.assertIn("max-width: 1200px", self.css, "缺少 <1200px 折叠轨迹面板的断点")
        self.assertIn("max-width: 900px", self.css, "缺少 <900px 单列断点")

    def test_no_external_assets(self):
        external = re.findall(r"@import\s+url\(|url\(\s*['\"]?https?:", self.css)
        self.assertEqual(external, [], "CSS 中不得引用外部字体 / 资源")

    def test_citation_and_trace_styles(self):
        for selector in ("sup.citation-ref", ".trace-step", ".msg", ".toast", ".citation-card"):
            with self.subTest(selector=selector):
                self.assertIn(selector, self.css, f"缺少样式 {selector}")


class ToolIconContractTest(unittest.TestCase):
    """前端工具图标表的键必须与后端注册的工具名一致。

    回归：app.js 里写的是 ``paper_meta`` / ``compare_papers`` / ``summarize``，
    而后端 ``Tool.name`` 是 ``paper_metadata`` / ``paper_compare`` /
    ``summarize_paper`` —— 三个工具在轨迹里一直显示兜底图标 🧩，
    却没有任何测试发现（图标是"看得见的"、没人断言的部分）。
    """

    TOOLS_PY = PROJECT_ROOT / "src" / "agent" / "tools.py"

    @classmethod
    def setUpClass(cls):
        cls.js = read_text(APP_JS)
        cls.tools_src = read_text(cls.TOOLS_PY)

    def test_icon_table_covers_every_registered_tool(self):
        backend_names = set(re.findall(r'^\s{4}name\s*=\s*"([a-z_]+)"', self.tools_src, re.M))
        self.assertGreaterEqual(len(backend_names), 6, "未能从 tools.py 解析出工具名")

        block = re.search(r"TOOL_ICON:\s*\{(.*?)\}", self.js, re.S)
        self.assertIsNotNone(block, "app.js 中找不到 TOOL_ICON 表")
        icon_keys = set(re.findall(r"([a-z_]+)\s*:", block.group(1)))

        missing = sorted(backend_names - icon_keys)
        self.assertEqual(missing, [], f"这些工具的图标键与后端工具名不一致：{missing}")
        # 反向检查：图标表里不应残留已改名/已删除的键
        stale = sorted(key for key in icon_keys - backend_names if key.strip())
        self.assertEqual(stale, [], f"TOOL_ICON 中存在后端已不存在的工具键：{stale}")


class MemberCallContractTest(unittest.TestCase):
    """``this.X(...)`` 的调用必须在**同一个对象字面量**里能找到 ``X``。

    回归：``Docs.load()`` 里写的是 ``this.item(d)``，而 ``Docs`` 的渲染方法叫
    ``node``（``item`` 是 ``Sessions`` 的方法）。运行时报
    "this.item is not a function"，整个侧栏知识库列表与底部统计一起空白，
    但纯 Python 的测试永远发现不了 —— 这是"只能在浏览器里看见"的缺陷。

    这里用大括号配对切出每个顶层对象字面量，做一次朴素的作用域检查：
    假阳性（跨对象的合法用法）不适用，因为本文件所有 ``this.`` 都指向
    自己所在的命名空间对象。
    """

    def setUp(self):
        self.lines = read_text(APP_JS).splitlines()

    def _block_end(self, start: int) -> int:
        depth = 0
        for index in range(start, len(self.lines)):
            for char in self.lines[index]:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return index
        return len(self.lines) - 1

    def _namespaces(self):
        found = []
        for index, line in enumerate(self.lines):
            match = re.match(r"^\s*(?:const\s+)?([A-Za-z_$][\w$]*)\s*=\s*\{\s*$", line)
            if match:
                found.append((match.group(1), index))
        return found

    def test_every_this_call_resolves_within_its_namespace(self):
        namespaces = self._namespaces()
        self.assertGreaterEqual(len(namespaces), 8, "未能从 app.js 解析出命名空间对象")

        problems = []
        for name, start in namespaces:
            body = "\n".join(self.lines[start + 1:self._block_end(start)])
            defined = set(re.findall(
                r"^\s{2,8}(?:async\s+)?([A-Za-z_$][\w$]*)\s*(?:\(|:)", body, re.M))
            called = set(re.findall(r"\bthis\.([A-Za-z_$][\w$]*)\s*\(", body))
            problems.extend(f"{name}.this.{member}() 未定义" for member in sorted(called - defined))

        self.assertEqual(problems, [], "存在调用不到的同名成员：" + "；".join(problems))


if __name__ == "__main__":
    unittest.main(verbosity=2)
