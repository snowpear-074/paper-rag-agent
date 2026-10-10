# -*- coding: utf-8 -*-
"""模块一（文档加载部分）测试：PDF / DOCX / TXT / Markdown / CSV / BIB / JSON 加载器。

运行::

    python -m unittest tests.test_data_loader -v
"""

from __future__ import annotations

import unittest
import zipfile
from pathlib import Path

from tests.common import PROJECT_ROOT, SAMPLE_PDF_NAMES, TempWorkspace, ensure_sample_pdfs


# ===========================================================================
# PDF 加载器
# ===========================================================================
class PdfLoaderTest(unittest.TestCase):
    """内置 PDF 解析器（零依赖）与后端探测。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pdfs = ensure_sample_pdfs()

    def test_backend_report_lists_builtin_available(self):
        from src.data_loader import available_pdf_backends

        reports = available_pdf_backends()
        names = {r.name for r in reports}
        self.assertIn("builtin", names)
        self.assertIn("pypdf", names)
        builtin = next(r for r in reports if r.name == "builtin")
        self.assertTrue(builtin.available)

    def test_load_pdf_extracts_text(self):
        from src.data_loader import load_pdf

        doc = load_pdf(self.pdfs[0], backend="builtin")
        self.assertGreater(len(doc.text), 500)
        self.assertGreaterEqual(doc.meta.num_pages, 1)
        self.assertEqual(doc.meta.file_type, "pdf")
        self.assertTrue(doc.loader.startswith("pdf:"))

    def test_pdf_metadata_extraction(self):
        """三篇样例 PDF 的标题/作者/年份都应被正确抽取。"""
        from src.data_loader import load_pdf

        expected = {
            "sample_transformer.pdf": ("Attention Is All You Need", 2017, "Ashish Vaswani"),
            "sample_resnet.pdf": ("Deep Residual Learning for Image Recognition", 2016, "Kaiming He"),
            "sample_bert.pdf": ("BERT", 2019, "Jacob Devlin"),
        }
        for path in self.pdfs:
            with self.subTest(pdf=path.name):
                doc = load_pdf(path, backend="builtin")
                title, year, author = expected[path.name]
                self.assertIn(title, doc.meta.title)
                self.assertEqual(doc.meta.year, year)
                self.assertTrue(
                    any(author in name for name in doc.meta.authors),
                    f"未抽到作者 {author}，实际: {doc.meta.authors}",
                )

    def test_pdf_abstract_extraction(self):
        from src.data_loader import load_pdf

        doc = load_pdf(self.pdfs[0], backend="builtin")
        self.assertGreater(len(doc.meta.abstract), 80)
        self.assertIn("attention", doc.meta.abstract.lower())

    def test_pdf_pages_align_with_text(self):
        from src.data_loader import load_pdf

        doc = load_pdf(self.pdfs[1], backend="builtin")
        self.assertEqual(len(doc.pages), doc.meta.num_pages)
        joined = "".join(doc.pages)
        self.assertGreater(len(joined), 0)

    def test_unknown_backend_raises(self):
        from src.data_loader import load_pdf

        with self.assertRaises(ValueError):
            load_pdf(self.pdfs[0], backend="no_such_backend")

    def test_missing_file_raises(self):
        from src.data_loader import load_pdf

        with self.assertRaises(FileNotFoundError):
            load_pdf(Path("definitely_missing.pdf"))


# ===========================================================================
# Word 加载器
# ===========================================================================
class DocxLoaderTest(unittest.TestCase):
    """内置 OOXML 解析器与（可选）python-docx 对比。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 项目自带的课程任务书就是一份真实的中文 docx
        candidate = PROJECT_ROOT / "南京农业大学课程实践(1).docx"
        cls.docx_path = candidate if candidate.exists() else None

    def setUp(self) -> None:
        if self.docx_path is None:
            self.skipTest("缺少测试用 docx 文件")

    def test_load_docx_extracts_chinese_text(self):
        from src.data_loader import load_docx

        doc = load_docx(self.docx_path)
        self.assertGreater(len(doc.text), 2000)
        self.assertEqual(doc.meta.file_type, "docx")
        self.assertIn("RAG", doc.text)
        self.assertTrue(doc.meta.title)

    def test_docx_preserves_heading_structure(self):
        """标题应被转成 Markdown # 形式，这对分块边界识别很重要。"""
        from src.data_loader import load_docx

        doc = load_docx(self.docx_path)
        heading_lines = [ln for ln in doc.text.split("\n") if ln.startswith("#")]
        self.assertGreater(len(heading_lines), 3, "未识别出 Markdown 标题结构")

    def test_docx_tables_converted_to_markdown(self):
        from src.data_loader import load_docx

        doc = load_docx(self.docx_path)
        self.assertIn("|", doc.text, "表格未被转换为 Markdown")
        self.assertIn("---|", doc.text.replace(" ", ""))

    def test_docx_pages_nonempty(self):
        from src.data_loader import load_docx

        doc = load_docx(self.docx_path)
        self.assertGreater(len(doc.pages), 1)
        self.assertTrue(any(page.strip() for page in doc.pages))

    def test_invalid_docx_raises(self):
        from src.data_loader import load_docx

        with TempWorkspace() as ws:
            bogus = ws.path("fake.docx")
            bogus.write_bytes(b"this is definitely not a zip archive")
            with self.assertRaises(Exception):
                load_docx(bogus)

    def test_synthetic_docx_roundtrip(self):
        """自造一个最小 docx（含标题/段落/表格），验证解析结果可控。"""
        from src.data_loader import load_docx

        with TempWorkspace() as ws:
            path = ws.path("synthetic.docx")
            _write_minimal_docx(path)
            doc = load_docx(path)
            self.assertIn("Synthetic Title", doc.text)
            self.assertIn("Body paragraph", doc.text)
            self.assertIn("Cell A", doc.text)
            self.assertTrue(any(ln.startswith("#") for ln in doc.text.split("\n")))


def _write_minimal_docx(path: Path) -> None:
    """写一个结构合法的最小 .docx（用于验证解析器而不依赖外部文件）。"""
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Synthetic Title</w:t></w:r></w:p>
    <w:p><w:r><w:t>Body paragraph with content.</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Cell A</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Cell B</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>Cell C</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Cell D</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
    <w:p><w:r><w:t>Tail paragraph.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Synthetic Doc Title</dc:title>
  <dc:creator>Test Author</dc:creator>
</cp:coreProperties>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("docProps/core.xml", core)


# ===========================================================================
# 纯文本 / Markdown / CSV / BIB / JSON 加载器
# ===========================================================================
class TextLoaderTest(unittest.TestCase):
    """文本类加载器的编码探测、结构保留与分页。"""

    def test_load_markdown_by_heading(self):
        from src.data_loader import load_markdown

        with TempWorkspace() as ws:
            path = ws.path("doc.md")
            path.write_text(
                "# 第一章 引言\n\n内容一。\n\n# 第二章 方法\n\n内容二。\n\n# 第三章 实验\n\n内容三。\n",
                encoding="utf-8",
            )
            doc = load_markdown(path)
            self.assertEqual(doc.meta.file_type, "md")
            self.assertIn("第一章", doc.meta.title)
            self.assertEqual(len(doc.pages), 3, "应按一级标题切成 3 页")
            self.assertIn("内容二", doc.pages[1])

    def test_load_plain_text_gbk_encoding(self):
        """GBK 编码的中文文本必须能被正确解码（不能变成乱码）。"""
        from src.data_loader import load_text

        with TempWorkspace() as ws:
            path = ws.path("gbk.txt")
            path.write_bytes("中文编码测试内容，用于验证 GBK 解码。".encode("gbk"))
            doc = load_text(path)
            self.assertIn("中文编码测试", doc.text)
            self.assertIn("gb", doc.loader.lower())

    def test_load_plain_text_utf8_bom(self):
        from src.data_loader import load_text

        with TempWorkspace() as ws:
            path = ws.path("bom.txt")
            path.write_bytes("\ufeff带 BOM 的文本".encode("utf-8"))
            doc = load_text(path)
            self.assertIn("带 BOM 的文本", doc.text)
            self.assertNotIn("\ufeff", doc.text)

    def test_detect_encoding_variants(self):
        from src.data_loader import detect_encoding

        self.assertEqual(detect_encoding("hello world".encode("utf-8"))[0], "utf-8")
        self.assertIn(detect_encoding("中文".encode("gbk"))[0], ("gb18030", "gbk"))
        self.assertEqual(detect_encoding(b"\xef\xbb\xbfabc")[0], "utf-8-sig")
        self.assertEqual(detect_encoding(b"")[1], "")

    def test_load_csv_converted_to_markdown_table(self):
        from src.data_loader import load_csv

        with TempWorkspace() as ws:
            path = ws.path("data.csv")
            path.write_text("name,score\nTransformer,28.4\nResNet,3.57\n", encoding="utf-8")
            doc = load_csv(path)
            self.assertIn("| name | score |", doc.text)
            self.assertIn("Transformer", doc.text)
            self.assertEqual(doc.meta.extra["rows"], 3)

    def test_load_bibtex(self):
        from src.data_loader import load_bibtex

        with TempWorkspace() as ws:
            path = ws.path("refs.bib")
            path.write_text(
                "@article{vaswani2017,\n"
                "  title = {Attention Is All You Need},\n"
                "  author = {Vaswani, Ashish and Shazeer, Noam},\n"
                "  year = {2017},\n"
                "  journal = {NeurIPS}\n"
                "}\n"
                "@inproceedings{he2016,\n"
                "  title = {Deep Residual Learning},\n"
                "  author = {He, Kaiming},\n"
                "  year = {2016}\n"
                "}\n",
                encoding="utf-8",
            )
            doc = load_bibtex(path)
            self.assertEqual(doc.meta.extra["num_entries"], 2)
            self.assertIn("Attention Is All You Need", doc.text)
            self.assertIn("2017", doc.text)

    def test_load_json_text(self):
        from src.data_loader import load_json_text

        with TempWorkspace() as ws:
            path = ws.path("q.json")
            path.write_text('[{"q": "问题一"}, {"q": "问题二"}]', encoding="utf-8")
            doc = load_json_text(path)
            self.assertEqual(doc.meta.extra["records"], 2)
            self.assertIn("问题一", doc.text)

    def test_load_text_file_dispatch_by_suffix(self):
        from src.data_loader import load_text_file

        with TempWorkspace() as ws:
            md = ws.path("a.md")
            md.write_text("# 标题\n\n正文", encoding="utf-8")
            self.assertEqual(load_text_file(md).meta.file_type, "md")
            txt = ws.path("b.txt")
            txt.write_text("纯文本", encoding="utf-8")
            self.assertEqual(load_text_file(txt).meta.file_type, "txt")


# ===========================================================================
# 统一加载器与批量导入
# ===========================================================================
class DocumentLoaderTest(unittest.TestCase):
    """策略分派、能力报告、批量导入与重试。"""

    def test_capability_report(self):
        from src.data_loader import DocumentLoader

        report = DocumentLoader().capability_report()
        self.assertIn(".pdf", report["supported_extensions"])
        self.assertIn(".docx", report["supported_extensions"])
        self.assertTrue(any(b["name"] == "builtin" for b in report["pdf_backends"]))

    def test_unsupported_extension_raises(self):
        from src.data_loader import DocumentLoader, LoaderError

        with TempWorkspace() as ws:
            path = ws.path("x.unknown")
            path.write_text("data", encoding="utf-8")
            self.assertFalse(DocumentLoader().supported(path))
            with self.assertRaises(LoaderError):
                DocumentLoader().load(path)

    def test_empty_file_raises(self):
        from src.data_loader import DocumentLoader, LoaderError

        with TempWorkspace() as ws:
            path = ws.path("empty.txt")
            path.write_text("", encoding="utf-8")
            with self.assertRaises(LoaderError):
                DocumentLoader().load(path)

    def test_load_dispatch_produces_documents(self):
        from src.data_loader import DocumentLoader

        loader = DocumentLoader(pdf_backend="builtin")
        for path in ensure_sample_pdfs():
            doc = loader.load(path)
            self.assertTrue(doc.doc_id)
            self.assertTrue(doc.meta.title)
            self.assertGreater(len(doc.text), 0)

    def test_batch_importer_reports_progress_and_failures(self):
        from src.data_loader import BatchImporter, DocumentLoader

        loader = DocumentLoader(pdf_backend="builtin")
        importer = BatchImporter(loader, max_retries=0)
        with TempWorkspace() as ws:
            good = ws.path("ok.txt")
            good.write_text("这是一段可用的文本内容。" * 20, encoding="utf-8")
            bad = ws.path("bad.pdf")
            bad.write_bytes(b"%PDF-1.4 not a real pdf at all")

            seen = []
            job, documents = importer.import_documents(
                [good, bad], on_progress=lambda j, r: seen.append(r.ok)
            )
            self.assertEqual(job.total, 2)
            self.assertEqual(job.finished, 2)
            self.assertEqual(job.succeeded, 1)
            self.assertEqual(job.failed, 1)
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(seen), 2)
            self.assertTrue(any(not r.ok for r in job.results))
            self.assertAlmostEqual(job.percent, 100.0)

    def test_batch_importer_collects_directory(self):
        from src.data_loader import BatchImporter, DocumentLoader

        importer = BatchImporter(DocumentLoader(pdf_backend="builtin"))
        with TempWorkspace() as ws:
            ws.path("a.txt").write_text("aaa", encoding="utf-8")
            ws.path("sub", "b.md").write_text("# b", encoding="utf-8")
            ws.path("sub", "ignore.png").write_bytes(b"\x89PNG")
            files = importer.collect_files(ws.root)
            names = {p.name for p in files}
            self.assertEqual(names, {"a.txt", "b.md"})

    def test_batch_importer_empty_input(self):
        from src.data_loader import BatchImporter, DocumentLoader

        importer = BatchImporter(DocumentLoader())
        with TempWorkspace() as ws:
            job, documents = importer.import_documents([ws.dir("nothing")])
            self.assertEqual(job.total, 0)
            self.assertEqual(documents, [])
            self.assertTrue(job.errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
