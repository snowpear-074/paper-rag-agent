"""生成最小可用 PDF 的极简生成器（测试与自检用）。

用途
----
在没有真实论文 PDF 的环境下（CI / 首次运行 / 单元测试），
快速合成结构合法、可被任何 PDF 阅读器打开的 PDF 文件，
用于验证加载器、分块器、检索链路与端到端流程。

支持：多页、标题/正文字号、Helvetica 标准字体、ASCII 文本。
中文字符会以 WinAnsi 无法表示的字符被跳过（PDF 标准字体限制），
因此本生成器主要用于**英文论文样例**；中文样例请用 Markdown/TXT。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

__all__ = ["SimplePdfWriter", "build_sample_paper_pdf", "write_sample_pdfs"]

_PAGE_W, _PAGE_H = 612, 792  # US Letter


def _escape(text: str) -> str:
    """转义 PDF 字符串字面量，并剔除标准字体无法编码的字符。"""
    cleaned = "".join(ch for ch in text if 32 <= ord(ch) < 127)
    return cleaned.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class SimplePdfWriter:
    """按行写入文本的极简 PDF 生成器。"""

    def __init__(self, title: str = "") -> None:
        self.pages: List[List[Tuple[float, float, float, str]]] = []
        self._lines: List[Tuple[float, float, float, str]] = []
        self.cursor_y = _PAGE_H - 72.0
        self.title = title

    # -- 排版 -------------------------------------------------------------
    def add_line(self, text: str, size: float = 11.0, leading: float = 15.0, indent: float = 72.0) -> "SimplePdfWriter":
        """追加一行文本，自动分页。"""
        if self.cursor_y < 72.0:
            self.new_page()
        self._lines.append((indent, self.cursor_y, size, text))
        self.cursor_y -= leading
        return self

    def add_paragraph(self, text: str, size: float = 11.0, leading: float = 15.0) -> "SimplePdfWriter":
        """按 ~90 字符宽度折行后写入。"""
        for line in _wrap(text, width=int(560 / (size * 0.5))):
            self.add_line(line, size=size, leading=leading)
        self.add_line("", size=size, leading=leading * 0.5)
        return self

    def add_heading(self, text: str, size: float = 15.0) -> "SimplePdfWriter":
        self.cursor_y -= 8
        self.add_line(text, size=size, leading=size + 8)
        return self

    def new_page(self) -> "SimplePdfWriter":
        if self._lines:
            self.pages.append(self._lines)
        self._lines = []
        self.cursor_y = _PAGE_H - 72.0
        return self

    # -- 输出 -------------------------------------------------------------
    def to_bytes(self) -> bytes:
        if self._lines or not self.pages:
            self.pages.append(self._lines)
            self._lines = []

        objects: List[bytes] = []
        # 1: Catalog, 2: Pages, 3: Font, 之后每页 2 个对象（Page, Contents）
        page_obj_nums = [4 + i * 2 for i in range(len(self.pages))]

        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{num} 0 R" for num in page_obj_nums)
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>".encode())
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

        for index, lines in enumerate(self.pages):
            page_num = 4 + index * 2
            content_num = page_num + 1
            page = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_W} {_PAGE_H}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>"
            ).encode()
            objects.append(page)

            stream = ["BT"]
            for x, y, size, text in lines:
                if not text.strip():
                    continue
                stream.append(f"/F1 {size:.1f} Tf")
                stream.append(f"1 0 0 1 {x:.1f} {y:.1f} Tm")
                stream.append(f"({_escape(text)}) Tj")
            stream.append("ET")
            body = "\n".join(stream).encode("latin-1")
            objects.append(b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream")

        # 组装文件（含 xref 表，保证结构合法）
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for num, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
        xref_pos = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for offset in offsets[1:]:
            out += f"{offset:010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()
        return bytes(out)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_bytes())
        return target


def _wrap(text: str, width: int = 90) -> List[str]:
    """按词折行。"""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


# ==============================================================================
# 样例论文生成
# ==============================================================================
SAMPLE_PAPERS: List[dict] = [
    {
        "filename": "sample_transformer.pdf",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": 2017,
        "venue": "NeurIPS 2017",
        "abstract": (
            "The dominant sequence transduction models are based on complex recurrent or "
            "convolutional neural networks that include an encoder and a decoder. We propose "
            "a new simple network architecture, the Transformer, based solely on attention "
            "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on "
            "two machine translation tasks show these models to be superior in quality while "
            "being more parallelizable and requiring significantly less time to train."
        ),
        "keywords": "attention mechanism, transformer, machine translation, self-attention",
        "sections": [
            ("1 Introduction", [
                "Recurrent neural networks, long short-term memory and gated recurrent neural "
                "networks in particular, have been firmly established as state of the art "
                "approaches in sequence modeling and transduction problems such as language "
                "modeling and machine translation.",
                "Recurrent models typically factor computation along the symbol positions of the "
                "input and output sequences. This inherently sequential nature precludes "
                "parallelization within training examples, which becomes critical at longer "
                "sequence lengths, as memory constraints limit batching across examples.",
                "Attention mechanisms have become an integral part of compelling sequence "
                "modeling and transduction models in various tasks, allowing modeling of "
                "dependencies without regard to their distance in the input or output sequences.",
            ]),
            ("2 Model Architecture", [
                "Most competitive neural sequence transduction models have an encoder-decoder "
                "structure. The encoder maps an input sequence of symbol representations to a "
                "sequence of continuous representations. The decoder then generates an output "
                "sequence of symbols one element at a time.",
                "The Transformer follows this overall architecture using stacked self-attention "
                "and point-wise, fully connected layers for both the encoder and decoder. The "
                "encoder is composed of a stack of six identical layers.",
                "The encoder contains self-attention layers. In a self-attention layer all of the "
                "keys, values and queries come from the same place, in this case, the output of "
                "the previous layer in the encoder.",
            ]),
            ("3 Scaled Dot-Product Attention", [
                "We call our particular attention Scaled Dot-Product Attention. The input "
                "consists of queries and keys of dimension dk, and values of dimension dv. We "
                "compute the dot products of the query with all keys, divide each by the square "
                "root of dk, and apply a softmax function to obtain the weights on the values.",
                "Multi-head attention allows the model to jointly attend to information from "
                "different representation subspaces at different positions. With a single "
                "attention head, averaging inhibits this.",
                "The total computational complexity per layer is O(n^2 d) for self-attention "
                "compared to O(n d^2) for recurrent layers, where n is the sequence length and d "
                "is the representation dimension.",
            ]),
            ("4 Training and Results", [
                "We trained on the WMT 2014 English-German dataset consisting of about 4.5 "
                "million sentence pairs, and on the WMT 2014 English-French dataset consisting "
                "of 36M sentences. Sentences were encoded using byte-pair encoding.",
                "On the WMT 2014 English-to-German translation task, the big transformer model "
                "achieves 28.4 BLEU, outperforming the best previously reported results by more "
                "than 2.0 BLEU, establishing a new state-of-the-art BLEU score.",
                "On the WMT 2014 English-to-French translation task, the big model achieves a "
                "BLEU score of 41.8, outperforming all of the previously published single models "
                "at less than one fourth of the training cost.",
            ]),
            ("5 Conclusion", [
                "In this work, we presented the Transformer, the first sequence transduction "
                "model based entirely on attention, replacing the recurrent layers most commonly "
                "used in encoder-decoder architectures with multi-headed self-attention.",
                "We are excited about the future of attention-based models and plan to apply them "
                "to other tasks. We plan to investigate local, restricted attention mechanisms "
                "to efficiently handle very large inputs and outputs.",
            ]),
        ],
    },
    {
        "filename": "sample_resnet.pdf",
        "title": "Deep Residual Learning for Image Recognition",
        "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        "year": 2016,
        "venue": "CVPR 2016",
        "abstract": (
            "Deeper neural networks are more difficult to train. We present a residual learning "
            "framework to ease the training of networks that are substantially deeper than those "
            "used previously. We explicitly reformulate the layers as learning residual functions "
            "with reference to the layer inputs, instead of learning unreferenced functions. On "
            "the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers, "
            "achieving 3.57 percent top-5 error and winning the ILSVRC 2015 classification task."
        ),
        "keywords": "residual learning, image recognition, deep networks, ImageNet, degradation",
        "sections": [
            ("1 Introduction", [
                "Deep convolutional neural networks have led to a series of breakthroughs for "
                "image classification. Deep networks naturally integrate low, mid and high level "
                "features and classifiers in an end-to-end multi-layer fashion.",
                "When deeper networks are able to start converging, a degradation problem has been "
                "exposed: with the network depth increasing, accuracy gets saturated and then "
                "degrades rapidly. Unexpectedly, such degradation is not caused by overfitting.",
                "In this paper, we address the degradation problem by introducing a deep residual "
                "learning framework. Instead of hoping each few stacked layers directly fit a "
                "desired underlying mapping, we explicitly let these layers fit a residual mapping.",
            ]),
            ("2 Residual Learning Formulation", [
                "We adopt residual learning to every few stacked layers. Formally, denoting the "
                "desired underlying mapping as H(x), we let the stacked nonlinear layers fit "
                "another mapping of F(x) = H(x) - x. The original mapping is recast into F(x) + x.",
                "The formulation of F(x) + x can be realized by feedforward neural networks with "
                "shortcut connections. Shortcut connections are those skipping one or more layers.",
                "The degradation problem suggests that the solvers might have difficulties in "
                "approximating identity mappings by multiple nonlinear layers. With the residual "
                "learning reformulation, if an identity mapping is optimal, the solvers may simply "
                "drive the weights of the nonlinear layers toward zero.",
            ]),
            ("3 Architecture and Experiments", [
                "We test our method on the ImageNet 2012 classification dataset that consists of "
                "1000 classes. The models are trained on the 1.28 million training images and "
                "evaluated on the 50k validation images.",
                "We compare plain networks and residual networks with 18 and 34 layers. The plain "
                "34-layer network has higher validation error than the plain 18-layer network. "
                "The residual networks instead show increasing accuracy as depth increases.",
                "Our 152-layer ResNet has 11.3 billion FLOPs, still lower than VGG-16 and VGG-19 "
                "which have 15.3 and 19.6 billion FLOPs respectively. The ensemble ResNet "
                "achieves 3.57 percent top-5 error on the ImageNet test set.",
            ]),
            ("4 Conclusion", [
                "We presented a residual learning framework that facilitates the training of "
                "substantially deeper networks. Our results demonstrate that residual learning "
                "is generic and can be applied to other vision and non-vision problems.",
            ]),
        ],
    },
    {
        "filename": "sample_bert.pdf",
        "title": "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "year": 2019,
        "venue": "NAACL 2019",
        "abstract": (
            "We introduce a new language representation model called BERT, which stands for "
            "Bidirectional Encoder Representations from Transformers. BERT is designed to "
            "pre-train deep bidirectional representations from unlabeled text by jointly "
            "conditioning on both left and right context in all layers. BERT obtains new "
            "state-of-the-art results on eleven natural language processing tasks, including "
            "pushing the GLUE score to 80.5 percent and SQuAD v1.1 question answering Test F1 "
            "to 93.2."
        ),
        "keywords": "pre-training, bidirectional encoder, language representation, fine-tuning, GLUE",
        "sections": [
            ("1 Introduction", [
                "Language model pre-training has been shown to be effective for improving many "
                "natural language processing tasks. These include sentence-level tasks such as "
                "natural language inference and paraphrasing, and token-level tasks such as named "
                "entity recognition and question answering.",
                "There are two existing strategies for applying pre-trained language "
                "representations to downstream tasks: feature-based and fine-tuning. The "
                "feature-based approach uses task-specific architectures that include the "
                "pre-trained representations as additional features.",
                "The major limitation of standard language models is that they are "
                "unidirectional, which limits the choice of architectures that can be used "
                "during pre-training. BERT instead uses a masked language model.",
            ]),
            ("2 Pre-training Tasks", [
                "To train a deep bidirectional representation, we simply mask some percentage of "
                "the input tokens at random, and then predict those masked tokens. We refer to "
                "this procedure as a masked language model, or MLM.",
                "The second pre-training task is next sentence prediction. Many important "
                "downstream tasks such as question answering and natural language inference are "
                "based on understanding the relationship between two sentences, which is not "
                "directly captured by a language model.",
                "BERT uses WordPiece embeddings with a 30000 token vocabulary. The first token of "
                "every sequence is always a special classification token, and sentence pairs are "
                "packed into a single sequence with a separator token.",
            ]),
            ("3 Model Configuration and Results", [
                "We denote the number of layers as L, the hidden size as H, and the number of "
                "self-attention heads as A. We primarily report results on two model sizes: "
                "BERT BASE with L=12, H=768, A=12, and BERT LARGE with L=24, H=1024, A=16.",
                "BERT advances the state of the art for eleven NLP tasks. On the GLUE benchmark "
                "the BERT LARGE model obtains an average accuracy of 80.5 percent, a 7.6 "
                "percentage point absolute improvement over the previous state of the art.",
                "On SQuAD v1.1 question answering, BERT achieves 93.2 F1 on the test set, which "
                "is a 1.5 absolute improvement over the previous best ensemble system.",
            ]),
            ("4 Ablation Studies", [
                "We perform ablation experiments over a number of facets of BERT in order to "
                "better understand their relative importance. The model without the next "
                "sentence prediction task hurts performance on the natural language inference "
                "tasks significantly.",
                "Training with a masked language model is crucial: a left-to-right language model "
                "performs worse than the masked language model on every single task, and the "
                "difference is significant on question answering and named entity recognition.",
            ]),
        ],
    },
]


def build_sample_paper_pdf(spec: dict) -> SimplePdfWriter:
    """把样例论文规格渲染成 PDF writer。"""
    writer = SimplePdfWriter(title=spec["title"])
    writer.add_heading(spec["title"], size=17)
    writer.add_line(", ".join(spec["authors"]), size=11)
    writer.add_line(f"{spec.get('venue', '')} {spec.get('year', '')}".strip(), size=10)
    writer.add_heading("Abstract", size=13)
    writer.add_paragraph(spec["abstract"])
    writer.add_line(f"Keywords: {spec.get('keywords', '')}", size=10)
    for heading, paragraphs in spec["sections"]:
        writer.add_heading(heading, size=13)
        for paragraph in paragraphs:
            writer.add_paragraph(paragraph)
    return writer


def write_sample_pdfs(directory: Path | str, specs: Sequence[dict] = SAMPLE_PAPERS) -> List[Path]:
    """把全部样例论文写入目录，返回生成的文件路径列表。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for spec in specs:
        writer = build_sample_paper_pdf(spec)
        written.append(writer.save(target / spec["filename"]))
    return written


if __name__ == "__main__":  # pragma: no cover
    import sys

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/samples")
    for path in write_sample_pdfs(out_dir):
        print(f"生成: {path} ({path.stat().st_size} bytes)")
