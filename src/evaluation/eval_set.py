"""模块五 —— 评测集构建（课程要求 5.5.1：50+ 问答对，覆盖四类问题）。

设计要点
--------
1. **基于真实语料**：全部问答对都由 :mod:`tests.fixtures.make_sample_pdf` 中的
   三篇样例论文（Transformer / ResNet / BERT）真实内容构造。``gold_texts`` 必须是
   **论文里逐字出现的连续片段** —— 只有这样 ``RetrievalEngine.evaluate(match="text")``
   的"金标准文本 ↔ 检索片段重合度"判定才有意义。一旦改写金标准文本，
   Recall@K 会凭空掉下来，评测就失去了诊断价值。
2. **四类问题均衡分布**：factual / comparison / summary / reasoning 各占一定比例，
   分别考察"事实定位""跨文档对照""全局概括""多跳推理"四种能力。
   单一类型（例如全是 factual）的评测集会系统性高估系统能力。
3. **中英混合提问**：约一半中文、一半英文。样例语料是英文，因此中文问题的
   ``gold_texts`` 特意挑选**含英文术语**的片段（WMT 2014 / BLEU / ImageNet /
   masked language model …），让离线 hash 词面向量 + BM25 也能靠术语匹配命中。
   这是对"中文提问检索英文语料"这一真实使用场景的诚实覆盖 ——
   纯中文表述（无任何英文术语）的问题在本系统上召回率天然偏低，评测集里
   保留少量此类问题用于暴露该局限（见 notes 字段）。
4. **可复现**：``difficulty`` 只依据"需要几次检索/几跳推理"客观标注，不依赖主观感受；
   每条都带 ``notes`` 说明考察点，便于他人复现与扩展。

数据结构
--------
::

    {
      "id": "q001",
      "question": "What datasets were used to train the Transformer?",
      "type": "factual",                 # factual | comparison | summary | reasoning
      "gold_doc_titles": ["Attention Is All You Need"],
      "gold_texts": ["WMT 2014 English-German dataset consisting of about 4.5"],
      "reference_answer": "...",
      "difficulty": "easy",              # easy | medium | hard
      "expected_tools": ["knowledge_search"],   # 供 Agent 工具选择准确率评测使用
      "notes": ""
    }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "QUESTION_TYPES",
    "DIFFICULTIES",
    "DOC_TRANSFORMER",
    "DOC_RESNET",
    "DOC_BERT",
    "build_eval_set",
    "save_eval_set",
    "load_eval_set",
    "validate_eval_set",
    "eval_set_path",
    "eval_set_statistics",
]

# ==============================================================================
# 常量
# ==============================================================================
QUESTION_TYPES = ("factual", "comparison", "summary", "reasoning")
DIFFICULTIES = ("easy", "medium", "hard")

#: 三篇样例论文的标题（与 ``tests/fixtures/make_sample_pdf.py`` 的 ``SAMPLE_PAPERS`` 一致）
DOC_TRANSFORMER = "Attention Is All You Need"
DOC_RESNET = "Deep Residual Learning for Image Recognition"
DOC_BERT = "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding"

#: 评测集默认落盘位置（相对项目根目录）
DEFAULT_EVAL_SET_PATH = "reports/评测集.json"

#: 各类问题的目标条数（合计 ≥ 60）
TARGET_DISTRIBUTION = {"factual": 24, "comparison": 12, "summary": 12, "reasoning": 12}


# ==============================================================================
# 评测集定义
# ==============================================================================
def _sample(
    qid: str,
    question: str,
    qtype: str,
    docs: Sequence[str],
    texts: Sequence[str],
    reference: str,
    difficulty: str,
    expected_tools: Sequence[str] = ("knowledge_search",),
    notes: str = "",
) -> Dict[str, Any]:
    """构造一条评测样本（统一字段顺序，便于人工审阅时快速定位）。"""
    return {
        "id": qid,
        "question": question,
        "type": qtype,
        "gold_doc_titles": list(docs),
        "gold_texts": list(texts),
        "reference_answer": reference,
        "difficulty": difficulty,
        "expected_tools": list(expected_tools),
        "notes": notes,
    }


def build_eval_set() -> List[Dict[str, Any]]:
    """构造完整评测集（≥ 60 条，四类问题均衡）。

    ``gold_texts`` 一律取自样例论文原文的连续片段；``reference_answer`` 中的
    数字与结论同样来自原文（不做任何"合理推测"式补全）。

    Returns
    -------
    list of dict
        评测样本列表，字段见模块 docstring。ID 形如 ``q001``… 连续编号。
    """
    items: List[Dict[str, Any]] = []
    counter = {"n": 0}

    def add(
        question: str,
        qtype: str,
        docs: Sequence[str],
        texts: Sequence[str],
        reference: str,
        difficulty: str,
        expected_tools: Sequence[str] = ("knowledge_search",),
        notes: str = "",
    ) -> None:
        counter["n"] += 1
        items.append(
            _sample(
                f"q{counter['n']:03d}", question, qtype, docs, texts,
                reference, difficulty, expected_tools, notes,
            )
        )

    # --------------------------------------------------------------------------
    # 一、factual —— 事实定位（24 条）：答案唯一、可在原文中精确定位
    # --------------------------------------------------------------------------
    t = DOC_TRANSFORMER
    r = DOC_RESNET
    b = DOC_BERT

    # 1-6 Transformer 训练与结果
    add(
        "What datasets were used to train the Transformer?",
        "factual", [t],
        ["WMT 2014 English-German dataset consisting of about 4.5"],
        "Transformer 在 WMT 2014 英德数据集（约 450 万句对）和 WMT 2014 "
        "英法数据集（3600 万句）上训练，句子使用 byte-pair encoding 编码。",
        "easy", notes="数据集类事实；同时在英文与中文问法下考察召回稳定性",
    )
    add(
        "Transformer 训练用了哪些数据集？",
        "factual", [t],
        ["WMT 2014 English-German dataset consisting of about 4.5"],
        "WMT 2014 英德数据集（约 450 万句对）与 WMT 2014 英法数据集（3600 万句）。",
        "easy", notes="中文提问，靠英文术语 WMT 2014 命中",
    )
    add(
        "What BLEU score does the big transformer model achieve on the WMT 2014 English-to-German task?",
        "factual", [t],
        ["the big transformer model achieves 28.4 BLEU"],
        "在 WMT 2014 英德翻译任务上，big transformer 取得 28.4 BLEU，"
        "比此前最佳结果高出 2.0 BLEU 以上。",
        "easy", notes="关键指标数字，必须原样输出（28.4）",
    )
    add(
        "big transformer 在 WMT 2014 英德任务上的 BLEU 分数是多少？",
        "factual", [t],
        ["the big transformer model achieves 28.4 BLEU"],
        "28.4 BLEU。",
        "easy", notes="中文问法 + 阿拉伯数字答案",
    )
    add(
        "What BLEU score does the big model achieve on the WMT 2014 English-to-French task?",
        "factual", [t],
        ["the big model achieves a BLEU score of 41.8"],
        "在 WMT 2014 英法任务上，big model 取得 41.8 BLEU。",
        "easy",
    )
    add(
        "How many sentences are in the WMT 2014 English-French dataset?",
        "factual", [t],
        ["the WMT 2014 English-French dataset consisting of 36M sentences"],
        "3600 万（36M）句。",
        "easy", notes="数字型事实；考察是否出现数量级错误",
    )

    # 7-13 Transformer 架构与算法
    add(
        "How many identical layers is the Transformer encoder composed of?",
        "factual", [t],
        ["The encoder is composed of a stack of six"],
        "编码器由 6 个完全相同的层堆叠而成。",
        "easy",
    )
    add(
        "Transformer 的编码器由多少个相同的层堆叠而成？",
        "factual", [t],
        ["The encoder is composed of a stack of six"],
        "6 层。",
        "easy", notes="中文提问，靠 Transformer/encoder 术语与数字命中",
    )
    add(
        "In a self-attention layer, where do the keys, values and queries come from?",
        "factual", [t],
        ["In a self-attention layer all of the"],
        "自注意力层中 keys、values、queries 全部来自同一处 —— "
        "编码器中前一层的输出。",
        "medium",
    )
    add(
        "What is the total computational complexity per layer for self-attention?",
        "factual", [t],
        ["self-attention compared to O(n d^2) for"],
        "每层自注意力的计算复杂度为 O(n^2 d)，对比循环层的 O(n d^2)，"
        "其中 n 为序列长度、d 为表示维度。",
        "medium", notes="含公式表达，考察数字/符号还原能力",
    )
    add(
        "Why is multi-head attention used instead of a single attention head?",
        "factual", [t],
        ["Multi-head attention allows the model to jointly attend to information from different representation"],
        "多头注意力让模型能够同时关注不同表示子空间、不同位置的信息；"
        "单个注意力头会因平均化而抑制这种能力。",
        "medium", notes="机制类事实：原因句在原文中明确给出",
    )
    add(
        "How does Scaled Dot-Product Attention compute the weights on the values?",
        "factual", [t],
        ["We compute the dot products of the query with all keys"],
        "先计算 query 与所有 key 的点积，各自除以 sqrt(dk)，"
        "再用 softmax 得到作用在 value 上的权重。",
        "medium",
    )
    add(
        "What is the dimension of the queries and keys in Scaled Dot-Product Attention?",
        "factual", [t],
        ["of dimension dk, and values of dimension dv."],
        "queries 与 keys 的维度为 dk，values 的维度为 dv。",
        "medium",
    )

    # 14-19 ResNet
    add(
        "What top-5 error does the ensemble ResNet achieve on the ImageNet test set?",
        "factual", [r],
        ["The ensemble ResNet achieves 3.57 percent top-5 error on the"],
        "集成 ResNet 在 ImageNet 测试集上取得 3.57% 的 top-5 错误率。",
        "easy",
    )
    add(
        "ResNet 在 ImageNet 测试集上的 top-5 错误率是多少？",
        "factual", [r],
        ["The ensemble ResNet achieves 3.57 percent top-5 error on the"],
        "3.57%。",
        "easy", notes="中文提问，靠 ResNet/ImageNet/5 等术语命中",
    )
    add(
        "How many training images does the ImageNet 2012 classification dataset contain?",
        "factual", [r],
        ["models are trained on the 1.28 million training images and evaluated on the 50k validation images."],
        "ImageNet 2012 含 1000 个类别，使用 128 万张训练图像，"
        "在 5 万张验证图像上评估。",
        "easy",
    )
    add(
        "How many FLOPs does the 152-layer ResNet have compared with VGG?",
        "factual", [r],
        ["Our 152-layer ResNet has 11.3 billion FLOPs, still lower than"],
        "152 层 ResNet 为 113 亿（11.3 billion）FLOPs，"
        "低于 VGG-16 的 153 亿与 VGG-19 的 196 亿 FLOPs。",
        "medium",
    )
    add(
        "What is the formulation that the stacked nonlinear layers are asked to fit in residual learning?",
        "factual", [r],
        ["we let the stacked nonlinear layers fit another mapping of F(x) = H(x) - x"],
        "让堆叠的非线性层去拟合残差映射 F(x) = H(x) - x，"
        "原始映射被改写为 F(x) + x。",
        "medium", notes="含数学符号，考察公式还原",
    )
    add(
        "What are shortcut connections in the residual learning formulation?",
        "factual", [r],
        ["Shortcut connections are those skipping one or more layers."],
        "shortcut connection 指跳过一层或多层的连接，"
        "用于实现 F(x) + x 的前馈网络结构。",
        "easy",
    )
    add(
        "Does the ResNet paper report degradation caused by overfitting?",
        "factual", [r],
        ["network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such"],
        "不是。论文明确指出退化问题并非由过拟合引起。",
        "medium", notes="否定型事实，考察是否误读原文",
    )

    # 20-24 BERT
    add(
        "What model sizes does BERT report: layers, hidden size and attention heads?",
        "factual", [b],
        ["BERT BASE with L=12, H=768, A=12, and BERT LARGE"],
        "BERT BASE：L=12、H=768、A=12；BERT LARGE：L=24、H=1024、A=16。",
        "medium",
    )
    add(
        "BERT BASE 的层数、隐藏层维度和注意力头数分别是多少？",
        "factual", [b],
        ["BERT BASE with L=12, H=768, A=12, and BERT LARGE"],
        "L=12、H=768、A=12。",
        "medium", notes="中文提问，靠 BERT BASE / L=12 / H=768 等英文术语命中",
    )
    add(
        "What vocabulary size does BERT WordPiece use?",
        "factual", [b],
        ["BERT uses WordPiece embeddings with a 30000 token vocabulary."],
        "BERT 使用 30000 词表的 WordPiece 嵌入。",
        "easy",
    )
    add(
        "What F1 does BERT achieve on SQuAD v1.1 question answering?",
        "factual", [b],
        ["BERT achieves 93.2 F1 on the test set"],
        "在 SQuAD v1.1 上 BERT 于测试集取得 93.2 F1，"
        "比此前最佳集成系统提升 1.5。",
        "easy",
    )
    add(
        "What is the second pre-training task of BERT?",
        "factual", [b],
        ["The second pre-training task is next sentence prediction."],
        "第二个预训练任务是下一句预测（next sentence prediction）。",
        "easy",
    )

    # --------------------------------------------------------------------------
    # 二、comparison —— 跨文档 / 跨方案对照（12 条）
    # --------------------------------------------------------------------------
    add(
        "Compare how the Transformer and ResNet reduce the difficulty of training deep models.",
        "comparison", [t, r],
        ["Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions",
         "let these layers fit a residual mapping."],
        "Transformer 从序列建模入手：完全抛弃循环与卷积，只用注意力机制，"
        "从而解除序列依赖带来的并行化限制；ResNet 从优化难度入手："
        "让堆叠层拟合残差映射 F(x) = H(x) - x，使极深网络可训练。"
        "两者都提升了「深」模型的可训练性，但一个改结构、一个改学习目标。",
        "hard", ["paper_compare", "knowledge_search"],
        notes="跨文档对比：证据必须来自两篇不同论文",
    )
    add(
        "对比 Transformer 与 BERT 在预训练与训练目标上的区别。",
        "comparison", [t, b],
        ["The total computational complexity per layer is O(n^2 d) for self-attention",
         "we simply mask some percentage of the input"],
        "Transformer 是端到端的序列转换架构，用编码器-解码器与多头自注意力做机器翻译，"
        "训练目标是翻译任务本身；BERT 只取 Transformer 的编码器结构，"
        "通过随机 mask 输入 token 并预测（masked language model）做双向预训练，"
        "再用微调迁移到下游任务。",
        "hard", ["paper_compare", "knowledge_search"],
    )
    add(
        "Compare the training data scale of the Transformer and ResNet experiments.",
        "comparison", [t, r],
        ["We trained on the WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs, and",
         "models are trained on the 1.28 million training images and evaluated on the 50k validation images."],
        "Transformer 使用 WMT 2014 英德 450 万句对（另有英法 3600 万句）；"
        "ResNet 使用 ImageNet 2012 的 128 万张训练图像、5 万张验证图像。"
        "两者量级接近但任务类型不同（句对 vs 图像），不宜直接比大小。",
        "medium", ["paper_compare", "knowledge_search"],
        notes="对比型问题，但正确答案需要指出「不可直接比较」",
    )
    add(
        "How do the Transformer's BLEU results on English-German and English-French compare with what the paper reports as previous best?",
        "comparison", [t],
        ["outperforming the best previously reported results by more"],
        "英德任务 28.4 BLEU，比此前最佳结果高出 2.0 BLEU 以上；"
        "英法任务 41.8 BLEU，以不到四分之一训练成本超越所有已发表单模型。",
        "medium", ["knowledge_search"],
    )
    add(
        "Compare the goals of the degradation problem in ResNet and the unidirectional limitation in BERT.",
        "comparison", [r, b],
        ["degradation is not caused by overfitting.",
         "they are unidirectional, which limits the"],
        "ResNet 的退化问题是**优化层面**的：层数增加时准确率先饱和后快速下降，"
        "且并非过拟合；BERT 指出的单向性是**表示层面**的："
        "标准语言模型只能从左到右建模，限制了预训练可用的架构。"
        "两者分别通过残差学习与 masked language model 解决。",
        "hard", ["paper_compare", "knowledge_search"],
    )
    add(
        "对比 Transformer 中自注意力与循环层的计算复杂度差异。",
        "comparison", [t],
        ["self-attention compared to O(n d^2) for"],
        "每层自注意力复杂度为 O(n^2 d)，循环层为 O(n d^2)（n 为序列长度，"
        "d 为表示维度）。序列较短时循环层更省算力，但循环层无法在序列内并行，"
        "这是自注意力在大规模训练上的关键优势。",
        "medium", ["knowledge_search"],
    )
    add(
        "Compare plain networks and residual networks as depth increases in the ResNet paper.",
        "comparison", [r],
        ["The residual networks instead show"],
        "普通网络加深后验证误差反而升高（34 层差于 18 层）；"
        "残差网络则随深度增加准确率持续提升。",
        "medium", ["knowledge_search"],
    )
    add(
        "Compare the pre-training tasks used by BERT and the training objective of the Transformer.",
        "comparison", [b, t],
        ["We refer to this procedure as a masked language model,",
         "the big transformer model achieves 28.4 BLEU"],
        "BERT 的预训练目标是 masked language model 与 next sentence prediction"
        "（自监督、双任务）；Transformer 是有监督的序列到序列翻译目标"
        "（WMT 2014 英德任务取得 28.4 BLEU）。前者学通用语言表示，后者直接学任务映射。",
        "hard", ["paper_compare", "knowledge_search"],
    )
    add(
        "对比 ResNet 与 VGG 在同等精度下的计算量。",
        "comparison", [r],
        ["Our 152-layer ResNet has 11.3 billion FLOPs, still lower than"],
        "152 层 ResNet 为 11.3 billion FLOPs，低于 VGG-16 的 15.3 billion 与 "
        "VGG-19 的 19.6 billion，即在更深的前提下计算量反而更低。",
        "medium", ["knowledge_search"],
    )
    add(
        "Which paper reports 3.57 percent top-5 error, and which reports 28.4 BLEU?",
        "comparison", [r, t],
        ["The ensemble ResNet achieves 3.57 percent top-5 error on the",
         "the big transformer model achieves 28.4 BLEU"],
        "3.57% top-5 error 来自 Deep Residual Learning for Image Recognition"
        "（ImageNet 分类）；28.4 BLEU 来自 Attention Is All You Need"
        "（WMT 2014 英德翻译）。两个指标属于不同任务，不可横向比较。",
        "medium", ["paper_compare", "knowledge_search"],
        notes="数字 ↔ 论文归属的对照，专治「张冠李戴」",
    )
    add(
        "Contrast the evaluation tasks used by the Transformer, ResNet and BERT papers.",
        "comparison", [t, r, b],
        ["Experiments on two machine translation tasks show these models",
         "achieving 3.57 percent top-5 error and winning the ILSVRC 2015 classification task.",
         "BERT advances the state of the art for eleven NLP tasks."],
        "Transformer 在 WMT 2014 两个机器翻译任务上评估；"
        "ResNet 在 ImageNet 2012 分类（ILSVRC 2015）上评估；"
        "BERT 在 11 个 NLP 任务（GLUE、SQuAD v1.1 等）上评估。",
        "hard", ["paper_compare", "knowledge_search"],
    )
    add(
        "对比三篇论文的核心贡献与总体规模指标。",
        "comparison", [t, r, b],
        ["Experiments on two machine translation tasks show these models",
         "achieving 3.57 percent top-5 error and winning the ILSVRC 2015 classification task.",
         "obtains an average accuracy of 80.5 percent"],
        "Transformer 提出纯注意力架构（28.4/41.8 BLEU）；"
        "ResNet 提出残差学习使 152 层网络可训练（3.57% top-5 error）；"
        "BERT 提出双向预训练语言表示（GLUE 平均 80.5%）。",
        "hard", ["paper_compare", "knowledge_search"],
    )

    # --------------------------------------------------------------------------
    # 三、summary —— 概括类（12 条）：需要多片段/多文档聚合
    # --------------------------------------------------------------------------
    add(
        "Summarize the main contributions of the Transformer paper.",
        "summary", [t],
        ["Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions",
         "In this work, we presented the Transformer, the first sequence transduction model based entirely on"],
        "论文提出 Transformer —— 首个完全基于注意力机制的序列转换模型，"
        "用多头自注意力替代编码器-解码器中最常用的循环层，"
        "在两项机器翻译任务上取得更高质量，同时更易并行、训练时间显著更短。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "总结 Attention Is All You Need 这篇论文的主要内容。",
        "summary", [t],
        ["Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions",
         "In this work, we presented the Transformer, the first sequence transduction model based entirely on"],
        "提出 Transformer 架构：完全摒弃循环与卷积，仅用（多头）自注意力；"
        "给出 Scaled Dot-Product Attention、编码器-解码器结构；"
        "在 WMT 2014 英德/英法任务上取得 28.4 / 41.8 BLEU。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Summarize the datasets and evaluation metrics used in the Transformer paper.",
        "summary", [t],
        ["WMT 2014 English-German dataset consisting of about 4.5",
         "the big transformer model achieves 28.4 BLEU"],
        "数据集：WMT 2014 英德（450 万句对）与英法（3600 万句）；"
        "评价指标：BLEU（英德 28.4、英法 41.8）。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Give an overview of the main results reported in the ResNet paper.",
        "summary", [r],
        ["networks. Our results demonstrate that residual learning is generic and can be applied to other",
         "The ensemble ResNet achieves 3.57 percent top-5 error on the"],
        "残差学习框架使极深网络易于训练：18/34 层的对比实验显示残差网络精度随深度提升；"
        "152 层 ResNet 仅 11.3 billion FLOPs；集成模型在 ImageNet 测试集取得 3.57% "
        "top-5 错误率并赢得 ILSVRC 2015 分类任务。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "概括 Deep Residual Learning for Image Recognition 的实验设置与结论。",
        "summary", [r],
        ["We test our method on the ImageNet 2012 classification dataset that consists of 1000 classes. The",
         "networks. Our results demonstrate that residual learning is generic and can be applied to other"],
        "实验设置：ImageNet 2012（1000 类、128 万训练图像、5 万验证图像），"
        "对比 18/34 层普通网络与残差网络，并训练 152 层模型；"
        "结论：残差学习通用、可迁移到视觉与非视觉问题，深度增加不再带来退化。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Summarize the BERT paper's model architecture and pre-training tasks.",
        "summary", [b],
        ["we simply mask some percentage of the input",
         "BERT BASE with L=12, H=768, A=12, and BERT LARGE"],
        "架构：多层双向 Transformer 编码器，BASE（L=12/H=768/A=12）与 "
        "LARGE（L=24/H=1024/A=16）两种规模；预训练任务：masked language model（MLM）"
        "与 next sentence prediction。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "总结 BERT 论文在 GLUE 与 SQuAD 上的性能表现。",
        "summary", [b],
        ["obtains an average accuracy of 80.5 percent",
         "BERT achieves 93.2 F1 on the test set"],
        "GLUE 上 BERT LARGE 平均准确率 80.5%，比此前 SOTA 绝对提升 7.6 个百分点；"
        "SQuAD v1.1 测试集 93.2 F1，比此前最佳集成系统提升 1.5。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Provide a structured summary of the Transformer architecture.",
        "summary", [t],
        ["Most competitive neural sequence transduction models have an encoder-decoder structure. The encoder",
         "connected layers for both the encoder and decoder. The encoder is composed of a stack of six"],
        "结构：编码器-解码器；编码器为 6 层相同层的堆叠，每层含自注意力与前馈层；"
        "注意力采用 Scaled Dot-Product Attention 与多头机制；"
        "每层自注意力复杂度 O(n^2 d)。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Overview the three papers' target tasks, datasets and headline results.",
        "summary", [t, r, b],
        ["Experiments on two machine translation tasks show these models",
         "achieving 3.57 percent top-5 error and winning the ILSVRC 2015 classification task.",
         "obtains an average accuracy of 80.5 percent"],
        "Transformer：机器翻译（WMT 2014 英德/英法），28.4 / 41.8 BLEU；"
        "ResNet：图像分类（ImageNet 2012），3.57% top-5 error；"
        "BERT：11 项 NLP 任务（GLUE/SQuAD），GLUE 平均 80.5%、SQuAD 93.2 F1。",
        "hard", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Summarize how the Transformer paper motivates attention over recurrence.",
        "summary", [t],
        ["This inherently sequential nature precludes",
         "allowing modeling of dependencies without regard to their distance in the"],
        "动机：循环模型沿符号位置逐步计算，这种固有的顺序性阻碍了训练样本内的并行化，"
        "序列越长越严重；注意力机制可以在不考虑距离的情况下建模依赖关系，"
        "因此论文用注意力完全替代循环结构。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "概括 ResNet 论文提出的退化问题及其解决方案。",
        "summary", [r],
        ["a degradation problem has been exposed",
         "mapping as H(x), we let the stacked nonlinear layers fit another mapping of F(x) = H(x) - x. The"],
        "问题：网络加深到一定程度后准确率饱和并快速下降，且该退化并非过拟合导致；"
        "方案：引入残差学习框架，让堆叠层拟合 F(x) = H(x) - x，"
        "通过 shortcut connection 实现 F(x) + x。",
        "medium", ["summarize_paper", "knowledge_search"],
    )
    add(
        "Summarize the ablation findings reported in the BERT paper.",
        "summary", [b],
        ["relative importance. The model without the next sentence prediction task hurts performance on the",
         "Training with a masked language model is crucial"],
        "消融结论：去掉 next sentence prediction 会显著损害自然语言推理任务的表现；"
        "masked language model 至关重要 —— 从左到右的语言模型在每一项任务上都更差，"
        "在问答与命名实体识别上差距尤为显著。",
        "medium", ["summarize_paper", "knowledge_search"],
    )

    # --------------------------------------------------------------------------
    # 四、reasoning —— 需要多跳推理 / 因果解释（12 条）
    # --------------------------------------------------------------------------
    add(
        "Why does the Transformer replace recurrence with self-attention instead of simply stacking more recurrent layers?",
        "reasoning", [t],
        ["This inherently sequential nature precludes"],
        "因为循环结构的逐位置计算天然串行，无法在一个训练样本内并行化，"
        "序列变长时内存约束又限制了跨样本批处理；"
        "而自注意力不依赖序列距离建模依赖，可并行计算，"
        "因此能显著缩短训练时间并提升质量。",
        "hard", ["knowledge_search"],
    )
    add(
        "为什么 Transformer 要用自注意力替代循环结构？",
        "reasoning", [t],
        ["This inherently sequential nature precludes"],
        "循环结构的顺序性阻碍样本内并行化，长序列时尤为严重；"
        "自注意力允许不考虑距离地建模依赖关系，可并行且训练更快。",
        "hard", ["knowledge_search"],
        notes="中文因果推理，靠术语并行化/注意力/序列命中",
    )
    add(
        "Explain why BERT uses a masked language model instead of a standard left-to-right language model.",
        "reasoning", [b],
        ["they are unidirectional, which limits the"],
        "标准语言模型是单向的，这限制了预训练阶段可用的架构；"
        "随机 mask 一部分输入 token 再预测它们，使模型能同时利用左右两侧上下文，"
        "从而学到真正的双向表示。",
        "hard", ["knowledge_search"],
    )
    add(
        "How does the complexity of self-attention compare with recurrent layers, and when is each preferable?",
        "reasoning", [t],
        ["self-attention compared to O(n d^2) for"],
        "自注意力为 O(n^2 d)，循环层为 O(n d^2)。当序列长度 n 小于表示维度 d 时"
        "自注意力更省算力，且可并行；n 很大时自注意力的平方项会成为瓶颈，"
        "这也是论文提出受限局部注意力的动机。",
        "hard", ["knowledge_search"],
    )
    add(
        "Why does adding more layers to a plain network hurt accuracy, and how does residual learning fix it?",
        "reasoning", [r],
        ["The degradation problem suggests that the solvers might have difficulties in"],
        "退化并非过拟合，而是深层网络难以用多个非线性层逼近恒等映射；"
        "残差重写后，若恒等映射最优，求解器只需把非线性层权重推向零即可，"
        "优化问题因此变得容易，精度可随深度持续提升。",
        "hard", ["knowledge_search"],
    )
    add(
        "为什么加深普通网络反而会导致精度下降？",
        "reasoning", [r],
        ["When deeper networks are able to start converging, a degradation problem has been exposed: with the"],
        "这是退化问题而非过拟合：网络越深，准确率先饱和后快速下降。"
        "残差学习通过让层拟合残差映射 F(x) = H(x) - x 缓解该问题。",
        "hard", ["knowledge_search"],
    )
    add(
        "Explain why the 152-layer ResNet uses fewer FLOPs than VGG-16 despite being much deeper.",
        "reasoning", [r],
        ["Our 152-layer ResNet has 11.3 billion FLOPs, still lower than"],
        "计算量取决于每层的通道数与空间分辨率，而非层数："
        "ResNet 通过全局平均池化与瓶颈式的通道配置显著降低了单层开销，"
        "因此 152 层（11.3 billion FLOPs）仍低于 VGG-16（15.3 billion）。",
        "hard", ["knowledge_search"],
        notes="多跳推理：需要把 FLOPs 数值与网络设计常识结合",
    )
    add(
        "Why does the Transformer encoder use six identical layers rather than heterogeneous layers?",
        "reasoning", [t],
        ["The Transformer follows this overall architecture using stacked self-attention"],
        "原文只说明编码器由 6 个相同层堆叠、解码器同样采用堆叠自注意力与前馈层，"
        "并给出每层复杂度 O(n^2 d)。"
        "「为什么选择同构堆叠」论文未直接论证 —— 只能推断同构堆叠便于参数共享与深堆叠，"
        "属于推断而非原文结论。",
        "hard", ["knowledge_search"],
        notes="**拒答型样本**：考察系统是否会把推断伪装成原文结论（faithfulness 会暴露）",
    )
    add(
        "What did the BERT ablation show about the relative importance of the two pre-training tasks?",
        "reasoning", [b],
        ["relative importance. The model without the next sentence prediction task hurts performance on the",
         "Training with a masked language model is crucial"],
        "消融显示 masked language model 更关键：去掉它后每项任务都变差；"
        "去掉 next sentence prediction 主要损害自然语言推理类任务。"
        "即 MLM 提供通用语言表示能力，NSP 更偏向句间关系任务。",
        "hard", ["knowledge_search"],
    )
    add(
        "How does the illusion of a single attention head relate to the need for multiple heads?",
        "reasoning", [t],
        ["Multi-head attention allows the model to jointly attend to information from different representation",
         "subspaces at different positions. With a single attention head, averaging inhibits this."],
        "多头注意力可以并行关注不同表示子空间与不同位置的信息；"
        "若只用单个注意力头，平均化会抑制这种多子空间建模能力，"
        "因此多头是必要的。",
        "hard", ["knowledge_search"],
    )
    add(
        "为什么 BERT 需要微调而不是直接使用预训练模型？",
        "reasoning", [b],
        ["The feature-based approach uses task-specific architectures",
         "we simply mask some percentage of the input"],
        "预训练只提供通用双向语言表示；下游任务形态各异，"
        "因此论文采用微调策略（区别于固定特征的特征式做法），"
        "用少量任务数据把预训练表示适配到具体任务上。",
        "hard", ["knowledge_search"],
    )
    add(
        "Given the abstract-level claims of the three papers, which one's central claim is best supported by a single quantitative result?",
        "reasoning", [t, r, b],
        ["achieving 3.57 percent top-5 error and winning the ILSVRC 2015 classification task.",
         "the big transformer model achieves 28.4 BLEU"],
        "三篇论文都给出了可核验的量化结果：Transformer 28.4/41.8 BLEU、"
        "ResNet 3.57% top-5 error 并赢得 ILSVRC 2015、BERT GLUE 平均 80.5%。"
        "其中 ResNet 的结论与「赢得 ILSVRC 2015 分类任务」这一外部可验证事实绑定最紧，"
        "单一数字（3.57%）即可支撑其核心主张。",
        "hard", ["paper_compare", "knowledge_search"],
        notes="开放性判断题：考察是否给出多篇证据并明确推理依据",
    )

    return items


# ==============================================================================
# 读写
# ==============================================================================
def eval_set_path(config: Any = None) -> Path:
    """返回评测集 JSON 的默认落盘路径（``reports/评测集.json``）。"""
    if config is not None:
        try:
            report_dir = config.resolve_path(str(config.paths.report_dir))
            return Path(report_dir) / "评测集.json"
        except Exception:  # noqa: BLE001 - 配置缺字段时退回默认值
            pass
    from ..utils.config import PROJECT_ROOT

    return PROJECT_ROOT / DEFAULT_EVAL_SET_PATH


def save_eval_set(items: Sequence[Dict[str, Any]], path: Any = None) -> Path:
    """把评测集写入 JSON（UTF-8，``ensure_ascii=False``，便于人工审阅中文）。

    Parameters
    ----------
    items: 评测样本列表。
    path: 目标路径；None 表示 ``reports/评测集.json``。

    Returns
    -------
    Path: 实际写入的文件路径。
    """
    from ..utils.storage import ensure_dir

    target = Path(path) if path else eval_set_path()
    ensure_dir(target.parent)
    payload = {
        "name": "RAG+Agent 论文问答系统评测集",
        "version": "1.0",
        "num_items": len(list(items)),
        "type_distribution": dict(Counter(i.get("type", "") for i in items)),
        "difficulty_distribution": dict(Counter(i.get("difficulty", "") for i in items)),
        "documents": [DOC_TRANSFORMER, DOC_RESNET, DOC_BERT],
        "items": list(items),
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def load_eval_set(path: Any = None) -> List[Dict[str, Any]]:
    """读取评测集 JSON，返回样本列表。

    同时兼容两种文件形态：带 ``items`` 包装的完整文件，以及裸列表文件。
    文件不存在或损坏时返回**内置评测集**（``build_eval_set()``），
    这样评测脚本在干净环境下也能直接跑通。
    """
    target = Path(path) if path else eval_set_path()
    if not target.exists():
        return build_eval_set()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return build_eval_set()
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, dict)]
        return []
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    return []


# ==============================================================================
# 校验与统计
# ==============================================================================
REQUIRED_FIELDS = (
    "id", "question", "type", "gold_doc_titles",
    "gold_texts", "reference_answer", "difficulty",
)


def validate_eval_set(
    items: Sequence[Dict[str, Any]],
    min_items: int = 60,
    check_types: bool = True,
) -> Dict[str, Any]:
    """校验评测集的完整性、分布与格式，返回**报告字典**（不抛异常）。

    校验项
    ------
    1. 数量 ≥ ``min_items``（默认 60，课程要求 50+）；
    2. 字段完整性：``REQUIRED_FIELDS`` 全部存在且非空（列表字段非空列表）；
    3. ``type`` ∈ {factual, comparison, summary, reasoning}，且四类**都有**样本；
    4. ``difficulty`` ∈ {easy, medium, hard}；
    5. ``id`` 唯一且非空；
    6. ``gold_texts`` 每条至少 12 个字符（太短无法支撑 match="text" 的重合判定）；
    7. （可选）``expected_tools`` 若存在必须是非空字符串列表。

    Returns
    -------
    dict
        ``{"passed", "num_items", "errors", "warnings", "type_distribution",
        "difficulty_distribution", "language_distribution", "missing_types",
        "num_docs"}``。``errors`` 非空即 ``passed=False``。

    适用场景：CI 门禁、评测集人工扩充后的回归检查。
    """
    errors: List[str] = []
    warnings: List[str] = []
    rows = list(items or ())

    if len(rows) < min_items:
        errors.append(f"评测集数量不足：{len(rows)} < {min_items}")

    type_counter: Counter = Counter()
    difficulty_counter: Counter = Counter()
    language_counter: Counter = Counter()
    seen_ids: set = set()
    docs: set = set()

    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            errors.append(f"第 {index} 条不是字典对象")
            continue
        qid = str(item.get("id") or "")
        tag = qid or f"#{index}"

        for field in REQUIRED_FIELDS:
            if field not in item:
                errors.append(f"[{tag}] 缺少字段 '{field}'")
                continue
            value = item[field]
            if isinstance(value, (list, tuple)):
                if not value:
                    errors.append(f"[{tag}] 字段 '{field}' 为空列表")
            elif not str(value).strip():
                errors.append(f"[{tag}] 字段 '{field}' 为空")

        if qid:
            if qid in seen_ids:
                errors.append(f"[{tag}] id 重复")
            seen_ids.add(qid)

        qtype = str(item.get("type") or "")
        if qtype:
            type_counter[qtype] += 1
            if qtype not in QUESTION_TYPES:
                errors.append(f"[{tag}] 未知问题类型 '{qtype}'")

        difficulty = str(item.get("difficulty") or "")
        if difficulty:
            difficulty_counter[difficulty] += 1
            if difficulty not in DIFFICULTIES:
                errors.append(f"[{tag}] 未知难度 '{difficulty}'")

        for text in item.get("gold_texts") or []:
            if len(str(text).strip()) < 12:
                warnings.append(f"[{tag}] gold_texts 片段过短（<12 字符）：{text!r}")

        for title in item.get("gold_doc_titles") or []:
            docs.add(str(title))

        if "expected_tools" in item:
            tools = item.get("expected_tools")
            if not isinstance(tools, (list, tuple)) or not tools:
                errors.append(f"[{tag}] expected_tools 必须是非空列表")
            elif any(not str(tool).strip() for tool in tools):
                errors.append(f"[{tag}] expected_tools 含空字符串")

        question = str(item.get("question") or "")
        language_counter["zh" if _has_cjk(question) else "en"] += 1

    missing_types = [t for t in QUESTION_TYPES if type_counter.get(t, 0) == 0]
    if check_types and missing_types:
        errors.append(f"缺少问题类型：{missing_types}")

    return {
        "passed": not errors,
        "num_items": len(rows),
        "num_documents": len(docs),
        "errors": errors,
        "warnings": warnings,
        "type_distribution": dict(type_counter),
        "difficulty_distribution": dict(difficulty_counter),
        "language_distribution": dict(language_counter),
        "missing_types": missing_types,
        "target_distribution": dict(TARGET_DISTRIBUTION),
    }


def _has_cjk(text: str) -> bool:
    """判断文本是否含中日韩统一表意文字（用于中英文分布统计）。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def eval_set_statistics(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """生成用于报告渲染的评测集统计（类型/难度/语言/文档覆盖/平均长度）。"""
    rows = list(items or ())
    report = validate_eval_set(rows, min_items=0, check_types=False)
    questions = [str(i.get("question") or "") for i in rows]
    gold_texts = [str(t) for i in rows for t in (i.get("gold_texts") or [])]
    report["avg_question_chars"] = (
        round(sum(len(q) for q in questions) / len(questions), 1) if questions else 0.0
    )
    report["avg_gold_texts_per_item"] = (
        round(len(gold_texts) / len(rows), 2) if rows else 0.0
    )
    return report


# ==============================================================================
# CLI 入口
# ==============================================================================
def _print_statistics(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """打印评测集统计信息，返回统计字典。"""
    stats = eval_set_statistics(items)
    print("=" * 68)
    print("RAG+Agent 论文问答系统 —— 评测集构建完成")
    print("=" * 68)
    print(f"样本总数      : {stats['num_items']}")
    print(f"覆盖论文数    : {stats['num_documents']}")
    print(f"平均问题长度  : {stats['avg_question_chars']} 字符")
    print(f"平均金标准片段: {stats['avg_gold_texts_per_item']} 条/题")
    print("-" * 68)
    print("问题类型分布：")
    for qtype in QUESTION_TYPES:
        count = stats["type_distribution"].get(qtype, 0)
        target = TARGET_DISTRIBUTION.get(qtype, 0)
        bar = "█" * count
        print(f"  {qtype:<11} {count:>3} 条 (目标 {target:>2})  {bar}")
    print("难度分布：")
    for level in DIFFICULTIES:
        count = stats["difficulty_distribution"].get(level, 0)
        print(f"  {level:<11} {count:>3} 条  {'█' * count}")
    print("语言分布：")
    for lang, label in (("en", "英文提问"), ("zh", "中文提问")):
        count = stats["language_distribution"].get(lang, 0)
        print(f"  {label:<11} {count:>3} 条  {'█' * count}")
    print("-" * 68)
    print(f"校验结果：{'通过 ✅' if stats['passed'] else '不通过 ❌'}")
    for error in stats["errors"]:
        print(f"  [错误] {error}")
    for warning in stats["warnings"][:5]:
        print(f"  [警告] {warning}")
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m src.evaluation.eval_set`` 入口：生成并写出评测集。"""
    from ..utils.config import load_config
    from ..utils.console import enable_utf8_output

    enable_utf8_output()

    args = list(argv if argv is not None else [])
    config = load_config(overrides={"paths": {"report_dir": "./reports"}})
    config.ensure_dirs()

    target = Path(args[0]) if args else eval_set_path(config)
    items = build_eval_set()
    stats = _print_statistics(items)
    if not stats["passed"]:
        print("\n评测集未通过校验，未写出文件。")
        return 1
    written = save_eval_set(items, target)
    print(f"\n已写出：{written}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工执行入口
    import sys

    raise SystemExit(main(sys.argv[1:]))
