# Bad Case 分析报告

> 生成时间：2026-10-10 23:08:10　|　分析样本数：**40**　|　Bad Case 总数：**13**　|　占比：**32.5%**

本报告全部数据来自实际运行（评测语料为 3 篇样例论文构建的临时索引），未运行的小节会显式标注「未运行」。

## 一、总览

| 失败类别 | 条数 | 占比 | 严重度分布 | 改进优先级 |
| --- | ---: | ---: | --- | --- |
| 检索失败（未召回相关文档） | 13 | 32.5% | high×13 | P1（影响面最大） |
| 生成错误（答案缺关键信息） | 0 | 0.0% | — | P2 |
| 引用错误（引用幻觉） | 0 | 0.0% | — | P0（可信度底线） |
| Agent 决策错误（工具选错/死循环/超轮次） | 0 | 0.0% | — | P2 |
| **合计** | **13** | **32.5%** | — | — |

各失败类别占比如下图（每格 = 5%）：

```text
检索失败（未召回相关文档）                ██████░░░░░░░░░░░░░░ 32.5%
生成错误（答案缺关键信息）                ░░░░░░░░░░░░░░░░░░░░ 0.0%
引用错误（引用幻觉）                   ░░░░░░░░░░░░░░░░░░░░ 0.0%
Agent 决策错误（工具选错/死循环/超轮次）     ░░░░░░░░░░░░░░░░░░░░ 0.0%
```

## 二、典型样例

### 检索失败（未召回相关文档）（13 条）

#### 样例 1：`q001`（严重度：high）

- **问题**：What datasets were used to train the Transformer?
- **期望**：WMT 2014 English-German dataset consisting of about 4.5
- **实际**：Top-5 召回片段均未覆盖金标准文本
- **根因**：检索层未召回相关片段：可能原因包括 (a) 中文提问与英文语料的词面/语义鸿沟（hash 词面向量对中英跨语言几乎无信号）；(b) 问题中的术语与原文表述差异大，BM25 也无法命中；(c) candidate_k 过小导致正确片段未能进入融合候选。
- **改进建议**：1) 调大 `retrieval.top_k`（5→8）与 `candidate_k`（20→30），给融合/重排更多候选；2) 确认 `retrieval.enable_bm25=true`（中文问题依赖 BM25 的术语字面匹配）；3) 若术语与原文差异大，把 `chunking.chunk_size` 从 512 调小到 384 提升片段聚焦度；4) 长期方案：把 `embedding.provider` 从 hash 换成 ollama/bge-m3 等真实语义向量模型。
- **证据**：

```json
{
  "gold_texts": [
    "WMT 2014 English-German dataset consisting of about 4.5"
  ],
  "gold_text_recall": 0.0,
  "num_retrieved": 5,
  "retrieved_preview": [
    "translation task, the big model achieves a BLEU score of 41.8,\noutperforming all of the previously published single models at less than one fourth of the traini",
    "e translation tasks show these models to be superior in quality\nwhile being more parallelizable and requiring significantly less time to train.\nKeywords: attent",
    "ng. The feature-based approach uses task-specific architectures\nthat include the pre-trained representations as additional features.\nThe major limitation of sta"
  ]
}
```

#### 样例 2：`q002`（严重度：high）

- **问题**：Transformer 训练用了哪些数据集？
- **期望**：WMT 2014 English-German dataset consisting of about 4.5
- **实际**：Top-5 召回片段均未覆盖金标准文本
- **根因**：检索层未召回相关片段：可能原因包括 (a) 中文提问与英文语料的词面/语义鸿沟（hash 词面向量对中英跨语言几乎无信号）；(b) 问题中的术语与原文表述差异大，BM25 也无法命中；(c) candidate_k 过小导致正确片段未能进入融合候选。
- **改进建议**：1) 调大 `retrieval.top_k`（5→8）与 `candidate_k`（20→30），给融合/重排更多候选；2) 确认 `retrieval.enable_bm25=true`（中文问题依赖 BM25 的术语字面匹配）；3) 若术语与原文差异大，把 `chunking.chunk_size` 从 512 调小到 384 提升片段聚焦度；4) 长期方案：把 `embedding.provider` 从 hash 换成 ollama/bge-m3 等真实语义向量模型。
- **证据**：

```json
{
  "gold_texts": [
    "WMT 2014 English-German dataset consisting of about 4.5"
  ],
  "gold_text_recall": 0.0,
  "num_retrieved": 5,
  "retrieved_preview": [
    "generates an output sequence of symbols one element at a time.\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fu",
    "Attention Is All You Need\nAshish Vaswani, Noam Shazeer, Niki Parmar\nNeurIPS 2017 2017\nAbstract\nThe dominant sequence transduction models are based on complex re",
    "e translation tasks show these models to be superior in quality\nwhile being more parallelizable and requiring significantly less time to train.\nKeywords: attent"
  ]
}
```

#### 样例 3：`q026`（严重度：high）

- **问题**：Compare how the Transformer and ResNet reduce the difficulty of training deep models.
- **期望**：Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions；let these layers fit a residual mapping.
- **实际**：Top-5 召回片段均未覆盖金标准文本
- **根因**：检索层未召回相关片段：可能原因包括 (a) 中文提问与英文语料的词面/语义鸿沟（hash 词面向量对中英跨语言几乎无信号）；(b) 问题中的术语与原文表述差异大，BM25 也无法命中；(c) candidate_k 过小导致正确片段未能进入融合候选。
- **改进建议**：1) 调大 `retrieval.top_k`（5→8）与 `candidate_k`（20→30），给融合/重排更多候选；2) 确认 `retrieval.enable_bm25=true`（中文问题依赖 BM25 的术语字面匹配）；3) 若术语与原文差异大，把 `chunking.chunk_size` 从 512 调小到 384 提升片段聚焦度；4) 长期方案：把 `embedding.provider` 从 hash 换成 ollama/bge-m3 等真实语义向量模型。
- **证据**：

```json
{
  "gold_texts": [
    "Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions",
    "let these layers fit a residual mapping."
  ],
  "gold_text_recall": 0.0,
  "num_retrieved": 5,
  "retrieved_preview": [
    "translation task, the big model achieves a BLEU score of 41.8,\noutperforming all of the previously published single models at less than one fourth of the traini",
    "ion training images and evaluated on the 50k validation images.\nWe compare plain networks and residual networks with 18 and 34 layers. The plain 34-layer networ",
    "ng. The feature-based approach uses task-specific architectures\nthat include the pre-trained representations as additional features.\nThe major limitation of sta"
  ]
}
```

- 该类改进方向（统一建议）：1) 调大 `retrieval.top_k`（5→8）与 `candidate_k`（20→30），给融合/重排更多候选；2) 确认 `retrieval.enable_bm25=true`（中文问题依赖 BM25 的术语字面匹配）；3) 若术语与原文差异大，把 `chunking.chunk_size` 从 512 调小到 384 提升片段聚焦度；4) 长期方案：把 `embedding.provider` 从 hash 换成 ollama/bge-m3 等真实语义向量模型。

### 生成错误（答案缺关键信息）（0 条）

本次评测未发现该类失败。

### 引用错误（引用幻觉）（0 条）

本次评测未发现该类失败。

### Agent 决策错误（工具选错/死循环/超轮次）（0 条）

本次评测未发现该类失败。

（共展示 3 个具体样例；完整清单见 `reports/性能评测数据.json` 与逐条明细表。）

## 三、根因归纳

| 根因类别 | 出现次数 | 占比 |
| --- | ---: | ---: |
| 检索层未召回相关片段 | 13 | 100.0% |

### 检索层整体表现（主口径：严格 Hit@5 / MRR / Recall@K / NDCG@K）

- 档位：`hybrid`　Top-K：5　判定方式：`text`　样本数：40
- Hit@K **0.7000**　MRR **0.5883**　Recall@K **0.6375**　NDCG@K **0.5827**
- 平均延迟 **0.800 ms**　P95 延迟 **1.020 ms**

```text
Hit@K     ██████████████░░░░░░ 0.700
MRR       ████████████░░░░░░░░ 0.588
Recall@K  █████████████░░░░░░░ 0.637
NDCG@K    ████████████░░░░░░░░ 0.583
```

## 四、一轮迭代优化的前后对比（真实对照实验）

实验设计：同一份样例语料（Transformer / ResNet / BERT 三篇），按问题类型分层抽取 61 条评测样本，两组仅参数不同，各自建立独立临时索引，指标由同一套代码计算（严格口径：只用真实返回的前 5 条 + 金标准文本精确归属的 chunk_id）。

**对照组（优化前）**：

```yaml
{
  "retrieval": {
    "candidate_k": 20,
    "enable_rerank": true,
    "top_k": 5
  },
  "chunking": {
    "chunk_size": 512,
    "chunk_overlap": 64
  }
}
```

**实验组（优化后）**：

```yaml
{
  "retrieval": {
    "candidate_k": 40,
    "enable_rerank": false,
    "top_k": 5
  },
  "chunking": {
    "chunk_size": 384,
    "chunk_overlap": 48
  },
  "generation": {
    "max_context_chars": 8000
  }
}
```

| 指标（严格口径） | 优化前 | 优化后 | 变化 | 相对变化 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Hit@5 | 0.7705 | 0.8033 | ↑ +0.0328 | +4.26% | 提升 |
| MRR | 0.6328 | 0.6503 | ↑ +0.0175 | +2.77% | 提升 |
| Recall@5 | 0.6858 | 0.7131 | ↑ +0.0273 | +3.98% | 提升 |
| NDCG@5 | 0.6238 | 0.6359 | ↑ +0.0121 | +1.94% | 提升 |
| 平均端到端延迟(ms) | 1.7767 | 1.5925 | ↓ -0.1842 | -10.37% | 提升 |
| 降级率 | 0.1639 | 0.0328 | ↓ -0.1311 | -79.99% | 提升 |
| 引用准确率 | 1.0 | 1.0 | → +0.0000 | +0.00% | 持平 |

```text
Hit@5     优化前 ███████████████░░░░░ 0.770
          优化后 ████████████████░░░░ 0.803
MRR       优化前 █████████████░░░░░░░ 0.633
          优化后 █████████████░░░░░░░ 0.650
Recall@5  优化前 ██████████████░░░░░░ 0.686
          优化后 ██████████████░░░░░░ 0.713
NDCG@5    优化前 ████████████░░░░░░░░ 0.624
          优化后 █████████████░░░░░░░ 0.636
```

语料规模：优化前 27 个分块（chunk_size=512），优化后 37 个分块（chunk_size=384）；引擎口径 Hit@K 0.8525 → 0.8361（辅助参考，该口径实际度量 Top-10 命中）。

### 4.1 单参数消融（收益归因）

「改了三个参数后变好了」无法指导下一步动哪个参数。因此额外跑了三组**单参数消融**：每组只在对照组基础上改动一个参数，其余保持一致。

| 配置 | 分块数 | 严格 Hit@5 | 严格 MRR | 严格 Recall@5 | 严格 NDCG@5 | 端到端延迟(ms) | 检索延迟(ms) | 降级率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 对照组（config.yaml 默认） | 27 | 0.7705 | 0.6328 | 0.6858 | 0.6238 | 1.777 | 1.010 | 0.1639 |
| 仅关闭启发式重排序（enable_rerank=False） | 27 | 0.8033 | 0.6432 | 0.7104 | 0.6359 | 1.746 | 0.868 | 0.0328 |
| 仅调细分块（chunk_size 512→384） | 37 | 0.7869 | 0.6104 | 0.6995 | 0.6117 | 1.983 | 1.182 | 0.1639 |
| 仅扩大候选池（candidate_k 20→40） | 27 | 0.7705 | 0.6328 | 0.6858 | 0.6238 | 1.946 | 1.147 | 0.1639 |
| **优化后（三者合并）** | 37 | 0.8033 | 0.6503 | 0.7131 | 0.6359 | 1.593 | 0.824 | 0.0328 |

**归因结论**：严格 NDCG@5 提升最明显的单参数是「仅关闭启发式重排序（enable_rerank=False）」（NDCG@5 +0.0121，Hit@5 +0.0328，降级率 -0.1311）。

同时注意以下单参数改动**降低了**严格 NDCG@5（说明参数之间存在交互，不能拆开单独使用）：
- 仅调细分块（chunk_size 512→384）：NDCG@5 -0.0121，Hit@5 +0.0164，降级率 +0.0000

### 4.2 结论与取舍

扩大候选池能为未来接入真实语义向量预留召回空间，代价是检索延迟上升；把 `chunk_size` 调小能提升片段与「金标准短句」的粒度匹配度，但会切碎长句、增加分块总数与索引成本；关闭启发式重排序省下了排序开销，但在个别问题上会失去它的排序收益。因此参数选择应结合延迟预算与**单参数消融表**做权衡，而不是单纯追求 Hit@K 最大化。
实验实际耗时：2.5 秒（含对照 + 优化 + 3 组单参数消融，共 5 次完整评测）。

## 五、后续优化路线

| 优先级 | 措施 | 预期收益 | 代价 |
| --- | --- | --- | --- |
| P0 | 引用后处理校验：来源不在检索结果中即标记为「未核实」 | 消除引用幻觉 | 需改生成层（模块二） |
| P0 | 复核 `retrieval.enable_rerank` 的默认值 | 见第四章消融表：关闭重排序对严格指标的影响 | 可能损失它在该问题上的排序收益 |
| P1 | 更换 Embedding 为真实语义向量（bge-m3 / ollama） | 中文提问检索英文语料的关键瓶颈 | 需本地模型或 API |
| P1 | 修复重排序器的并列分数问题，使其 `confidence` 具备区分度 | 让 full 档真正优于 hybrid 档 | 需改检索层（模块一） |
| P2 | 提示词约束「不得推断原文未给出的内容」 | 降低 generation_error | 可能降低答案丰富度 |
| P2 | 工具描述补充「不适用场景」 | 提升工具选择准确率 | 提示词变长 |

---

_本报告由 `src/evaluation/badcase.py` 自动生成；所有指标均来自实际运行，未运行的项已显式标注。_