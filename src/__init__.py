"""智能科研助理 —— 基于 RAG + Agent 的论文知识库问答系统。

顶层包。子模块对应课程实践的五个模块：

- ``src.data_loader``  模块一：多格式文档加载
- ``src.chunking``     模块一：文本分块策略
- ``src.retrieval``    模块一：向量存储 / BM25 / 混合检索 / 重排序
- ``src.generation``   模块二：RAG 生成层
- ``src.agent``        模块三：Agent 决策层
- ``src.frontend``     模块四：Web 前端与系统集成
- ``src.evaluation``   模块五：评测与 Bad Case 分析
- ``src.utils``        基础设施：配置 / 日志 / 数据结构 / 文本处理
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
