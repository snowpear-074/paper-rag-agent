"""模块二 —— RAG 生成层。

    from src.generation import create_rag_pipeline

    pipeline = create_rag_pipeline()
    result = pipeline.answer("Transformer 用了什么数据集？")
    for event in pipeline.answer_stream("BERT 的预训练任务是什么？"):
        ...  # token / citation / final 事件

包含：Prompt 工程、生成流水线、流式输出、语义缓存、降级策略、可观测性。
"""

from .cache import CacheEntry, NullCache, SemanticCache, create_cache
from .prompt_template import (
    CITATION_PATTERN,
    MEMORY_SUMMARY_PROMPT,
    NO_CONTEXT_PROMPT,
    PROMPT_REGISTRY,
    RAG_QA_PROMPT,
    RAG_STRICT_PROMPT,
    RAG_SUMMARY_PROMPT,
    PromptTemplate,
    build_citations,
    build_context,
    format_context_block,
    get_prompt,
    parse_citations,
    validate_citation_coverage,
)
from .rag_pipeline import RAGConfig, RAGPipeline, create_rag_pipeline
from .streaming import (
    EventType,
    StreamBuffer,
    StreamEvent,
    collect_stream,
    stream_events,
    stream_text,
)

__all__ = [
    "RAGPipeline", "RAGConfig", "create_rag_pipeline",
    "SemanticCache", "CacheEntry", "NullCache", "create_cache",
    "PromptTemplate", "PROMPT_REGISTRY", "get_prompt",
    "RAG_QA_PROMPT", "RAG_STRICT_PROMPT", "RAG_SUMMARY_PROMPT",
    "NO_CONTEXT_PROMPT", "MEMORY_SUMMARY_PROMPT",
    "build_context", "build_citations", "format_context_block",
    "parse_citations", "validate_citation_coverage", "CITATION_PATTERN",
    "StreamEvent", "EventType", "StreamBuffer",
    "stream_events", "stream_text", "collect_stream",
]
