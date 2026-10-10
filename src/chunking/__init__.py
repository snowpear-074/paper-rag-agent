"""模块一 —— 文本分块策略。

    from src.chunking import create_chunker, chunk_documents

支持 fixed / recursive / semantic / paper 四种策略，
用于《分块策略对比实验报告》的横向对比。
"""

from .strategies import (
    CHUNKER_REGISTRY,
    STRATEGIES,
    BaseChunker,
    ChunkingConfig,
    FixedSizeChunker,
    PaperChunker,
    RecursiveChunker,
    SemanticChunker,
    chunk_document,
    chunk_documents,
    create_chunker,
)

__all__ = [
    "BaseChunker", "ChunkingConfig",
    "FixedSizeChunker", "RecursiveChunker", "SemanticChunker", "PaperChunker",
    "CHUNKER_REGISTRY", "STRATEGIES",
    "create_chunker", "chunk_document", "chunk_documents",
]
