"""模块一 —— 检索层。

    from src.retrieval import create_retrieval_engine

    engine = create_retrieval_engine()
    engine.index_documents(docs)
    hits = engine.retrieve("Transformer 用了什么数据集？", mode="full")

包含：向量库（numpy/chroma/faiss）、BM25、混合检索（RRF）、重排序。
"""

from .bm25_retriever import BM25Okapi, BM25Retriever
from .engine import EvalMetrics, IndexReport, RetrievalEngine, create_retrieval_engine
from .hybrid_retriever import HybridRetriever, RetrievalTrace, rrf_fuse
from .reranker import (
    RERANKER_PROVIDERS,
    BaseReranker,
    HeuristicReranker,
    LocalReranker,
    OllamaReranker,
    create_reranker,
)
from .vector_store import (
    VECTOR_STORE_BACKENDS,
    BaseVectorStore,
    ChromaVectorStore,
    FaissVectorStore,
    NumpyVectorStore,
    create_vector_store,
)

__all__ = [
    "RetrievalEngine", "create_retrieval_engine", "IndexReport", "EvalMetrics",
    "HybridRetriever", "rrf_fuse", "RetrievalTrace",
    "BM25Retriever", "BM25Okapi",
    "BaseReranker", "HeuristicReranker", "OllamaReranker", "LocalReranker",
    "create_reranker", "RERANKER_PROVIDERS",
    "BaseVectorStore", "NumpyVectorStore", "ChromaVectorStore", "FaissVectorStore",
    "create_vector_store", "VECTOR_STORE_BACKENDS",
]
