# -*- coding: utf-8 -*-
"""测试包。

运行全部测试::

    python -m unittest discover -s tests -v

各测试文件与模块的对应关系：

======================  ============================================
测试文件                 覆盖模块
======================  ============================================
test_utils.py           基础设施（配置/文本/类型/存储/日志）
test_data_loader.py     模块一：多格式文档加载
test_chunking.py        模块一：分块策略
test_retrieval.py       模块一：向量库/BM25/RRF/重排序/检索引擎
test_generation.py      模块二：Prompt/流式/缓存/降级
test_agent.py           模块三：工具/ReAct/路由/记忆
test_frontend.py        模块四：容器/HTTP API/SSE
test_frontend_static.py 模块四：前端静态资源与契约一致性
test_evaluation.py      模块五：评测指标与评测集
======================  ============================================
"""
