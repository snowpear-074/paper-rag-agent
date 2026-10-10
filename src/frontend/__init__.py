"""模块四 —— 系统集成与前端展示。

    from src.frontend import get_container

    container = get_container()                 # 装配全系统
    container.index_files(["./data/samples"])   # 导入论文
    print(container.ask("Transformer 用了什么数据集？")["answer"])

包含：全链路容器（RAG+Agent 深度融合）、零依赖 HTTP 服务（REST + SSE）、
静态 Web 界面、命令行入口。
"""

from .container import SystemContainer, get_container, reset_container

__all__ = ["SystemContainer", "get_container", "reset_container"]

__version__ = "1.0.0"
