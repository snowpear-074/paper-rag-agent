"""模块三 —— Agent 决策层。

    from src.agent import create_agent

    agent = create_agent(retrieval_engine=engine)
    result = agent.run("Transformer 用了什么数据集？", session_id="sess_1")
    for step in result.steps:
        print(step.render())

包含：手写 ReAct 循环、8 个工具、意图路由、错误恢复、并行调用、对话记忆。
"""

from .base_tool import (
    PYTHON_TYPE_MAP,
    BaseTool,
    ToolError,
    ToolParameter,
    ToolRegistry,
    tool_from_function,
)
from .memory import (
    ConversationMemory,
    MemoryConfig,
    NullMemory,
    create_memory,
)
from .prompts import (
    FINAL_ANSWER_PROMPT,
    REACT_SYSTEM_PROMPT,
    ROUTE_PROMPT,
    build_final_answer_prompt,
    build_react_system_prompt,
    build_react_user_prompt,
    build_route_prompt,
    render_scratchpad,
)
from .react_loop import AgentConfig, ReActAgent, create_agent
from .router import LLMRouter, NullRouter, RouteDecision, RuleRouter, create_router
from .tools import (
    CalculatorTool,
    CurrentTimeTool,
    KeywordExtractTool,
    KnowledgeSearchTool,
    PaperCompareTool,
    PaperMetadataTool,
    SummarizePaperTool,
    WebSearchTool,
    build_default_registry,
)

__all__ = [
    # 核心
    "ReActAgent", "AgentConfig", "create_agent",
    # 工具框架
    "BaseTool", "ToolParameter", "ToolRegistry", "ToolError",
    "tool_from_function", "PYTHON_TYPE_MAP",
    # 八个工具
    "KnowledgeSearchTool", "PaperMetadataTool", "PaperCompareTool",
    "KeywordExtractTool", "SummarizePaperTool", "CurrentTimeTool",
    "CalculatorTool", "WebSearchTool", "build_default_registry",
    # 路由
    "RuleRouter", "LLMRouter", "NullRouter", "RouteDecision", "create_router",
    # 记忆
    "ConversationMemory", "MemoryConfig", "NullMemory", "create_memory",
    # 提示词
    "REACT_SYSTEM_PROMPT", "ROUTE_PROMPT", "FINAL_ANSWER_PROMPT",
    "build_react_system_prompt", "build_react_user_prompt",
    "build_route_prompt", "build_final_answer_prompt", "render_scratchpad",
]
