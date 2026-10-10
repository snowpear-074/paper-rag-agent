# ==============================================================================
# 智能科研助理 —— Docker 镜像定义
#
# 特点：
#   • 多阶段构建，最终镜像不含编译工具链
#   • 默认走「零依赖」运行模式（仅需 numpy），镜像体积小、无需联网拉模型
#   • 通过环境变量 APP_LLM__PROVIDER 可在运行时切换到 Ollama / OpenAI 兼容接口
#   • 以非 root 用户运行，挂载卷持久化知识库索引与数据
#
# 构建：  docker build -f docker/Dockerfile -t research-assistant:1.0.0 .
# 运行：  docker run --rm -p 8501:8501 -v %cd%/data:/app/data research-assistant:1.0.0
# ==============================================================================

# ------------------------------------------------------------------------------
# 阶段一：依赖安装
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# 只装必需依赖（numpy）。可选增强依赖按需在下方追加，例如：
#   RUN pip install --prefix=/install pypdf python-docx
COPY requirements.txt .
RUN grep -v '^\s*#' requirements.txt | grep -v '^\s*$' > req.txt \
    && pip install --prefix=/install -r req.txt

# ------------------------------------------------------------------------------
# 阶段二：运行时
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="智能科研助理" \
      org.opencontainers.image.description="基于 RAG + Agent 的论文知识库问答系统" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai \
    # 容器内必须监听 0.0.0.0，否则宿主无法访问
    APP_SERVER__HOST=0.0.0.0 \
    APP_SERVER__PORT=8501 \
    # 在线模式默认：DeepSeek 对话 + 必应联网搜索
    # ⚠️ 密钥不写在这里 —— 由 docker-compose 从宿主机环境变量注入（APP_LLM__API_KEY）
    APP_LLM__PROVIDER=openai \
    APP_LLM__MODEL=deepseek-chat \
    APP_LLM__BASE_URL=https://api.deepseek.com \
    APP_EMBEDDING__PROVIDER=hash \
    APP_WEB_SEARCH__ENABLED=true \
    APP_WEB_SEARCH__PROVIDER=auto

COPY --from=builder /install /usr/local

# 以非 root 用户运行（安全基线）
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# 只复制运行所需内容（.dockerignore 已排除测试/缓存）
COPY --chown=appuser:appuser config.yaml run.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser tests/fixtures/ ./tests/fixtures/
COPY --chown=appuser:appuser docs/ ./docs/

# 数据目录（挂载卷覆盖以持久化）
RUN mkdir -p /app/data /app/logs /app/reports \
    && chown -R appuser:appuser /app/data /app/logs /app/reports

USER appuser

EXPOSE 8501

# 健康检查：直接探测 /api/health（无需 curl，用 Python 标准库）
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=5).status==200 else 1)"

# 入口：环境变量可覆盖主机/端口等配置
ENTRYPOINT ["python", "run.py"]
