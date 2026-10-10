/* ============================================================================
 * app.js —— 智能科研助理前端主程序（原生 ES2020+，零依赖）
 *
 * 模块划分
 * --------
 *   Config   —— 常量与 UI 文案
 *   Util     —— 纯函数工具（格式化 / DOM 助手）
 *   Store    —— 全局状态与 localStorage 持久化
 *   Toast    —— 右上角通知（3 秒自动消失）
 *   API      —— REST 调用封装（统一错误语义）
 *   SSE      —— fetch + ReadableStream 手工解析 text/event-stream
 *   UI       —— 主题、布局、弹窗、输入框、滚动、事件委托
 *   Sessions —— 会话管理（新建 / 切换 / 删除 / 重命名）
 *   Docs     —— 知识库（拖拽上传 / base64 / 进度 / 删除 / 重建索引）
 *   Chat     —— 对话主流程（SSE 打字机 / 引用角标 / 元信息）
 *   Trace    —— Agent 推理轨迹时间线 + 原始 JSON 视图
 *   Metrics  —— 系统指标弹窗（5 秒自动刷新）
 * ==========================================================================*/
(function () {
  'use strict';

  /* ==========================================================================
   * Config —— 常量
   * ========================================================================*/
  const Config = {
    VERSION: '1.0.0',
    STORAGE_THEME: 'sra.theme',
    STORAGE_SESSION: 'sra.session_id',
    ACCEPT_EXT: ['.pdf', '.docx', '.txt', '.md'],
    MAX_UPLOAD_MB: 40,
    HEALTH_INTERVAL: 30000,
    METRICS_INTERVAL: 5000,
    TOAST_MS: 3000,
    TRACE_PREVIEW: 220,
    // 键必须与后端注册的工具名完全一致（src/agent/tools.py 的 Tool.name），
    // 否则 Agent 轨迹里只会显示兜底图标 🧩：早期这里写的是 paper_meta /
    // compare_papers / summarize，与后端 paper_metadata / paper_compare /
    // summarize_paper 对不上，三个工具的图标一直是错的。
    TOOL_ICON: { knowledge_search: '🔍', paper_metadata: '📄', paper_compare: '⚖️',
      keyword_extract: '🏷️', summarize_paper: '📝', calculator: '🧮', current_time: '🕒', web_search: '🌐' },
    EXAMPLES: [
      { tag: '事实性', q: 'Transformer 的核心创新是什么？' },
      { tag: '对比性', q: '对比论文A和论文B的方法差异' },
      { tag: '归纳性', q: '这一方向主流的评估指标有哪些？' },
      { tag: '推理性', q: '为什么作者选择自注意力而不是循环结构？' }
    ]
  };

  /* ==========================================================================
   * Util —— 工具函数
   * ========================================================================*/
  const Util = {
    $(sel, root) { return (root || document).querySelector(sel); },
    $$(sel, root) { return Array.from((root || document).querySelectorAll(sel)); },

    /** 创建元素：el('div', {class:'x'}, ['文本', childEl]) */
    el(tag, attrs, children) {
      const node = document.createElement(tag);
      if (attrs) {
        for (const [k, v] of Object.entries(attrs)) {
          if (v === null || v === undefined || v === false) continue;
          if (k === 'class') node.className = v;
          else if (k === 'text') node.textContent = v;
          else if (k === 'html') node.innerHTML = v;
          else if (k === 'dataset') Object.assign(node.dataset, v);
          else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
          else node.setAttribute(k, v === true ? '' : String(v));
        }
      }
      for (const c of [].concat(children || [])) {
        if (c === null || c === undefined || c === false) continue;
        node.append(c instanceof Node ? c : document.createTextNode(String(c)));
      }
      return node;
    },

    /** 全文转义（Toast / 报错等非 markdown 场景使用） */
    esc(s) { return (window.Markdown && Markdown.escapeHtml) ? Markdown.escapeHtml(s) : String(s ?? ''); },

    /** 人类可读文件大小 */
    bytes(n) {
      const v = Number(n) || 0;
      if (v <= 0) return '0 B';
      const units = ['B', 'KB', 'MB', 'GB'];
      const i = Math.min(units.length - 1, Math.floor(Math.log(v) / Math.log(1024)));
      const num = v / Math.pow(1024, i);
      return `${num >= 100 || i === 0 ? Math.round(num) : num.toFixed(1)} ${units[i]}`;
    },

    /** 耗时格式化 */
    ms(n) {
      const v = Number(n);
      if (!isFinite(v) || v < 0) return '—';
      return v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
    },

    /** 紧凑数字（8000 → 8.0k） */
    num(n, digits = 0) {
      const v = Number(n);
      if (!isFinite(v)) return '—';
      return v >= 10000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(digits).replace(/\.0+$/, '');
    },

    /** 百分比 */
    pct(v, digits = 1) {
      const n = Number(v);
      return isFinite(n) ? `${(n * 100).toFixed(digits)}%` : '—';
    },

    /** 秒级时间戳 → 本地时间字符串 */
    time(ts, withDate = false) {
      const d = ts ? new Date(Number(ts) * 1000) : new Date();
      if (isNaN(d.getTime())) return '—';
      const p = (x) => String(x).padStart(2, '0');
      const hm = `${p(d.getHours())}:${p(d.getMinutes())}`;
      return withDate ? `${d.getMonth() + 1}-${p(d.getDate())} ${hm}` : hm;
    },

    relTime(ts) {
      const diff = Date.now() / 1000 - (Number(ts) || 0);
      if (!isFinite(diff) || diff < 0) return Util.time(ts);
      if (diff < 60) return '刚刚';
      if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
      if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
      if (diff < 604800) return `${Math.floor(diff / 86400)} 天前`;
      return Util.time(ts, true);
    },

    /** 安全 JSON 字符串化（处理循环引用） */
    json(v, space = 2) {
      try { return JSON.stringify(v, null, space); } catch (_) { return String(v); }
    },

    /** 截断文本 */
    cut(s, n, tail = '…') {
      const str = String(s ?? '');
      return str.length > n ? str.slice(0, n) + tail : str;
    },

    /** 生成会话短标题 */
    autoTitle(question) {
      const t = (window.Markdown ? Markdown.toPlainText(question, 24) : Util.cut(question, 24)) || '新会话';
      return t.replace(/\s+/g, ' ').trim() || '新会话';
    },

    /** 文件扩展名 */
    ext(name) {
      const m = /\.([A-Za-z0-9]+)$/.exec(String(name || ''));
      return m ? m[1].toLowerCase() : '';
    },

    uid(prefix = 'ui') { return `${prefix}_${Math.random().toString(36).slice(2, 9)}`; }
  };

  /* ==========================================================================
   * Store —— 全局状态
   * ========================================================================*/
  const Store = {
    state: {
      sessionId: null,
      sessionTitle: '新会话',
      sessions: [],
      messages: [],          // 当前会话前端渲染所需的数据快照
      traces: [],            // 轨迹步骤（对象数组）
      traceJsonMode: false,
      docs: [],
      docSummary: { num_documents: 0, num_chunks: 0, num_vectors: 0 },
      health: null,
      streaming: false,
      abort: null,
      uploads: new Map()     // fileKey -> { name, size, percent, message, state }
    },

    load() {
      try { this.state.sessionId = localStorage.getItem(Config.STORAGE_SESSION) || null; } catch (_) { /* 忽略 */ }
    },
    saveSession(id) {
      this.state.sessionId = id;
      try { id ? localStorage.setItem(Config.STORAGE_SESSION, id) : localStorage.removeItem(Config.STORAGE_SESSION); } catch (_) { /* 忽略 */ }
    }
  };

  /* ==========================================================================
   * Toast —— 右上角通知
   * ========================================================================*/
  const Toast = {
    ICON: { success: '✔', error: '✕', warn: '!', info: 'ⓘ' },
    show(message, type = 'info', ms = Config.TOAST_MS) {
      const stack = Util.$('#toast-stack');
      if (!stack) return;
      const node = Util.el('div', { class: `toast ${type}` }, [
        Util.el('span', { class: 't-ico', text: this.ICON[type] || this.ICON.info }),
        Util.el('div', { class: 't-msg', text: String(message ?? '') })
      ]);
      stack.append(node);
      const close = () => {
        node.classList.add('out');
        setTimeout(() => node.remove(), 220);
      };
      node.addEventListener('click', close);
      setTimeout(close, ms);
    },
    success(m) { this.show(m, 'success'); },
    error(m) { this.show(m, 'error', 4200); },
    warn(m) { this.show(m, 'warn'); },
    info(m) { this.show(m, 'info'); }
  };

  /* ==========================================================================
   * API —— REST 封装
   * ========================================================================*/
  const API = {
    health() { return this.get('/api/health'); },
    sessions() { return this.get('/api/sessions'); },
    createSession(title) { return this.post('/api/sessions', { title: title || '新会话' }); },
    session(id) { return this.get(`/api/sessions/${encodeURIComponent(id)}`); },
    deleteSession(id) { return this.del(`/api/sessions/${encodeURIComponent(id)}`); },
    documents() { return this.get('/api/documents'); },
    deleteDocument(docId) { return this.del(`/api/documents/${encodeURIComponent(docId)}`); },
    reindex(docId) { return this.post('/api/documents/reindex', docId ? { doc_id: docId } : {}); },
    metrics() { return this.get('/api/metrics'); },
    tools() { return this.get('/api/tools'); },
    traces(n = 20) { return this.get(`/api/traces?n=${encodeURIComponent(n)}`); },
    config() { return this.get('/api/config'); },

    async request(method, url, body, { timeout = 30000, raw = false } = {}) {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), timeout);
      try {
        const res = await fetch(url, {
          method,
          headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: ctrl.signal
        });
        if (raw) return res;
        const text = await res.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch (_) { data = null; }
        if (!res.ok) throw new Error((data && (data.error || data.message)) || `HTTP ${res.status} ${res.statusText}`);
        if (data && data.ok === false) throw new Error(data.error || '服务返回 ok=false');
        return data;
      } catch (err) {
        if (err && err.name === 'AbortError') throw new Error('请求超时或已被取消');
        throw err;
      } finally {
        clearTimeout(timer);
      }
    },
    post(url, body) { return this.request('POST', url, body === undefined ? {} : body); },
    get(url) { return this.request('GET', url); },
    del(url) { return this.request('DELETE', url); }
  };

  /* ==========================================================================
   * SSE —— fetch + ReadableStream 手工解析（EventSource 不支持 POST）
   * ========================================================================*/
  const SSE = {
    /** 聊天流事件名（与后端契约一一对应） */
    CHAT_EVENTS: ['start', 'route', 'retrieval', 'thought', 'action',
                  'observation', 'token', 'citation', 'final', 'error', 'done'],
    /** 上传流事件名 */
    UPLOAD_EVENTS: ['progress', 'indexed', 'error'],

    /**
     * 打开一条 SSE 流并逐事件回调。
     * @returns {Promise<{aborted:boolean, status:number}>}
     */
    async open(url, body, { onEvent, signal, onOpen } = {}) {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify(body || {}),
        signal
      });
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
          const t = await res.text();
          const j = JSON.parse(t);
          detail = (j && (j.error || j.message)) || detail;
        } catch (_) { /* 保持 HTTP 状态描述 */ }
        throw new Error(detail);
      }
      if (onOpen) onOpen(res);
      if (!res.body || !res.body.getReader) throw new Error('当前浏览器不支持流式响应');

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let aborted = false;

      const dispatch = (chunk) => {
        if (!chunk) return;
        let event = 'message';
        const dataLines = [];
        for (const rawLine of chunk.split('\n')) {
          const line = rawLine.replace(/\r$/, '');
          if (!line || line.startsWith(':')) continue;
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
        }
        if (!dataLines.length) return;
        let payload = null;
        const data = dataLines.join('\n');
        try { payload = JSON.parse(data); } catch (_) { payload = { raw: data }; }
        if (onEvent) onEvent(event, payload);
      };

      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop() || '';
          for (const part of parts) dispatch(part.trim());
        }
        buffer += decoder.decode();
        if (buffer.trim()) dispatch(buffer.trim());
      } catch (err) {
        if (err && err.name === 'AbortError') aborted = true;
        else throw err;
      }
      return { aborted, status: res.status };
    }
  };

  /* ==========================================================================
   * UI —— 主题 / 布局 / 弹窗 / 输入框
   * ========================================================================*/
  const UI = {
    init() {
      this.initTheme();
      this.bindLayout();
      this.bindModals();
      this.bindComposer();
      this.bindDelegates();
      this.bindTracePanel();
      this.applyTextareaHeight();
    },

    /* ---------- 主题 ---------- */
    initTheme() {
      let saved = null;
      try { saved = localStorage.getItem(Config.STORAGE_THEME); } catch (_) { /* 忽略 */ }
      this.applyTheme(saved || 'auto');
      Util.$('#btn-theme')?.addEventListener('click', () => {
        const cur = document.documentElement.dataset.theme || 'auto';
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const effectiveDark = cur === 'dark' || (cur === 'auto' && prefersDark);
        this.applyTheme(effectiveDark ? 'light' : 'dark');
        Toast.info(effectiveDark ? '已切换到浅色主题' : '已切换到深色主题');
      });
    },
    applyTheme(mode) {
      document.documentElement.dataset.theme = mode;
      try { localStorage.setItem(Config.STORAGE_THEME, mode); } catch (_) { /* 忽略 */ }
    },

    /* ---------- 布局：窄屏轨迹抽屉 ---------- */
    bindLayout() {
      const panel = Util.$('#trace-panel');
      const toggle = Util.$('#btn-trace-toggle');
      toggle?.addEventListener('click', () => panel?.classList.toggle('open'));
      document.addEventListener('click', (e) => {
        if (window.innerWidth > 1200 || !panel?.classList.contains('open')) return;
        if (panel.contains(e.target) || toggle?.contains(e.target)) return;
        panel.classList.remove('open');
      });
    },

    /* ---------- 弹窗 ---------- */
    bindModals() {
      const openMetrics = () => { Metrics.open(); };
      Util.$('#btn-metrics')?.addEventListener('click', openMetrics);
      Util.$('#btn-metrics-close')?.addEventListener('click', () => Metrics.close());
      Util.$('#btn-metrics-refresh')?.addEventListener('click', () => Metrics.load(true));
      Util.$('#metrics-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'metrics-modal') Metrics.close();
      });

      Util.$('#btn-tools')?.addEventListener('click', () => this.openTools());
      Util.$('#btn-tools-close')?.addEventListener('click', () => this.closeTools());
      Util.$('#tools-modal')?.addEventListener('click', (e) => {
        if (e.target.id === 'tools-modal') this.closeTools();
      });

      document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (!Util.$('#metrics-modal')?.hidden) Metrics.close();
        if (!Util.$('#tools-modal')?.hidden) this.closeTools();
      });
    },

    async openTools() {
      const modal = Util.$('#tools-modal');
      const body = Util.$('#tools-body');
      body.innerHTML = '<p class="empty-hint">加载中…</p>';
      modal.hidden = false;
      try {
        const data = await API.tools();
        const tools = data?.tools || [];
        body.replaceChildren();
        if (!tools.length) { body.append(Util.el('p', { class: 'empty-hint', text: '暂无已注册工具' })); return; }
        for (const t of tools) {
          const params = t?.parameters || {};
          const names = Object.keys(params.properties || {});
          body.append(Util.el('div', { class: 'tool-card' }, [
            Util.el('div', { class: 'tc-name', text: `${Config.TOOL_ICON[t.name] || '🧩'} ${t.name || '未命名工具'}` }),
            Util.el('div', { class: 'tc-desc', text: t.description || '（无描述）' }),
            names.length
              ? Util.el('pre', { text: Util.json({ parameters: names, required: params.required || [] }, 2) })
              : null
          ]));
        }
      } catch (err) {
        body.innerHTML = `<p class="empty-hint">加载失败：${Util.esc(err.message)}</p>`;
        Toast.error(`工具列表加载失败：${err.message}`);
      }
    },
    closeTools() { const m = Util.$('#tools-modal'); if (m) m.hidden = true; },

    /** 绑定点击：on('#id', fn) */
    on(selector, fn) { Util.$(selector)?.addEventListener('click', fn); },

    /* ---------- 输入框：自适应高度 / Enter 发送 ---------- */
    bindComposer() {
      const input = Util.$('#message-input');
      if (!input) return;
      input.addEventListener('input', () => this.applyTextareaHeight());
      input.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
        e.preventDefault();
        Chat.send();
      });
      this.on('#btn-send', () => Chat.send());
      this.on('#btn-stop', () => Chat.stop());
    },
    applyTextareaHeight() {
      const input = Util.$('#message-input');
      if (!input) return;
      input.style.height = 'auto';
      input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    },

    /* ---------- 事件委托：示例问题 / 引用跳转 / 代码复制 / 引用卡片展开 ---------- */
    bindDelegates() {
      document.addEventListener('click', (e) => {
        const example = e.target.closest('.example-btn');
        if (example) {
          const input = Util.$('#message-input');
          input.value = example.dataset.example || example.textContent.trim();
          this.applyTextareaHeight();
          input.focus();
          Chat.send();
          return;
        }
        const ref = e.target.closest('sup.citation-ref');
        if (ref) { Chat.focusCitation(ref); return; }
        const copy = e.target.closest('.cb-copy');
        if (copy) { this.copyCode(copy); return; }
        const more = e.target.closest('.c-more');
        if (more) {
          const card = more.closest('.citation-card');
          if (!card) return;
          const opened = card.classList.toggle('open');
          more.textContent = opened ? '收起 ▴' : '展开全文 ▾';
        }
      });

      this.on('#btn-refresh-docs', () => Docs.load(true));
      this.on('#btn-new-session', () => Sessions.create());
      this.on('#btn-rename-session', () => Sessions.rename());
      this.on('#btn-delete-session', () => Sessions.remove());
      this.on('#btn-trace-json', () => Trace.toggleJson());
      this.on('#btn-trace-clear', () => Trace.clear());
      this.on('#health-pill', () => Health.refresh(true));
    },

    async copyCode(btn) {
      const text = btn.closest('.code-block')?.querySelector('code')?.textContent || '';
      try {
        if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
        else {
          const ta = Util.el('textarea', { class: 'visually-hidden' });
          ta.value = text; document.body.append(ta); ta.select();
          document.execCommand('copy'); ta.remove();
        }
        btn.textContent = '已复制';
        btn.classList.add('copied');
        setTimeout(() => { btn.textContent = '复制'; btn.classList.remove('copied'); }, 1400);
      } catch (_) { Toast.warn('复制失败，请手动选择代码'); }
    },

    /* ---------- 轨迹面板（窄屏抽屉按钮注入） ---------- */
    bindTracePanel() {
      if (!Util.$('#btn-trace-toggle')) {
        const btn = Util.el('button', {
          class: 'mini-btn ghost trace-toggle', id: 'btn-trace-toggle',
          type: 'button', title: '查看 Agent 推理轨迹', text: '轨迹'
        });
        Util.$('.chat-head-actions')?.prepend(btn);
      }
    },

    scrollMessages(force = false) {
      const box = Util.$('#messages');
      if (!box) return;
      const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 140;
      if (force || nearBottom) box.scrollTop = box.scrollHeight;
    },
    scrollTrace() {
      const box = Util.$('.trace-body');
      if (box) box.scrollTop = box.scrollHeight;
    },

    setStreaming(on) {
      Store.state.streaming = on;
      const send = Util.$('#btn-send');
      const stop = Util.$('#btn-stop');
      const input = Util.$('#message-input');
      if (send) send.disabled = on;
      if (stop) stop.hidden = !on;
      if (input) input.disabled = false;
      Util.$('#composer-hint').textContent = on ? 'Agent 正在推理，可点击「停止生成」中断' : '提示：回答基于本地知识库，引用可点击跳转';
    }
  };

  /* ==========================================================================
   * Health —— 顶部健康状态
   * ========================================================================*/
  const Health = {
    timer: null,
    start() {
      this.refresh();
      clearInterval(this.timer);
      this.timer = setInterval(() => this.refresh(), Config.HEALTH_INTERVAL);
    },
    async refresh(notify = false) {
      const dot = Util.$('#health-dot');
      const label = Util.$('#health-label');
      try {
        const data = await API.health();
        Store.state.health = data;
        const llm = data?.llm || {};
        const emb = data?.embedding || {};
        const vs = data?.vector_store || {};
        const llmOk = this.okFlag(llm.status ?? llm.ok ?? llm.available);
        const embOk = this.okFlag(emb.status ?? emb.ok ?? emb.available);
        const state = (llmOk && embOk) ? 'ok' : (llmOk || embOk ? 'warn' : 'bad');
        if (dot) dot.dataset.state = state;
        if (label) {
          label.textContent = `${llm.provider || 'llm'}·${llm.model || '未知模型'} · ${data?.num_documents ?? 0} 篇 / ${data?.num_chunks ?? 0} 块`;
        }
        const pill = Util.$('#health-pill');
        if (pill) {
          pill.title = [
            `服务状态：${state === 'ok' ? '正常' : state === 'warn' ? '部分异常' : '异常'}`,
            `LLM：${llm.provider || '-'} / ${llm.model || '-'}`,
            `Embedding：${emb.provider || '-'} / ${emb.model || '-'}（${emb.dimension ?? '-'} 维）`,
            `向量库：${vs.backend || vs.provider || vs.type || '-'}（${vs.num_vectors ?? data?.num_chunks ?? 0} 向量）`,
            `运行时长：${Math.floor((data?.uptime_seconds || 0) / 60)} 分钟`,
            `版本：${data?.version || Config.VERSION}`
          ].join('\n');
        }
        if (notify) Toast.success('健康检查通过');
      } catch (err) {
        if (dot) dot.dataset.state = 'bad';
        if (label) label.textContent = '服务不可用';
        if (notify) Toast.error(`健康检查失败：${err.message}`);
      }
    },
    okFlag(v) {
      if (v === undefined || v === null) return true;
      if (typeof v === 'string') return !['down', 'error', 'fail', 'unavailable', 'false'].includes(v.toLowerCase());
      return Boolean(v);
    }
  };

  /* ==========================================================================
   * Sessions —— 会话管理
   * ========================================================================*/
  const Sessions = {
    async load(autoOpen = true) {
      const box = Util.$('#session-list');
      try {
        const data = await API.sessions();
        const list = data?.sessions || [];
        Store.state.sessions = list;
        // 后端会在第一轮问答后自动给会话起标题（用第一个问题），
        // 这里把标题栏同步过来 —— 否则标题栏一直挂着"新会话"。
        const active = list.find((s) => s.session_id === Store.state.sessionId);
        if (active?.title && active.title !== Store.state.sessionTitle) {
          Store.state.sessionTitle = active.title;
          const titleNode = Util.$('#chat-session-title');
          if (titleNode) titleNode.textContent = active.title;
        }
        box.replaceChildren();
        if (!list.length) {
          box.append(Util.el('p', { class: 'empty-hint', text: '暂无会话，点击「新建」开始' }));
          if (autoOpen) await this.create(false);
          return;
        }
        for (const s of list) box.append(this.item(s));
        const exists = list.some((s) => s.session_id === Store.state.sessionId);
        if (autoOpen) await this.open(exists ? Store.state.sessionId : list[0].session_id);
      } catch (err) {
        box.innerHTML = `<p class="empty-hint">会话列表加载失败：${Util.esc(err.message)}</p>`;
        Toast.error(`会话列表加载失败：${err.message}`);
      }
    },

    item(s) {
      const title = s.title || '未命名会话';
      const node = Util.el('div', {
        class: `session-item${s.session_id === Store.state.sessionId ? ' active' : ''}`,
        dataset: { id: s.session_id },
        title: `${title}\n${s.num_turns ?? 0} 轮 · ${Util.time(s.updated_at, true)}`
      }, [
        Util.el('div', { class: 's-main' }, [
          Util.el('div', { class: 's-title', text: title }),
          Util.el('div', { class: 's-meta', text: `${s.num_turns ?? 0} 轮 · ${Util.relTime(s.updated_at)}` })]),
        Util.el('div', { class: 's-actions' }, [
          Util.el('button', { class: 's-act', type: 'button', title: '重命名', text: '✎',
            onclick: (e) => { e.stopPropagation(); this.rename(s.session_id, s.title); } }),
          Util.el('button', { class: 's-act del', type: 'button', title: '删除', text: '🗑',
            onclick: (e) => { e.stopPropagation(); this.remove(s.session_id); } })])
      ]);
      node.addEventListener('click', () => this.open(s.session_id));
      return node;
    },

    async create(silent = true) {
      try {
        const data = await API.createSession('新会话');
        const s = data?.session;
        if (!s) throw new Error('服务未返回会话对象');
        Store.state.sessionId = s.session_id;
        // 标题/ID 也要跟着换：只改 sessionId 的话，新建会话后标题栏
        // 还挂着上一个会话的名字，看起来像"没切过去"。
        Store.state.sessionTitle = s.title || '新会话';
        Store.saveSession(s.session_id);
        Util.$('#chat-session-id').textContent = s.session_id;
        Chat.reset(`会话已创建 · ${s.session_id}`);
        await this.load(false);
        this.markActive();
        if (!silent) Toast.success('已新建会话');
      } catch (err) { Toast.error(`新建会话失败：${err.message}`); }
    },

    async open(id) {
      if (!id) return;
      if (Store.state.streaming) { Toast.warn('正在生成回答，请先停止生成'); return; }
      try {
        const data = await API.session(id);
        const session = data?.session;
        if (!session) throw new Error('服务未返回会话详情');
        Store.saveSession(id);
        Store.state.sessionTitle = session.title || '新会话';
        Util.$('#chat-session-title').textContent = Store.state.sessionTitle;
        Util.$('#chat-session-id').textContent = id;

        Chat.reset();
        const turns = session.turns || [];
        turns.forEach((t, i) => Chat.restoreTurn(t, i === turns.length - 1));
        if (!turns.length) Chat.showEmpty(true);
        this.markActive();
      } catch (err) { Toast.error(`会话加载失败：${err.message}`); }
    },

    markActive() {
      Util.$$('.session-item').forEach((n) => {
        n.classList.toggle('active', n.dataset.id === Store.state.sessionId);
      });
    },

    async rename(id, current) {
      const target = id || Store.state.sessionId;
      if (!target) { Toast.warn('当前没有可重命名的会话'); return; }
      const title = window.prompt('输入新的会话标题', current || Store.state.sessionTitle || '新会话');
      if (title === null) return;
      const clean = title.trim();
      if (!clean) { Toast.warn('标题不能为空'); return; }
      try {
        await API.post(`/api/sessions/${encodeURIComponent(target)}`, { title: clean });
        if (target === Store.state.sessionId) {
          Store.state.sessionTitle = clean;
          Util.$('#chat-session-title').textContent = clean;
        }
        await this.load(false);
        this.markActive();
        Toast.success('会话已重命名');
      } catch (err) { Toast.error(`重命名失败：${err.message}`); }
    },

    async remove(id) {
      const target = id || Store.state.sessionId;
      if (!target) { Toast.warn('当前没有可删除的会话'); return; }
      if (!window.confirm(`确认删除会话「${target}」？该操作不可撤销。`)) return;
      try {
        await API.deleteSession(target);
        Toast.success('会话已删除');
        if (target === Store.state.sessionId) {
          Store.saveSession(null);
          Chat.reset('已删除当前会话');
        }
        await this.load(true);
      } catch (err) { Toast.error(`删除失败：${err.message}`); }
    }
  };

  /* ==========================================================================
   * Docs —— 知识库管理
   * ========================================================================*/
  const Docs = {
    init() {
      const zone = Util.$('#dropzone');
      const input = Util.$('#file-input');
      if (!zone || !input) return;

      zone.addEventListener('click', () => input.click());
      zone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
      });
      input.addEventListener('change', () => {
        this.upload(Array.from(input.files || []));
        input.value = '';
      });
      ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => {
        e.preventDefault(); zone.classList.add('dragover');
      }));
      ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => {
        e.preventDefault();
        if (ev === 'dragleave' && zone.contains(e.relatedTarget)) return;
        zone.classList.remove('dragover');
      }));
      zone.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer?.files || []);
        this.upload(files);
      });
      // 阻止整页拖放打开文件
      ['dragover', 'drop'].forEach((ev) => window.addEventListener(ev, (e) => {
        if (!zone.contains(e.target)) e.preventDefault();
      }));
    },

    /** 批量上传：逐个文件走 SSE 进度流 */
    async upload(files) {
      const valid = [];
      for (const f of files) {
        const ext = `.${Util.ext(f.name)}`;
        if (!Config.ACCEPT_EXT.includes(ext)) { Toast.warn(`已跳过不支持的文件：${f.name}`); continue; }
        if (f.size > Config.MAX_UPLOAD_MB * 1024 * 1024) { Toast.warn(`文件过大（>${Config.MAX_UPLOAD_MB}MB）：${f.name}`); continue; }
        valid.push(f);
      }
      if (!valid.length) return;

      for (const file of valid) {
        const fileKey = `${file.name}_${file.size}_${Util.uid('up')}`;
        this.addUploadItem(fileKey, file);
        try {
          const base64 = await this.readBase64(file);
          await SSE.open('/api/documents/upload', { filename: file.name, content_base64: base64 }, {
            onEvent: (event, data) => this.onUploadEvent(fileKey, event, data),
            onOpen: () => this.patchUpload(fileKey, { message: '已连接，等待解析…', percent: 2 })
          });
        } catch (err) {
          this.patchUpload(fileKey, { state: 'error', message: `上传失败：${err.message}`, percent: 100 });
          Toast.error(`${file.name} 上传失败：${err.message}`);
        }
      }
      await this.load(true);
      await Health.refresh();
    },

    readBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = String(reader.result || '');
          resolve(result.includes(',') ? result.slice(result.indexOf(',') + 1) : result);
        };
        reader.onerror = () => reject(new Error('文件读取失败'));
        reader.readAsDataURL(file);
      });
    },

    STAGE_TEXT: {
      received: '文件已接收', parsing: '正在解析文档', chunking: '正在切分文本',
      embedding: '正在向量化', indexing: '正在写入向量库', done: '索引完成'
    },

    onUploadEvent(key, event, data) {
      if (event === 'progress') {
        this.patchUpload(key, {
          percent: Number(data?.percent) || 0,
          message: data?.message || this.STAGE_TEXT[data?.stage] || data?.stage || '处理中…'
        });
      } else if (event === 'indexed') {
        this.patchUpload(key, { state: 'done', percent: 100, message: `索引完成 · ${data?.num_chunks ?? 0} 个分块` });
        Toast.success(`${this.fileName(key)} 已完成索引（${data?.num_chunks ?? 0} 分块）`);
      } else if (event === 'error') {
        this.patchUpload(key, { state: 'error', percent: 100, message: `失败：${data?.message || data?.detail || '未知错误'}` });
        Toast.error(`${this.fileName(key)} 处理失败`);
      }
    },

    fileName(key) { return (Store.state.uploads.get(key)?.name) || '文件'; },

    addUploadItem(key, file) {
      const box = Util.$('#upload-list');
      const node = Util.el('div', {
        class: 'upload-item', dataset: { key, state: 'running' },
        title: `${file.name} · ${Util.bytes(file.size)}`
      }, [
        Util.el('div', { class: 'u-top' }, [
          Util.el('span', { class: 'u-name', text: file.name }),
          Util.el('span', { class: 'u-pct', text: '0%' })
        ]),
        Util.el('div', { class: 'progress' }, [Util.el('i')]),
        Util.el('div', { class: 'u-msg', text: '准备上传…' })
      ]);
      box.append(node);
      Store.state.uploads.set(key, { name: file.name, size: file.size, percent: 0, message: '准备上传…', state: 'running' });
    },

    patchUpload(key, patch) {
      const item = Store.state.uploads.get(key);
      if (item) Object.assign(item, patch);
      const node = Util.$(`.upload-item[data-key="${key}"]`);
      if (!node || !item) return;
      node.dataset.state = item.state;
      const pct = Util.$('.u-pct', node);
      const msg = Util.$('.u-msg', node);
      const fill = Util.$('.progress > i', node);
      if (pct) pct.textContent = `${Math.round(item.percent)}%`;
      if (msg) msg.textContent = item.message;
      if (fill) fill.style.width = `${Math.max(2, Math.min(100, item.percent))}%`;
    },

    async load(notify = false) {
      const box = Util.$('#doc-list');
      try {
        const data = await API.documents();
        const docs = data?.documents || [];
        Store.state.docs = docs;
        Store.state.docSummary = data?.summary || {
          num_documents: docs.length,
          num_chunks: docs.reduce((a, d) => a + (d.num_chunks || 0), 0),
          num_vectors: docs.reduce((a, d) => a + (d.num_vectors || 0), 0)
        };
        box.replaceChildren();
        if (!docs.length) {
          box.append(Util.el('p', { class: 'empty-hint', text: '知识库为空，先上传一篇论文吧' }));
        } else {
          // 渲染方法在本对象里叫 node（会话列表那边才叫 item）——
          // 以前这里写成 this.item，抛 "this.item is not a function"，
          // 结果侧栏知识库列表和底部统计一起空白。
          docs.forEach((d) => box.append(this.node(d)));
        }
        this.renderSummary();
        if (notify) Toast.success(`文档列表已刷新（${docs.length} 篇）`);
      } catch (err) {
        box.innerHTML = `<p class="empty-hint">文档列表加载失败：${Util.esc(err.message)}</p>`;
        Toast.error(`文档列表加载失败：${err.message}`);
      }
    },

    node(doc) {
      const statusMap = { indexed: ['ok', '已索引'], pending: ['warn', '待处理'],
        processing: ['info', '处理中'], failed: ['bad', '失败'] };
      const [cls, text] = statusMap[doc.status] || ['', doc.status || '未知'];
      const title = doc.title || '未命名文档';
      const tip = [`标题：${title}`, `作者：${(doc.authors || []).join('、') || '—'}`,
        `年份：${doc.year || '—'}　类型：${(doc.file_type || '').toUpperCase() || '—'}`,
        `来源：${doc.source_path || '—'}`, `大小：${Util.bytes(doc.file_size)}　页数：${doc.num_pages ?? '—'}`,
        `分块：${doc.num_chunks ?? 0}　向量：${doc.num_vectors ?? 0}`,
        `导入：${Util.time(doc.created_at, true)}　ID：${doc.doc_id}`].join('\n');
      const sub = Util.el('div', { class: 'd-sub' }, [
        Util.el('span', { class: `badge ${cls}`, text }),
        doc.year ? Util.el('span', { class: 'badge', text: String(doc.year) }) : null,
        Util.el('span', { class: 'badge info', text: (doc.file_type || 'file').toUpperCase() }),
        Util.el('span', { text: `${doc.num_chunks ?? 0} 块` }),
        Util.el('span', { text: Util.bytes(doc.file_size) })]);
      const main = Util.el('div', { class: 'd-main' }, [
        Util.el('div', { class: 'd-title', text: title }), sub]);
      const del = Util.el('button', { class: 'd-del', type: 'button', title: '删除文档', text: '🗑',
        onclick: (e) => { e.stopPropagation(); this.remove(doc); } });
      const tipBox = Util.el('div', { class: 'd-tip' }, [
        Util.el('div', {}, [Util.el('b', { text: '作者：' }), (doc.authors || []).join('、') || '—']),
        Util.el('div', {}, [Util.el('b', { text: '来源：' }), doc.source_path || '—']),
        Util.el('div', {}, [Util.el('b', { text: '页数/分块：' }), `${doc.num_pages ?? '—'} 页 / ${doc.num_chunks ?? 0} 块`]),
        Util.el('div', {}, [Util.el('b', { text: '向量数：' }), String(doc.num_vectors ?? 0)]),
        Util.el('div', {}, [Util.el('b', { text: '文档 ID：' }), doc.doc_id || '—'])]);
      return Util.el('div', { class: 'doc-item', dataset: { id: doc.doc_id }, title: tip },
        [main, del, tipBox]);
    },

    async remove(doc) {
      if (!window.confirm(`确认从知识库删除《${doc.title || doc.doc_id}》？\n该文档的所有向量会被一并移除。`)) return;
      const node = Util.$(`.doc-item[data-id="${doc.doc_id}"]`);
      node?.style.setProperty('opacity', '.55');
      try {
        await API.deleteDocument(doc.doc_id);
        Toast.success('文档已删除');
        await this.load();
        await Health.refresh();
      } catch (err) {
        node?.style.removeProperty('opacity');
        Toast.error(`删除失败：${err.message}`);
      }
    },

    renderSummary() {
      const s = Store.state.docSummary || {};
      const docs = s.num_documents ?? Store.state.docs.length;
      const chunks = s.num_chunks ?? 0;
      const vectors = s.num_vectors ?? 0;
      Util.$('#stat-docs').textContent = String(docs);
      Util.$('#stat-chunks').textContent = Util.num(chunks);
      Util.$('#stat-vectors').textContent = Util.num(vectors);
    }
  };

  /* ==========================================================================
   * Chat —— 对话主流程（SSE 打字机 / 引用 / 元信息）
   * ========================================================================*/
  const Chat = {
    el: {},              // 当前流式回答的 DOM 引用
    turn: null,          // 当前轮的原始事件快照

    reset(hint) {
      const box = Util.$('#messages');
      box.replaceChildren();
      this.el = {};
      this.turn = null;
      Store.state.messages = [];
      Store.state.traces = [];
      Trace.render();
      Trace.setJson([]);
      Util.$('#chat-session-title').textContent = Store.state.sessionTitle || '新会话';
      this.showEmpty(!hint);
      if (hint) Toast.info(hint);
    },

    showEmpty(show) {
      const empty = Util.$('#empty-state');
      if (empty) empty.hidden = !show;
    },

    buildEmptyState() {
      const wrap = Util.el('div', { class: 'empty-state', id: 'empty-state' }, [
        Util.el('div', { class: 'empty-logo', text: '📚' }),
        Util.el('h3', { text: '开始你的科研问答' }),
        Util.el('p', { text: '先上传论文构建知识库，然后向 Agent 提问。回答会附带引用来源与完整推理轨迹。' })
      ]);
      const examples = Util.el('div', { class: 'examples' });
      for (const ex of Config.EXAMPLES) {
        examples.append(Util.el('button', {
          class: 'example-btn', type: 'button', dataset: { example: ex.q }
        }, [Util.el('span', { class: 'ex-tag', text: ex.tag }), ex.q]));
      }
      wrap.append(examples);
      return wrap;
    },

    /* ---------- 发送 ---------- */
    async send() {
      if (Store.state.streaming) { Toast.warn('正在生成回答，请稍候或点击停止生成'); return; }
      const input = Util.$('#message-input');
      const question = (input?.value || '').trim();
      if (!question) { Toast.warn('请输入问题后再发送'); input?.focus(); return; }
      if (!Store.state.sessionId) await Sessions.create(false);
      if (!Store.state.sessionId) { Toast.error('无法创建会话，请检查后端服务'); return; }

      input.value = '';
      UI.applyTextareaHeight();
      this.showEmpty(false);
      this.appendUser(question);
      Store.state.messages.push({ role: 'user', content: question });

      const assistant = this.appendAssistant();
      Trace.begin(question);
      UI.setStreaming(true);
      Store.state.abort = new AbortController();
      this.turn = { question, tokens: '', route: null, retrieval: null, citations: [], final: null, error: null, events: [] };

      try {
        const { aborted } = await SSE.open('/api/chat', {
          question, session_id: Store.state.sessionId, stream: true
        }, {
          signal: Store.state.abort.signal,
          onEvent: (event, data) => this.onEvent(event, data, assistant)
        });
        if (aborted) this.finishInterrupted(assistant);
      } catch (err) {
        this.fail(assistant, err.message);
      } finally {
        UI.setStreaming(false);
        Store.state.abort = null;
        Trace.flush();
        UI.scrollMessages();
      }
    },

    stop() {
      if (!Store.state.streaming || !Store.state.abort) { Toast.info('当前没有正在生成的回答'); return; }
      Store.state.abort.abort();
      Toast.warn('已请求停止生成');
    },

    /* 事件 → 处理逻辑表（时间线四类 step 直接内联映射） */
    HANDLERS: {
      thought: (c, d) => Trace.push({ kind: 'thought', round: d?.round, content: d?.content }),
      action: (c, d) => Trace.push({ kind: 'action', round: d?.round, tool: d?.tool, args: d?.args,
        call_id: d?.call_id, content: `调用工具 ${d?.tool || '未知工具'}` }),
      observation: (c, d) => Trace.push({
        kind: d?.status && d.status !== 'success' ? 'error' : 'observation',
        round: d?.round, tool: d?.tool, status: d?.status,
        latency_ms: d?.latency_ms, content: d?.preview || d?.content || '' })
    },

    /* ---------- 事件分发 ---------- */
    onEvent(event, data, assistant) {
      if (this.turn) this.turn.events.push({ event, data });
      const inline = this.HANDLERS[event];
      if (inline) { inline(this, data); return; }
      const handler = this[`on${event.charAt(0).toUpperCase()}${event.slice(1)}`];      if (typeof handler === 'function') handler.call(this, data, assistant);
    },

    onStart(data) {
      if (data?.session_id && data.session_id !== Store.state.sessionId) {
        Store.saveSession(data.session_id);
        Util.$('#chat-session-id').textContent = data.session_id;
      }
      Trace.setMeta({ turnId: data?.turn_id });
      Trace.push({ kind: 'start', round: 0, content: `会话 ${data?.session_id || '-'} · 轮次 ${data?.turn_id || '-'}` });
    },

    onRoute(data, assistant) {
      if (this.turn) this.turn.route = data;
      const tools = data?.tools || [];
      Trace.push({
        kind: 'route', round: 0,
        content: `路由决策：${tools.length ? tools.join(' + ') : '直接回答'}${data?.reason ? ` · ${data.reason}` : ''}`,
        args: data
      });
      this.setTyping(assistant, `已选择工具：${tools.join('、') || '无需工具'}`);
    },

    onRetrieval(data, assistant) {
      if (this.turn) this.turn.retrieval = data;
      const results = data?.results || [];
      Trace.push({
        kind: 'retrieval', round: 0,
        content: `检索到 ${data?.num ?? results.length} 个片段`,
        args: results
      });
      this.setTyping(assistant, `检索到 ${data?.num ?? results.length} 个相关片段，正在组织答案…`);
    },

    onToken(data, assistant) {
      if (this.turn) this.turn.tokens += data?.text || '';
      this.ensureTyping(assistant);
      assistant.content.textContent += data?.text || '';
      UI.scrollMessages();
    },

    onCitation(data, assistant) {
      if (!this.turn) return;
      if (!this.turn.citations.some((c) => Number(c.index) === Number(data?.index))) {
        this.turn.citations.push(data);
      }
      Trace.push({ kind: 'citation', round: 0, content: `引用 [${data?.index}] ${data?.source || ''}${data?.page ? ` p.${data.page}` : ''}` });
      void assistant;
    },

    onFinal(data, assistant) {
      const citations = data?.citations || this.turn?.citations || [];
      if (this.turn) {
        this.turn.final = data;
        if (!this.turn.citations.length) this.turn.citations = citations;
      }
      this.renderAnswer(assistant, data?.answer || this.turn?.tokens || '', citations);
      this.renderMeta(assistant, data);
      Trace.push({
        kind: 'final', round: 0,
        content: `生成完成 · ${data?.iterations ?? 0} 轮推理 · ${data?.tool_calls ?? 0} 次工具调用 · ${Util.ms(data?.latency_ms)}`,
        args: data
      });
    },

    onError(data, assistant) {
      // 后端会把 Agent 内部"可恢复错误"（某个工具调用失败、参数校验不通过等）
      // 也以 error 事件推过来，并带 recoverable=true。这类错误 Agent 会自行
      // 重试或改道，**不应中断整个回答**；只有真正的致命错误才终止流。
      const recoverable = data?.recoverable === true;
      if (recoverable) {
        Trace.push({
          kind: 'error',
          round: 0,
          content: data?.message || '工具调用失败（已自动恢复）',
          args: data?.detail ?? null
        });
        if (this.turn) (this.turn.events || []).push({ event: 'error', data });
        return;
      }
      if (this.turn) this.turn.error = data;
      Trace.push({ kind: 'error', round: 0, content: data?.message || '未知错误', args: data?.detail ?? null });
      this.fail(assistant, data?.message || '生成失败', data?.detail);
    },

    onDone(data) {
      Trace.push({ kind: 'done', round: 0, content: `链路结束 · trace_id=${data?.trace_id || '-'}` });
      UI.scrollMessages();
      // 一轮结束后刷新会话列表：轮次计数与自动标题都会变
      Sessions.load(false).catch(() => {});
    },

    /* ---------- 消息 DOM ---------- */
    appendUser(text) {
      const node = Util.el('div', { class: 'msg user' }, [
        Util.el('div', { class: 'bubble', text })
      ]);
      Util.$('#messages').append(node);
      UI.scrollMessages(true);
      return node;
    },

    appendAssistant() {
      const meta = Util.el('div', { class: 'msg-meta' });
      const body = Util.el('div', { class: 'msg-body' });
      const typing = Util.el('div', { class: 'typing' }, [Util.el('i'), Util.el('i'), Util.el('i'),
        Util.el('span', { text: 'Agent 正在思考…' })]);
      body.append(typing);
      const bubble = Util.el('div', { class: 'bubble' }, [body, meta]);
      const node = Util.el('div', { class: 'msg assistant' }, [
        Util.el('div', { class: 'avatar', text: '🤖' }), bubble]);
      Util.$('#messages').append(node);
      UI.scrollMessages(true);
      return { node, bubble, body, meta, typing, content: null, finished: false };
    },

    setTyping(assistant, text) {
      if (!assistant?.typing) return;
      const span = assistant.typing.querySelector('span');
      if (span) span.textContent = text;
    },

    ensureTyping(assistant) {
      if (assistant.content) return;
      assistant.typing?.remove();
      assistant.content = Util.el('div', { class: 'md' });
      assistant.cursor = Util.el('span', { class: 'cursor' });
      assistant.body.replaceChildren(assistant.content, assistant.cursor);
    },

    renderAnswer(assistant, answer, citations) {
      this.ensureTyping(assistant);
      assistant.cursor?.remove();
      assistant.cursor = null;
      assistant.content.innerHTML = Markdown.render(answer || '（无内容）');
      this.renderCitations(assistant, citations || []);
      assistant.finished = true;
      UI.scrollMessages();
    },

    /** 渲染一条静态助手消息（历史恢复 / 最终答案共用） */
    fillAssistant(assistant, answer, citations, meta) {
      this.ensureTyping(assistant);
      assistant.cursor?.remove();
      assistant.cursor = null;
      assistant.content.innerHTML = Markdown.render(answer || '（无内容）');
      this.renderCitations(assistant, citations || []);
      if (meta) this.renderMeta(assistant, meta);
      assistant.finished = true;
    },

    renderCitations(assistant, citations) {
      if (!citations.length) return;
      const details = Util.el('details', { class: 'citations' });
      details.append(Util.el('summary', { text: `引用来源（${citations.length}）` }));
      const ordered = citations.slice().sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
      for (const c of ordered) {
        const score = Number(c.score);
        const scoreText = isFinite(score) ? `${(score * 100).toFixed(1)}%` : '—';
        const head = Util.el('div', { class: 'c-head' }, [
          Util.el('span', { class: 'c-idx', text: String(c.index ?? '?') }),
          Util.el('span', { class: 'c-source', text: c.source || '未知来源' }),
          c.page !== undefined && c.page !== null ? Util.el('span', { class: 'c-page', text: `第 ${c.page} 页` }) : null,
          Util.el('span', { class: 'c-score', text: `相关度 ${scoreText}` })]);
        const bar = Util.el('div', { class: 'c-bar' }, [Util.el('i', {
          style: `width:${isFinite(score) ? Math.max(3, Math.min(100, score * 100)) : 0}%` })]);
        const card = Util.el('div', {
          class: 'citation-card', dataset: { index: String(c.index ?? ''), source: c.source || '' }
        }, [head, bar, Util.el('div', { class: 'c-snippet', text: c.snippet || '（无片段预览）' })]);
        if ((c.snippet || '').length > 70) {
          card.append(Util.el('button', { class: 'c-more', type: 'button', text: '展开全文 ▾' }));
        }
        details.append(card);
      }
      assistant.body.append(details);
    },

    renderMeta(assistant, data) {
      const turn = this.turn || {};
      const usage = data?.usage || {};
      const toolCalls = Number(data?.tool_calls ?? 0);
      // 只认服务端给的权威标记：早期实现用"没有工具调用且没有检索结果"来推断，
      // 会把"检索为空 → 降级回答"也标成 ⚡命中缓存
      const cacheHit = Boolean(data?.cache_hit);
      const degraded = Boolean(data?.degraded || data?.degrade_reason || data?.terminated_reason === 'degraded');
      // 成本：价格未收录的模型只显示"未收录"，绝不显示一个凭空的金额
      // （后端 TokenUsage.pricing_known 字段；估算值统一带"估算"后缀）
      let cost = '—';
      if (usage.pricing_known === false) cost = '价格未收录';
      else if (typeof usage.cost_yuan === 'number') {
        cost = `¥${usage.cost_yuan < 0.01 ? usage.cost_yuan.toFixed(5) : usage.cost_yuan.toFixed(4)}`;
        if (usage.estimated) cost += '（估算）';
      }
      const chips = [['耗时', Util.ms(data?.latency_ms)],
        ['tokens', usage.total_tokens !== undefined ? String(usage.total_tokens) : '—'],
        ['成本', cost],
        ['迭代', `${data?.iterations ?? 0} 轮`], ['工具', `${toolCalls} 次`]];
      assistant.meta.replaceChildren();
      for (const [k, v] of chips) assistant.meta.append(Util.el('span', { class: 'meta-chip', text: `${k} ${v}` }));
      assistant.meta.append(Util.el('span', {
        class: `meta-chip ${cacheHit ? 'hit' : 'miss'}`,
        text: cacheHit ? '⚡ 命中缓存' : '未命中缓存'
      }));
      if (degraded) {
        assistant.meta.append(Util.el('span', {
          class: 'meta-chip degraded', text: `⚠ 已降级${data?.terminated_reason ? `（${data.terminated_reason}）` : ''}`
        }));
      }
      if (data?.terminated_reason) {
        assistant.meta.append(Util.el('span', { class: 'meta-chip', text: `终止：${data.terminated_reason}` }));
      }
    },

    fail(assistant, message, detail) {
      assistant.typing?.remove();
      assistant.cursor?.remove();
      assistant.cursor = null;
      assistant.node.classList.add('error');
      assistant.body.replaceChildren(Util.el('div', { class: 'md' }, [
        Util.el('p', {}, [Util.el('strong', { text: '生成中断：' }), String(message || '未知错误')]),
        detail ? Util.el('p', { text: Util.cut(typeof detail === 'string' ? detail : Util.json(detail), 300) }) : null
      ]));
      Toast.error(`生成失败：${message || '未知错误'}`);
    },

    finishInterrupted(assistant) {
      assistant.cursor?.remove();
      assistant.cursor = null;
      if (!assistant.finished && assistant.content) {
        assistant.meta.append(Util.el('span', { class: 'meta-chip degraded', text: '⏹ 已手动停止' }));
      }
    },

    /** 点击角标：滚动并高亮对应引用卡片 */
    focusCitation(ref) {
      const bubble = ref.closest('.bubble');
      const index = ref.dataset.index;
      const card = bubble?.querySelector(`.citation-card[data-index="${index}"]`);
      if (!card) { Toast.info(`未找到引用 [${index}] 的卡片`); return; }
      const details = card.closest('details');
      if (details) details.open = true;
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.classList.add('flash');
      setTimeout(() => card.classList.remove('flash'), 1600);
    },

    /** 恢复历史会话中的一轮问答 */
    restoreTurn(turn, isLast) {
      if (turn.question) this.appendUser(turn.question);
      const assistant = this.appendAssistant();
      const traj = turn.trajectory || [];
      // cache_hit / degraded 取落库时记下的真实值（turn.metadata），不再靠"有没有引用"倒推
      const meta = turn.metadata || {};
      this.fillAssistant(assistant, turn.answer, turn.citations, {
        latency_ms: turn.latency_ms, usage: turn.usage,
        iterations: traj.filter((s) => s.type === 'thought').length,
        tool_calls: traj.filter((s) => s.type === 'action').length,
        cache_hit: Boolean(meta.cache_hit),
        degraded: Boolean(meta.degraded),
        terminated_reason: meta.terminated_reason || ''
      });
      if (isLast) {
        Trace.restore(traj);
        Trace.setJson(traj);
      }
    }
  };

  /* ==========================================================================
   * Trace —— Agent 推理轨迹
   * ========================================================================*/
  const Trace = {
    meta: {},
    begin(question) {
      Store.state.traces = [];
      this.meta = { question, started: Date.now() };
      this.render();
    },
    push(step) {
      Store.state.traces.push(step);
      this.appendNode(step);
      this.renderStats();
      this.syncJson();
      UI.scrollTrace();
    },
    setMeta(patch) { Object.assign(this.meta, patch); },
    flush() { this.renderStats(); this.syncJson(); },

    clear() {
      Store.state.traces = [];
      this.render();
      Toast.info('轨迹已清空');
    },

    toggleJson() {
      Store.state.traceJsonMode = !Store.state.traceJsonMode;
      const btn = Util.$('#btn-trace-json');
      btn?.classList.toggle('on', Store.state.traceJsonMode);
      Util.$('#trace-timeline').hidden = Store.state.traceJsonMode;
      Util.$('#trace-json').hidden = !Store.state.traceJsonMode;
      if (Store.state.traceJsonMode) this.syncJson();
    },

    render() {
      const box = Util.$('#trace-timeline');
      if (!box) return;
      box.replaceChildren();
      if (!Store.state.traces.length) {
        box.append(Util.el('p', { class: 'empty-hint' }, [
          '暂无推理轨迹', Util.el('br'), Util.el('span', { text: '发起提问后，这里会实时展示 Thought → Action → Observation' })
        ]));
      } else {
        Store.state.traces.forEach((s) => box.append(this.node(s)));
      }
      this.renderStats();
    },

    appendNode(step) {
      const box = Util.$('#trace-timeline');
      if (!box) return;
      if (Store.state.traces.length === 1) box.replaceChildren();
      box.append(this.node(step));
    },

    KIND: {
      start: { icon: '▶', label: 'START' }, route: { icon: '⑂', label: 'ROUTE' },
      retrieval: { icon: '🔍', label: 'RETRIEVE' }, thought: { icon: '💭', label: 'THOUGHT' },
      action: { icon: '⚙', label: 'ACTION' }, observation: { icon: '👁', label: 'OBSERVE' },
      citation: { icon: '❝', label: 'CITATION' }, final: { icon: '✔', label: 'FINAL' },
      error: { icon: '✕', label: 'ERROR' }, done: { icon: '⏹', label: 'DONE' }
    },

    node(step) {
      const kind = step.kind || 'thought';
      const meta = this.KIND[kind] || { icon: '•', label: kind.toUpperCase() };
      const bodyText = String(step.content || '');
      const needClamp = bodyText.length > Config.TRACE_PREVIEW;
      const body = Util.el('div', { class: `t-body${needClamp ? ' clamp' : ''}`, text: bodyText });
      const card = Util.el('div', { class: 't-card' }, [
        Util.el('div', { class: 't-top' }, [
          Util.el('span', { class: 't-type', text: `${meta.icon} ${meta.label}` }),
          step.round ? Util.el('span', { class: 't-round', text: `R${step.round}` }) : null,
          step.tool ? Util.el('span', { class: 't-tool', text: `${Config.TOOL_ICON[step.tool] || ''} ${step.tool}` }) : null,
          step.status ? Util.el('span', { class: `badge ${step.status === 'success' ? 'ok' : 'bad'}`, text: step.status }) : null,
          step.latency_ms !== undefined && step.latency_ms !== null
            ? Util.el('span', { class: 't-lat', text: Util.ms(step.latency_ms) }) : null]),
        body]);
      if (needClamp) {
        const btn = Util.el('button', { class: 't-expand', type: 'button', text: '展开 ▾' });
        btn.addEventListener('click', () => {
          const clamped = body.classList.toggle('clamp');
          btn.textContent = clamped ? '展开 ▾' : '收起 ▴';
        });
        card.append(btn);
      }
      if (step.args !== undefined && step.args !== null) {
        card.append(Util.el('details', {}, [Util.el('summary', { text: '参数 / 原始数据' }),
          Util.el('pre', { text: Util.cut(Util.json(step.args), 4000) })]));
      }
      return Util.el('div', { class: 'trace-step', dataset: { kind } },
        [Util.el('div', { class: 't-dot', text: meta.icon }), card]);
    },

    renderStats() {
      const steps = Store.state.traces;
      const rounds = steps.reduce((a, s) => Math.max(a, Number(s.round) || 0), 0);
      const actions = steps.filter((s) => s.kind === 'action').length;
      const observations = steps.filter((s) => s.kind === 'observation' || (s.kind === 'error' && s.tool));
      const okCount = observations.filter((s) => s.status === 'success' || s.kind === 'observation').length;
      Util.$('#trace-rounds').textContent = String(rounds);
      Util.$('#trace-tools').textContent = String(actions);
      Util.$('#trace-success').textContent = observations.length
        ? `${((okCount / observations.length) * 100).toFixed(0)}%` : '—';
    },

    syncJson() {
      const pre = Util.$('#trace-json');
      if (!pre) return;
      pre.textContent = Util.json({
        question: this.meta.question || null,
        turn_id: this.meta.turnId || null,
        steps: Store.state.traces
      });
    },

    setJson(value) {
      const pre = Util.$('#trace-json');
      if (pre) pre.textContent = Util.json(value);
    },

    /** 从会话历史恢复轨迹 */
    restore(trajectory) {
      Store.state.traces = (trajectory || []).map((s) => ({
        kind: s.type === 'observation' && s.status && s.status !== 'success' ? 'error' : (s.type || 'thought'),
        round: s.round,
        tool: s.tool,
        args: s.args,
        status: s.status,
        latency_ms: s.latency_ms,
        content: s.content || ''
      }));
      this.render();
    }
  };

  /* ==========================================================================
   * Metrics —— 系统指标弹窗
   * ========================================================================*/
  const Metrics = {
    timer: null,
    latest: null,

    open() {
      const modal = Util.$('#metrics-modal');
      if (!modal) return;
      modal.hidden = false;
      this.load();
      clearInterval(this.timer);
      this.timer = setInterval(() => this.load(), Config.METRICS_INTERVAL);
    },

    close() {
      const modal = Util.$('#metrics-modal');
      if (modal) modal.hidden = true;
      clearInterval(this.timer);
      this.timer = null;
    },

    async load(notify = false) {
      const body = Util.$('#metrics-body');
      try {
        const data = await API.metrics();
        this.latest = data;
        this.render(data);
        if (notify) Toast.success('指标已刷新');
      } catch (err) {
        const kpi = Util.$('#kpi-grid');
        if (kpi) kpi.innerHTML = `<p class="empty-hint">指标加载失败：${Util.esc(err.message)}</p>`;
        void body;
        if (notify) Toast.error(`指标加载失败：${err.message}`);
      }
    },

    render(data) {
      const rag = data?.rag || {};
      const agent = data?.agent || {};
      const retrieval = data?.retrieval || {};
      const tools = data?.tools || [];

      /* Web 端的提问都走 Agent，而 Agent 的 knowledge_search 是**直连检索引擎**的
         （不构造 RAGAnswer），所以 rag.requests 在真实使用下恒为 0。
         这时 KPI 改用「检索/Agent」侧的计数，否则整块面板全是 0。*/
      const pipeline = Number(rag.requests) > 0;
      const kpi = {
        requests: pipeline ? rag.requests : (agent.requests ?? 0),
        hitRate: pipeline ? rag.hit_rate : retrieval.hit_rate,
        cacheHitRate: rag.cache_hit_rate,
        avgLatency: pipeline ? rag.avg_latency_ms : agent.avg_latency_ms,
        p95: pipeline ? rag.p95_latency_ms : agent.p95_latency_ms,
        retrievalP50: pipeline ? rag.retrieval_p50_ms : retrieval.p50_latency_ms,
        degradeRate: rag.degrade_rate,
        topScore: pipeline ? rag.avg_top_score : retrieval.avg_top_score
      };
      const num = (value, digits) => (typeof value === 'number' ? value.toFixed(digits) : '—');

      /* --- KPI 卡片 --- */
      const kpis = [
        ['请求数', Util.num(kpi.requests ?? 0), 'accent'],
        ['检索命中率', kpi.hitRate !== undefined ? Util.pct(kpi.hitRate) : '—', 'ok'],
        ['缓存命中率', rag.cache_hit_rate !== undefined ? Util.pct(rag.cache_hit_rate) : '—', ''],
        ['平均延迟', kpi.avgLatency !== undefined ? Util.ms(kpi.avgLatency) : '—', ''],
        ['P95 延迟', kpi.p95 !== undefined ? Util.ms(kpi.p95) : '—', 'warn'],
        ['检索 P50', kpi.retrievalP50 !== undefined ? Util.ms(kpi.retrievalP50) : '—', ''],
        ['工具成功率', agent.tool_success_rate !== undefined ? Util.pct(agent.tool_success_rate) : '—', 'ok'],
        ['降级率', rag.degrade_rate !== undefined ? Util.pct(rag.degrade_rate) : '—', 'warn'],
        ['总 tokens', Util.num((rag.total_tokens || 0) + (agent.total_tokens || 0)), 'accent'],
        ['平均轮次', agent.avg_iterations !== undefined ? String(agent.avg_iterations) : '—', ''],
        ['平均 Top 分数', num(kpi.topScore, 3), ''],
        ['工具调用总数', Util.num(agent.total_tool_calls ?? 0), '']];
      const grid = Util.$('#kpi-grid');
      grid.replaceChildren();
      for (const [label, value, cls] of kpis) {
        grid.append(Util.el('div', { class: `kpi-card ${cls}` },
          [Util.el('div', { class: 'k-label', text: label }), Util.el('div', { class: 'k-value', text: value })]));
      }

      /* --- 工具调用表格 --- */
      const wrap = Util.$('#metrics-tools-table');
      wrap.replaceChildren();
      if (!tools.length) {
        wrap.append(Util.el('p', { class: 'empty-hint', text: '暂无工具调用记录' }));
      } else {
        const rows = tools.map((t) => [`${Config.TOOL_ICON[t.tool] || '🧩'} ${t.tool || '-'}`,
          Util.num(t.calls ?? 0), Util.num(t.success ?? 0),
          t.success_rate !== undefined ? Util.pct(t.success_rate) : '—',
          t.avg_latency_ms !== undefined ? Util.ms(t.avg_latency_ms) : '—']);
        wrap.append(this.table(['工具', '调用', '成功', '成功率', '平均耗时'], rows));
      }

      /* --- 终止原因分布 --- */
      const reasons = agent.terminated_reasons || {};
      this.bars('#metrics-reasons', Object.entries(reasons).map(([k, v]) => ({ label: k, value: Number(v) || 0 })),
        (v, max) => `${v} 次（${max ? ((v / max) * 100).toFixed(0) : 0}%）`);

      /* --- 检索质量 --- */
      const ragBars = [['命中率', Number(kpi.hitRate) || 0, false], ['缓存命中率', Number(rag.cache_hit_rate) || 0, false],
        ['降级率', Number(rag.degrade_rate) || 0, true], ['平均 Top 分数', Number(kpi.topScore) || 0, false]];
      this.bars('#metrics-rag', ragBars.map(([label, value, warn]) => ({ label, value, ratio: true, warn })),
        (v, max, item) => (item.ratio ? Util.pct(v) : `${v} / ${max}`));

      /* --- 最近问题 --- */
      const list = Util.$('#metrics-recent');
      list.replaceChildren();
      const recent = data?.recent_questions || [];
      if (!recent.length) list.append(Util.el('li', { class: 'empty-hint', text: '暂无提问记录' }));
      else recent.slice(0, 12).forEach((q) => list.append(Util.el('li', { text: Util.cut(q, 120) })));
    },

    table(headers, rows) {
      const thead = Util.el('thead', {}, [Util.el('tr', {}, headers.map((h) => Util.el('th', { text: h })))]);
      const tbody = Util.el('tbody', {}, rows.map((r) => Util.el('tr', {},
        r.map((c, i) => Util.el('td', { class: i ? 'num' : '', text: String(c) })))));
      return Util.el('table', { class: 'data-table' }, [thead, tbody]);
    },

    /** 纯 CSS 条形图（不引入任何图表库） */
    bars(selector, items, formatter) {
      const box = Util.$(selector);
      if (!box) return;
      box.replaceChildren();
      if (!items.length) { box.append(Util.el('p', { class: 'empty-hint', text: '暂无数据' })); return; }
      const max = Math.max(...items.map((i) => Number(i.value) || 0), 0.0001);
      for (const item of items) {
        const v = Number(item.value) || 0;
        const width = item.ratio ? Math.max(1, Math.min(100, v * 100)) : Math.max(1, (v / max) * 100);
        box.append(Util.el('div', { class: 'bar-row' }, [
          Util.el('span', { class: 'b-label', text: item.label, title: item.label }),
          Util.el('span', { class: 'b-track' },
            [Util.el('i', { class: `b-fill${item.warn ? ' warn' : ''}`, style: `width:${width}%` })]),
          Util.el('span', { class: 'b-value', text: formatter ? formatter(v, max, item) : String(v) })]));
      }
    }
  };

  /* ==========================================================================
   * 启动
   * ========================================================================*/
  async function boot() {
    UI.init();
    Docs.init();
    await Health.refresh();
    Health.start();
    Docs.load();
    await Sessions.load(true);
    UI.setStreaming(false);
    console.info(`%c智能科研助理 v${Config.VERSION}%c 前端已就绪`, 'color:#6366f1;font-weight:700', 'color:inherit');
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();

  // 暴露给调试台（便于手工联调）
  window.App = { Config, Util, Store, Toast, API, SSE, UI, Sessions, Docs, Chat, Trace, Metrics, Health };
})();
