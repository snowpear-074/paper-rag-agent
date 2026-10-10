/* ============================================================================
 * markdown.js —— 手写轻量 Markdown 渲染器（零依赖）
 *
 * 安全模型（重要）
 * ----------------
 * 全文遵循「先转义、后插标签」的唯一路径：所有用户/模型产出的原始文本，
 * 必须先经过 escapeHtml() 才能进入任何 HTML 拼接语句；渲染过程中产生的
 * HTML 片段一律先寄存到 slot 里，用哨兵占位符参与后续正则处理，最后统一
 * 还原。因此任何 `<script>`、`onerror=`、`javascript:` 都无法逃逸为活标签。
 *
 * 支持语法：标题 #~####、粗体 **、斜体 *、删除线 ~~、行内代码 `、
 * 围栏代码块 ```（带语言标签 + 复制按钮）、有序/无序列表（支持一层嵌套）、
 * 引用 >、表格 | a | b |、水平线 ---、链接 [t](url)、图片 ![alt](src)、自动链接、
 * 段落与换行。
 *
 * 额外规则：把「【来源名 p.3】」与独立的 [1] [2] 引用标记转成
 * <sup class="citation-ref" data-index="1">[1]</sup>，供前端点击跳转引用卡片。
 * 该正则同时被 tests/test_frontend_static.py 以纯 Python 复现校验，两端保持一致。
 * ==========================================================================*/
(function (global) {
  'use strict';

  /* ------------------------------------------------------------------
   * 0. HTML 转义 —— 整个渲染器的安全基石，必须在任何标签插入之前调用
   * ----------------------------------------------------------------*/
  function escapeHtml(input) {
    if (input === null || input === undefined) return '';
    return String(input)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /* ------------------------------------------------------------------
   * 1. 引用角标规则
   *    CITATION_SOURCE_RE  匹配「【来源名 p.3】」「【[1] 来源名 p.3】」「【Attention p.3】」
   *    CITATION_STANDALONE 匹配独立的 [1]（文本之后，用负向断言排除 markdown 链接）
   * ----------------------------------------------------------------*/
  var CITATION_SOURCE_RE = /【\s*(?:\[?(\d{1,3})\]?\s*)?([^】\n]{1,120}?)\s+(?:[pP]\.|第)\s*(\d{1,4})\s*页?\s*】/g;
  var CITATION_STANDALONE_RE = /(^|[^\w\]])\[(\d{1,3})\](?![(（\w])/g;

  /** 生成角标 HTML（参数一律先转义） */
  function citationTag(index, source, page) {
    var title = source ? source + (page ? ' 第 ' + page + ' 页' : '') : '引用来源 ' + index;
    var attrs = ' class="citation-ref" data-index="' + escapeHtml(index) + '"';
    if (source) attrs += ' data-source="' + escapeHtml(source) + '"';
    if (page) attrs += ' data-page="' + escapeHtml(page) + '"';
    attrs += ' title="' + escapeHtml(title) + '" role="button" tabindex="0"';
    return '<sup' + attrs + '>[' + escapeHtml(index) + ']</sup>';
  }

  /** 兜底分配引用序号（同一来源复用同一序号） */
  function fallbackIndex(source, page) {
    var key = String(source || '') + '|' + String(page || '');
    if (!fallbackIndex.map) fallbackIndex.map = {};
    var map = fallbackIndex.map;
    if (!map[key]) map[key] = Object.keys(map).length + 1;
    return map[key];
  }

  /**
   * 把引用标记替换成角标 HTML。
   * @param {string} escaped 已完成 HTML 转义的文本
   * @param {function} put   寄存 HTML 片段并返回哨兵占位符
   */
  function rewriteCitations(escaped, put) {
    var out = escaped.replace(CITATION_SOURCE_RE, function (whole, index, source, page) {
      var name = String(source || '').trim();
      var idx = index ? String(index) : String(fallbackIndex(name, page));
      return put(citationTag(idx, name, String(page)));
    });
    out = out.replace(CITATION_STANDALONE_RE, function (whole, prev, index) {
      return prev + put(citationTag(String(index), '', ''));
    });
    return out;
  }

  /* ------------------------------------------------------------------
   * 2. 行内解析：转义 → 引用角标 → 行内代码 → 图片/链接 → 强调 → 换行
   *    所有 HTML 片段先寄存到 slot，绝不参与后续正则，避免标签被二次加工。
   * ----------------------------------------------------------------*/
  function parseInline(raw) {
    var slots = [];
    var put = function (html) {
      slots.push(html);
      return '\u0001' + (slots.length - 1) + '\u0001';
    };

    // 第一步：转义（先于任何标签插入）
    var text = rewriteCitations(escapeHtml(raw), put);

    // 行内代码（内容已转义，直接包裹标签）
    text = text.replace(/`([^`\n]+)`/g, function (whole, code) {
      return put('<code class="inline">' + code + '</code>');
    });

    // 图片（先于链接处理，避免 ![]() 被链接规则吃掉）
    text = text.replace(/!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+&quot;([^)]*)&quot;)?\)/g,
      function (whole, alt, src, title) {
        var safeSrc = /^\s*(?:javascript|data|vbscript)\s*:/i.test(src) ? '' : src;
        if (!safeSrc) return put('<span>' + alt + '</span>');
        var t = title ? ' title="' + title + '"' : '';
        return put('<img src="' + safeSrc + '" alt="' + alt + '"' + t + ' loading="lazy" />');
      });

    // 链接：仅允许 http/https/mailto，其余降级为纯文本
    text = text.replace(/\[([^\]\n]*)\]\(([^)\s]+)(?:\s+&quot;([^)]*)&quot;)?\)/g,
      function (whole, label, href, title) {
        var ok = /^(?:https?:|mailto:|#|\/|\.)/i.test(href);
        if (!ok) return label;
        var t = title ? ' title="' + title + '"' : '';
        return put('<a href="' + href + '" target="_blank" rel="noopener noreferrer"' + t + '>' +
          (label || href) + '</a>');
      });

    // 自动链接（裸 URL）
    text = text.replace(/(^|[\s(（])(https?:\/\/[^\s<)）]+)/g, function (whole, prev, url) {
      return prev + put('<a href="' + url + '" target="_blank" rel="noopener noreferrer">' + url + '</a>');
    });

    // 粗体 / 斜体 / 删除线（由强到弱，避免 ** 被 * 抢先匹配）
    text = text.replace(/\*\*([^\n]+?)\*\*/g, function (whole, inner) {
      return put('<strong>' + inner + '</strong>');
    });
    text = text.replace(/(^|[^*\w])\*([^*\n]+?)\*(?![*\w])/g, function (whole, prev, inner) {
      return prev + put('<em>' + inner + '</em>');
    });
    text = text.replace(/~~([^\n]+?)~~/g, function (whole, inner) {
      return put('<del>' + inner + '</del>');
    });

    // 还原寄存的 HTML
    text = text.replace(/\u0001(\d+)\u0001/g, function (whole, i) {
      return slots[Number(i)] !== undefined ? slots[Number(i)] : '';
    });
    return text;
  }

  /* ------------------------------------------------------------------
   * 3. 块级解析
   * ----------------------------------------------------------------*/
  var RE_FENCE = /^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
  var RE_HEADING = /^(#{1,4})\s+(.*)$/;
  var RE_HR = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;
  var RE_QUOTE = /^\s*>\s?(.*)$/;
  var RE_UL = /^(\s*)([-*+])(?![*+-])\s+(.*)$/;
  var RE_OL = /^(\s*)(\d{1,3})[.)]\s+(.*)$/;
  var RE_TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$/;

  function isBlank(line) { return !line || !line.trim(); }

  function splitRow(line) {
    var s = line.trim();
    if (s.charAt(0) === '|') s = s.slice(1);
    if (s.charAt(s.length - 1) === '|') s = s.slice(0, -1);
    return s.split('|').map(function (c) { return c.trim(); });
  }

  function renderTable(header, rows) {
    var html = '<div class="table-wrap"><table><thead><tr>';
    header.forEach(function (c) { html += '<th>' + parseInline(c) + '</th>'; });
    html += '</tr></thead><tbody>';
    rows.forEach(function (row) {
      html += '<tr>';
      for (var i = 0; i < header.length; i++) {
        html += '<td>' + parseInline(row[i] === undefined ? '' : row[i]) + '</td>';
      }
      html += '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function renderCodeBlock(lang, lines) {
    var label = lang || 'text';
    return '<div class="code-block">' +
      '<div class="cb-head"><span class="cb-lang">' + escapeHtml(label) + '</span>' +
      '<button type="button" class="cb-copy" data-copy="1">复制</button></div>' +
      '<pre><code data-lang="' + escapeHtml(label) + '">' +
      escapeHtml(lines.join('\n')) + '</code></pre></div>';
  }

  /** 列表：同一层级的项归入同一 <ul>/<ol>，更深缩进递归成嵌套列表 */
  function renderList(items) {
    var ordered = items[0].ordered;
    var tag = ordered ? 'ol' : 'ul';
    var html = '<' + tag + '>';
    var cur = null;
    for (var k = 0; k < items.length; k++) {
      var it = items[k];
      if (it.level <= items[0].level) {
        if (cur) html += '</li>';
        html += '<li>' + parseInline(it.text);
        cur = it;
      } else {
        if (!cur) { html += '<li>'; cur = items[0]; }
        var sub = [];
        for (; k < items.length && items[k].level > items[0].level; k++) {
          sub.push({ text: items[k].text, level: items[k].level, ordered: items[k].ordered });
        }
        k--;
        html += renderList(sub); // 递归渲染嵌套层
      }
    }
    if (cur) html += '</li>';
    return html + '</' + tag + '>';
  }

  function describeList(line) {
    var m = RE_OL.exec(line);
    if (m) return { level: Math.floor(m[1].replace(/\t/g, '  ').length / 2), text: m[3], ordered: true };
    m = RE_UL.exec(line);
    if (m) return { level: Math.floor(m[1].replace(/\t/g, '  ').length / 2), text: m[3], ordered: false };
    return null;
  }

  function renderBlocks(text) {
    var lines = String(text === null || text === undefined ? '' : text)
      .replace(/\r\n?/g, '\n').split('\n');
    var out = [];
    var i = 0;
    var para = [];

    function flushPara() {
      if (!para.length) return;
      var html = para.map(function (l) { return parseInline(l); }).join('<br />');
      out.push('<p>' + html + '</p>');
      para = [];
    }

    while (i < lines.length) {
      var line = lines[i];

      // 围栏代码块
      var fence = RE_FENCE.exec(line);
      if (fence) {
        flushPara();
        var marker = fence[1].charAt(0);
        var lang = fence[2] || '';
        var buf = [];
        i++;
        while (i < lines.length) {
          var close = RE_FENCE.exec(lines[i]);
          if (close && close[1].charAt(0) === marker) { i++; break; }
          buf.push(lines[i]); i++;
        }
        out.push(renderCodeBlock(lang, buf));
        continue;
      }

      if (isBlank(line)) { flushPara(); i++; continue; }

      // 水平线（若紧跟段落文本，则视为 setext 标题下划线，归入段落，不打断阅读）
      if (RE_HR.test(line) && !RE_UL.test(line) && !para.length) {
        flushPara(); out.push('<hr />'); i++; continue;
      }

      // 标题
      var h = RE_HEADING.exec(line);
      if (h) {
        flushPara();
        var lv = h[1].length;
        out.push('<h' + lv + '>' + parseInline(h[2].replace(/\s+#+\s*$/, '')) + '</h' + lv + '>');
        i++; continue;
      }

      // 表格
      if (line.indexOf('|') !== -1 && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1])) {
        flushPara();
        var header = splitRow(line);
        var rows = [];
        i += 2;
        while (i < lines.length && !isBlank(lines[i]) && lines[i].indexOf('|') !== -1) {
          rows.push(splitRow(lines[i])); i++;
        }
        out.push(renderTable(header, rows));
        continue;
      }

      // 引用
      if (RE_QUOTE.test(line)) {
        flushPara();
        var quote = [];
        while (i < lines.length && RE_QUOTE.test(lines[i])) {
          quote.push(RE_QUOTE.exec(lines[i])[1]); i++;
        }
        out.push('<blockquote>' + renderBlocks(quote.join('\n')) + '</blockquote>');
        continue;
      }

      // 列表
      if (describeList(line)) {
        flushPara();
        var items = [];
        while (i < lines.length && describeList(lines[i])) {
          items.push(describeList(lines[i])); i++;
        }
        out.push(renderList(items));
        continue;
      }

      // 普通段落
      para.push(line);
      i++;
    }
    flushPara();
    return out.join('\n');
  }

  /* ------------------------------------------------------------------
   * 4. 对外 API
   * ----------------------------------------------------------------*/
  /** 渲染 Markdown 为安全 HTML */
  function renderMarkdown(src) { return renderBlocks(src); }

  /** 提取纯文本摘要（去掉 markdown 标记），用于会话标题 / 列表预览 */
  function toPlainText(src, maxLength) {
    var plain = String(src === null || src === undefined ? '' : src)
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`([^`]*)`/g, '$1')
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(CITATION_SOURCE_RE, '')
      .replace(/^\s{0,3}#{1,6}\s+/gm, '')
      .replace(/^\s{0,3}>\s?/gm, '')
      .replace(/\*\*|~~|\*/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    var limit = maxLength || 60;
    return plain.length > limit ? plain.slice(0, limit) + '…' : plain;
  }

  global.Markdown = {
    escapeHtml: escapeHtml,
    render: renderMarkdown,
    renderInline: parseInline,
    toPlainText: toPlainText,
    citationTag: citationTag,
    CITATION_SOURCE_RE: CITATION_SOURCE_RE,
    CITATION_STANDALONE_RE: CITATION_STANDALONE_RE
  };
})(typeof window !== 'undefined' ? window : globalThis);
