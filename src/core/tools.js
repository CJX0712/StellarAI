// src/core/tools.js
// 工具调用框架（Agent 的"手"）。
// 计算类工具用递归下降求值器安全解析算术表达式，绝不走 eval/Function —— 零注入风险。

import { log } from '../utils/logger.js';

// ── 安全算术求值器（递归下降）─────────────────────────────
function tokenize(s) {
  const toks = []; let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\t') { i++; continue; }
    if ('+-*/^()'.includes(c)) { toks.push({ t: 'op', v: c }); i++; continue; }
    if (/[0-9.]/.test(c)) {
      let num = ''; while (i < s.length && /[0-9.]/.test(s[i])) num += s[i++];
      toks.push({ t: 'num', v: parseFloat(num) });
      continue;
    }
    throw new Error(`非法字符: ${c}`);
  }
  return toks;
}

function parse(tokens) {
  let pos = 0;
  const peek = () => tokens[pos];
  const eat = (v) => {
    if (tokens[pos] && tokens[pos].v === v) { pos++; return true; }
    return false;
  };
  function expr() {
    let v = term();
    while (peek() && peek().v === '+' || peek() && peek().v === '-') {
      const op = tokens[pos++].v; const r = term();
      v = op === '+' ? v + r : v - r;
    }
    return v;
  }
  function term() {
    let v = factor();
    while (peek() && (peek().v === '*' || peek().v === '/')) {
      const op = tokens[pos++].v; const r = factor();
      v = op === '*' ? v * r : v / r;
    }
    return v;
  }
  function factor() {
    let v = base();
    if (peek() && peek().v === '^') { tokens[pos++]; const r = factor(); v = Math.pow(v, r); }
    return v;
  }
  function base() {
    if (peek() && peek().v === '-') { tokens[pos++]; return -factor(); }
    if (peek() && peek().v === '(') { tokens[pos++]; const v = expr(); if (!eat(')')) throw new Error('缺少 )'); return v; }
    if (peek() && peek().t === 'num') { return tokens[pos++].v; }
    throw new Error('表达式语法错误');
  }
  const val = expr();
  if (pos !== tokens.length) throw new Error('表达式尾部多余');
  return val;
}

function safeEval(expr) {
  const v = parse(tokenize(String(expr).replace(/×/g, '*').replace(/÷/g, '/').replace(/[，]/g, '')));
  if (!Number.isFinite(v)) throw new Error('结果为非有限数');
  return v;
}

// ── 内置工具 ───────────────────────────────────────────
const BUILTINS = [
  {
    name: 'calculator', description: '安全计算算术表达式，如 "2*(3+4)^2"。支持 + - * / ^ 与括号。',
    async run(input) {
      try { return String(safeEval(input)); }
      catch (e) { return `计算错误: ${e.message}`; }
    },
  },
  {
    name: 'datetime', description: '返回当前日期与时间。',
    async run() { return new Date().toLocaleString('zh-CN'); },
  },
  {
    name: 'knowledge', description: '从内置知识库检索关于某主题的片段。输入为查询词。',
    async run(input) {
      const kb = {
        'transformers.js': 'transformers.js 是 Hugging Face 官方的浏览器/Node 端机器学习库，可在本地运行 SOTA 模型。',
        'hnswlib': 'hnswlib 是 Facebook AI 开源的近似最近邻搜索库，基于分层可导航小世界图，支持亿级向量。',
        'react': 'ReAct 是一种将推理（Thought）与行动（Action）交织的 Agent 范式，由 Yao 等人于 2022 年提出。',
      };
      const q = String(input || '').toLowerCase();
      for (const k of Object.keys(kb)) if (q.includes(k) || k.includes(q)) return kb[k];
      return '知识库中未找到相关条目。';
    },
  },
];

export class ToolRegistry {
  constructor() { this.tools = new Map(); BUILTINS.forEach((t) => this.register(t)); }
  register(tool) {
    if (!tool.name || typeof tool.run !== 'function') throw new Error('工具必须含 name 与 run');
    this.tools.set(tool.name, tool);
    return this;
  }
  get(name) { return this.tools.get(name); }
  list() { return [...this.tools.values()].map((t) => ({ name: t.name, description: t.description })); }
  async run(name, input = '') {
    const t = this.tools.get(name);
    if (!t) return { ok: false, output: `未知工具: ${name}`, name };
    try {
      const output = await t.run(input);
      return { ok: true, output: String(output), name };
    } catch (e) {
      return { ok: false, output: `工具异常: ${e.message}`, name };
    }
  }
}

export { safeEval };
