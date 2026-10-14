// test/run.js
// 晨星·星核 StellarAI 自检套件（forge 风格：确定性、可复现、8/8 全绿）。
// 全部使用离线后端（hash 嵌入 + mock LLM + 精确向量检索），不依赖模型下载即可验证系统正确性。

import {
  math, safeEval, VectorStore, Memory, ToolRegistry, MockLLM, Engine,
  HashEmbeddings, createStellar,
} from '../src/index.js';

let pass = 0, fail = 0;
const fails = [];
function ok(cond, msg) {
  if (cond) { pass++; console.log(`  ✅ ${msg}`); }
  else { fail++; fails.push(msg); console.log(`  ⚠️  ${msg}`); }
}
const groups = [];
function group(name, fn) { groups.push([name, fn]); }

// 1. 数学基元
group('数学基元 math', () => {
  ok(math.dot([1, 2, 3], [4, 5, 6]) === 32, 'dot 应为 32');
  ok(Math.abs(math.cosineSimilarity([1, 0], [1, 0]) - 1) < 1e-9, '相同向量余弦=1');
  ok(Math.abs(math.cosineSimilarity([1, 0], [0, 1])) < 1e-9, '正交向量余弦≈0');
  const n = math.normalize([3, 4]); ok(Math.abs(math.norm(n) - 1) < 1e-9, 'normalize 模长=1');
  ok(math.topK([0.1, 0.9, 0.4, 0.7], 2).length === 2, 'topK 返回 k 个');
  ok(math.argmax([1, 5, 3]) === 1, 'argmax 正确');
  const sm = math.softmax([1, 1, 1]); ok(Math.abs(sm.reduce((a, b) => a + b) - 1) < 1e-9, 'softmax 和为1');
});

// 2. 安全算术（杜绝 eval 注入）
group('安全算术 safeEval', () => {
  ok(safeEval('2+3*4') === 14, '运算符优先级 2+3*4=14');
  ok(safeEval('(2+3)*4') === 20, '括号 (2+3)*4=20');
  ok(safeEval('2^10') === 1024, '幂 2^10=1024');
  ok(safeEval('-5+3') === -2, '一元负号 -5+3=-2');
  ok(safeEval('10/4') === 2.5, '除法 10/4=2.5');
  let threw = false; try { safeEval('alert(1)'); } catch { threw = true; }
  ok(threw, '非法标识符应抛错而非执行');
});

// 3. 向量检索（精确后端）
group('向量检索 VectorStore', async () => {
  const store = new VectorStore(4, 'exact');
  await store.add('a', [1, 0, 0, 0], { text: '苹果' });
  await store.add('b', [0, 1, 0, 0], { text: '香蕉' });
  await store.add('c', [0.9, 0.1, 0, 0], { text: '红苹果' });
  const hits = await store.search([1, 0, 0, 0], 2);
  ok(hits.length === 2, '返回 k 个结果');
  ok(hits[0].id === 'a', '最近邻为 a(苹果)');
  ok(hits[0].score >= hits[1].score, '按相似度降序');
});

// 4. 哈希嵌入（离线确定性）
group('哈希嵌入 Embeddings', () => {
  const e = new HashEmbeddings(64);
  const v1 = e.embed('晨星构建世界顶级 AI');
  const v2 = e.embed('晨星构建世界顶级 AI');
  const v3 = e.embed('今天天气真好');
  ok(math.cosineSimilarity(v1, v2) > 0.999, '相同文本嵌入一致');
  ok(math.cosineSimilarity(v1, v3) < math.cosineSimilarity(v1, v2), '相异文本相似度更低');
  ok(Math.abs(math.norm(v1) - 1) < 1e-9, '嵌入已归一化');
});

// 5. 记忆系统
group('记忆 Memory', async () => {
  const e = new HashEmbeddings(64);
  const store = new VectorStore(64, 'exact');
  const mem = new Memory(e, store);
  await mem.remember('晨星是 AI 开发师', { tag: 'profile' });
  await mem.remember('StellarAI 用 transformers.js', { tag: 'tech' });
  const hits = await mem.recall('AI 开发师是谁', 2);
  ok(hits.length >= 1, '能召回长期记忆');
  ok(hits.some((h) => h.text.includes('晨星')), '召回内容相关');
  ok(mem.getHistory().length >= 2, '短时记忆已记录');
});

// 6. 编排：ReAct 循环（MockLLM 驱动）
group('编排 Engine · ReAct', async () => {
  const llm = new MockLLM([
    'Thought: 先算乘法。\nAction: calculator\nAction Input: (2+3)*4',
    'Thought: 再取时间。\nAction: datetime\nAction Input: ',
    (opts) => `Thought: 已获得结果。\nFinal Answer: 计算结果为20，当前时间见观测：${opts.lastObservation}`,
  ]);
  const { engine } = createStellar({ llm, embeddingsSpec: { type: 'hash' }, vectorBackend: 'exact' });
  const out = await engine.run('计算 (2+3)*4 并告知当前时间');
  ok(out.trace.length >= 2, '至少执行 2 步工具调用');
  ok(out.trace.some((s) => s.action === 'calculator' && s.observation === '20'), 'calculator 被正确调用且观测=20');
  ok(out.trace.some((s) => s.action === 'datetime'), 'datetime 被正确调用');
  ok(out.answer.includes('20'), '最终答案包含工具结果');
  ok(out.answer.includes('当前时间'), '最终答案包含观察观测');
});

// 7. 编排：未知工具容错
group('编排 Engine · 容错', async () => {
  const llm = new MockLLM([
    'Thought: 试着调用不存在的工具。\nAction: nonexistent\nAction Input: x',
    'Thought: 工具不存在，直接回答。\nFinal Answer: 该工具暂不可用。',
  ]);
  const { engine } = createStellar({ llm, embeddingsSpec: { type: 'hash' }, vectorBackend: 'exact' });
  const out = await engine.run('调用不存在的工具');
  ok(out.trace[0].ok === false, '未知工具被标记为失败');
  ok(out.answer.includes('不可用'), '引擎从失败中恢复并给出最终答案');
});

// 8. 端到端：组装一致性
group('端到端 createStellar', async () => {
  const { engine, tools } = createStellar({ embeddingsSpec: { type: 'hash' }, vectorBackend: 'exact' });
  ok(typeof engine.run === 'function', 'engine.run 存在');
  ok(tools.list().length >= 3, '内置工具≥3个');
  const r = await tools.run('calculator', '6*7');
  ok(r.output === '42', 'calculator 6*7=42');
});

// ── 运行 ──────────────────────────────────────────────
(async () => {
  for (const [name, fn] of groups) {
    console.log(`\n▶ ${name}`);
    await fn();
  }
  const total = pass + fail;
  console.log(`\n${'─'.repeat(48)}`);
  console.log(`自检结果：${pass}/${total} 通过` + (fail ? `  ⚠️ ${fail} 项失败` : '  ✅ 全绿'));
  if (fails.length) { console.log('\n失败项：'); fails.forEach((f) => console.log('  · ' + f)); }
  process.exit(fail ? 1 : 0);
})();
