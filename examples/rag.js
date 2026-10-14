// examples/rag.js
// 演示：记忆增强检索（RAG）流程 —— 把知识写入长时记忆，再按语义召回并作为上下文作答。
// 离线用 hash 嵌入即可验证链路；生产把 embeddingsSpec 换成 transformers 的 all-MiniLM-L6-v2 即获 SOTA 语义表征。

import { createStellar, MockLLM } from '../src/index.js';

// 同一引擎实例贯穿"记忆写入 → 召回 → 作答"，确保上下文一致。
const llm = new MockLLM([
  (opts) => `Thought: 已从记忆检索到相关上下文。\nFinal Answer: ${opts.context || '无相关记忆'}`,
]);
const { engine, memory } = createStellar({ llm, embeddingsSpec: { type: 'hash' }, vectorBackend: 'exact' });

await memory.remember('StellarAI 由晨星构建，集成 transformers.js 与 hnswlib。');
await memory.remember('ReAct 是推理与行动交织的 Agent 范式。');
await memory.remember('hnswlib 支持亿级向量的近似最近邻检索。');

const query = 'StellarAI 是否使用了 hnswlib 做向量检索？';
const hits = await memory.recall(query, 3);
console.log('语义召回结果：');
for (const h of hits) console.log(`  · (${(h.score * 100).toFixed(1)}%) ${h.text}`);

const out = await engine.run(query);
console.log('\n引擎回答（记忆增强）：\n' + out.answer);
