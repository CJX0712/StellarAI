// examples/chat.js
// 演示：用晨星·星核引擎跑一次 ReAct 推理（默认离线 mock 后端，开箱即跑）。
// 想跑真实 SOTA 模型时：把 llmSpec 换成 { type:'transformers', model:'Xenova/distilgpt2' }，
// 并确保已 `npm i @huggingface/transformers`（或直接使用 index.html 浏览器演示）。

import { createStellar, MockLLM } from '../src/index.js';

const llm = new MockLLM([
  'Thought: 用户想算个乘法。\nAction: calculator\nAction Input: 123*456',
  'Thought: 已得到结果，给出最终回答。\nFinal Answer: 123×456 = 56088。',
]);

const { engine, tools } = createStellar({ llm, embeddingsSpec: { type: 'hash' }, vectorBackend: 'exact' });

const out = await engine.run('帮我算一下 123 乘以 456');
console.log('答案：', out.answer);
console.log('工具轨迹：');
for (const s of out.trace) console.log(`  · ${s.action}(${JSON.stringify(s.actionInput)}) => ${s.observation}`);
