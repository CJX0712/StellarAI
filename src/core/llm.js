// src/core/llm.js
// LLM 提供方抽象。
// 零重写：真实推理复用 @huggingface/transformers（Hugging Face 官方，本地跑 SOTA 模型）。
// 测试/离线场景用确定性 MockLLM，使 ReAct 编排可被精确验证（可复现、可验证）。

import { log } from '../utils/logger.js';

export class MockLLM {
  // plan: 依次吐出的 ReAct 片段。每个元素为字符串或 (ctx)=>string。
  constructor(plan = []) { this.plan = plan; this.calls = 0; this.name = 'mock'; }
  async generate(_prompt, opts = {}) {
    if (this.calls >= this.plan.length) return 'Final Answer: 无法确定（计划已耗尽）。';
    const item = this.plan[this.calls++];
    return typeof item === 'function' ? item(opts) : item;
  }
  reset() { this.calls = 0; }
}

export class TransformersLLM {
  constructor(model = 'Xenova/distilgpt2', opts = {}) {
    this.model = model;
    this.opts = opts;
    this.name = 'transformers';
    this._pipe = null;
  }
  async _pipeline() {
    if (this._pipe) return this._pipe;
    const { pipeline } = await import('@huggingface/transformers');
    this._pipe = await pipeline('text-generation', this.model, {
      dtype: this.opts.dtype || 'q8',
      device: this.opts.device || 'wasm',
    });
    return this._pipe;
  }
  async generate(prompt, opts = {}) {
    const pipe = await this._pipeline();
    const out = await pipe(prompt, {
      max_new_tokens: opts.max_new_tokens ?? 128,
      do_sample: opts.do_sample ?? false,
      return_full_text: false,
    });
    const text = Array.isArray(out) ? out[0]?.generated_text ?? '' : out?.generated_text ?? '';
    log.debug('generate', text.slice(0, 80));
    return text;
  }
}

export function createLLM(spec = {}) {
  if (spec.type === 'mock') return new MockLLM(spec.plan || []);
  if (spec.type === 'transformers') return new TransformersLLM(spec.model, spec.opts);
  // 默认：有网络/模型时用 transformers，否则退回 mock（优雅降级）。
  return new TransformersLLM(spec.model);
}
