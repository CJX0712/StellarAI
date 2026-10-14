// src/core/embeddings.js
// 向量化提供方抽象。
// 零重写：真实句向量复用 @huggingface/transformers 的 feature-extraction（SOTA 语义表征）。
// 离线/测试用确定性 HashEmbeddings（哈希技巧，类 scikit-learn HashingVectorizer），
// 使向量检索与 RAG 逻辑可在无网络、无模型下被精确验证。

import { DEFAULTS } from './types.js';
import { normalize } from '../utils/math.js';
import { log } from '../utils/logger.js';

function stableHash(str) {
  // FNV-1a 32-bit，确定性跨平台。
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export class HashEmbeddings {
  constructor(dim = DEFAULTS.embeddingDim) { this.dim = dim; this.name = 'hash'; }
  embed(text) {
    const vec = new Array(this.dim).fill(0);
    const tokens = String(text).toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
    for (const t of tokens) {
      const h = stableHash(t);
      const idx = h % this.dim;
      vec[idx] += (h & 1) ? 1 : -1; // 符号携带词序无关特征
    }
    return normalize(vec);
  }
  embedBatch(texts) { return texts.map((t) => this.embed(t)); }
}

export class TransformersEmbeddings {
  constructor(model = 'Xenova/all-MiniLM-L6-v2', dim = 384) {
    this.model = model; this.dim = dim; this.name = 'transformers'; this._pipe = null;
  }
  async _pipeline() {
    if (this._pipe) return this._pipe;
    const { pipeline } = await import('@huggingface/transformers');
    this._pipe = await pipeline('feature-extraction', this.model, { dtype: 'q8', device: 'wasm' });
    return this._pipe;
  }
  async _one(text) {
    const pipe = await this._pipeline();
    const out = await pipe(text, { pooling: 'mean', normalize: true });
    return Array.from(out.data);
  }
  async embed(text) {
    log.debug('embed', String(text).slice(0, 40));
    return this._one(text);
  }
  async embedBatch(texts) {
    const pipe = await this._pipeline();
    const out = await pipe(texts, { pooling: 'mean', normalize: true });
    const data = Array.from(out.data);
    const per = data.length / texts.length;
    return Array.from({ length: texts.length }, (_, i) => data.slice(i * per, i * per + per));
  }
}

export function createEmbeddings(spec = {}) {
  if (spec.type === 'hash') return new HashEmbeddings(spec.dim);
  if (spec.type === 'transformers') return new TransformersEmbeddings(spec.model, spec.dim);
  return new HashEmbeddings(); // 默认离线可验证
}
