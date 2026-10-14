// src/core/memory.js
// 记忆系统。
// · 短时记忆：对话缓冲（Working Memory），受 token 预算约束。
// · 长时记忆：情景记忆（Episodic），经 Embeddings 写入 VectorStore，支持语义召回。

import { ROLE } from './types.js';

export class Memory {
  constructor(embeddings, vectorStore, opts = {}) {
    this.emb = embeddings;
    this.store = vectorStore;
    this.history = [];
    this.maxTokens = opts.maxTokens || 4096;
    this._seq = 0;
  }
  addMessage(role, content) {
    this.history.push({ role, content, ts: Date.now() });
    return this;
  }
  async remember(text, metadata = {}) {
    const vec = await this.emb.embed(text);
    const id = `mem_${this._seq++}`;
    await this.store.add(id, vec, { ...metadata, text });
    this.addMessage(ROLE.ASSISTANT, `[记忆] ${text}`);
    return id;
  }
  async recall(query, k = 5) {
    const qv = await this.emb.embed(query);
    const hits = await this.store.search(qv, k);
    return hits
      .filter((h) => h.score > 0.01)
      .map((h) => ({ text: h.metadata.text, score: h.score }));
  }
  getHistory() { return this.history; }
  estimateTokens() { return this.history.reduce((a, m) => a + Math.ceil(m.content.length / 2), 0); }
  trim() {
    while (this.estimateTokens() > this.maxTokens && this.history.length > 1) this.history.shift();
  }
  clear() { this.history = []; }
}
