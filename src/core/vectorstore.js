// src/core/vectorstore.js
// 向量存储与近邻检索。
// 集大成：生产后端复用 hnswlib（Facebook AI Similarity Search，亿级向量 ANN，世界顶级）。
// 优雅降级：hnswlib 不可用时退回纯 JS 精确检索，保证离线/测试可运行可验证。

import { cosineSimilarity, topK } from '../utils/math.js';
import { log } from '../utils/logger.js';

class ExactStore {
  constructor() { this.items = []; }
  async add(id, vector, metadata = {}) { this.items.push({ id, vector, metadata }); }
  async search(vector, k = 5) {
    const scored = this.items.map((it) => ({
      id: it.id, score: cosineSimilarity(vector, it.vector), metadata: it.metadata,
    }));
    return topK(scored.map((s) => s.score), k).map(({ index }) => scored[index]);
  }
  get size() { return this.items.length; }
}

class HnswStore {
  constructor(dim) { this.dim = dim; this.meta = []; this.index = null; }
  async _init() {
    if (this.index) return;
    const { HierarchicalNSW } = await import('hnswlib-node');
    this.index = new HierarchicalNSW('cosine', this.dim);
    this.index.initIndex(Math.max(1000, this.meta.length + 1));
  }
  async add(id, vector, metadata = {}) {
    await this._init();
    const label = this.meta.length;
    this.meta.push({ id, metadata });
    this.index.addPoint(vector, label);
  }
  async search(vector, k = 5) {
    await this._init();
    const { neighbors, distances } = this.index.searchKnn(vector, k);
    return neighbors.map((label, i) => ({
      id: this.meta[label].id,
      score: 1 - distances[i], // hnswlib 余弦距离 = 1 - 相似度
      metadata: this.meta[label].metadata,
    }));
  }
  get size() { return this.meta.length; }
}

export class VectorStore {
  constructor(dim, backend = 'auto') {
    this.dim = dim; this.backendName = backend; this._store = null;
  }
  async _resolve() {
    if (this._store) return this._store;
    if (this.backendName === 'exact') { this._store = new ExactStore(); }
    else if (this.backendName === 'hnsw') { this._store = new HnswStore(this.dim); }
    else { // auto
      try {
        await import('hnswlib-node');
        this._store = new HnswStore(this.dim);
        log.info('VectorStore → hnswlib (world-class ANN)');
      } catch (e) {
        this._store = new ExactStore();
        log.warn('hnswlib 不可用，降级到精确检索（生产建议安装 hnswlib-node）');
      }
    }
    return this._store;
  }
  async add(id, vector, metadata) { return (await this._resolve()).add(id, vector, metadata); }
  async search(vector, k) { return (await this._resolve()).search(vector, k); }
}
