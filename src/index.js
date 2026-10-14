// src/index.js
// 晨星·星核 StellarAI —— 公共 API 入口。
// 集大成：世界顶级模型(transformers.js) + 世界顶级向量检索(hnswlib) + 自研编排/工具/记忆。

import { Logger, log } from './utils/logger.js';
import * as math from './utils/math.js';
import { ROLE, REACT, DEFAULTS } from './core/types.js';
import { MockLLM, TransformersLLM, createLLM } from './core/llm.js';
import { HashEmbeddings, TransformersEmbeddings, createEmbeddings } from './core/embeddings.js';
import { VectorStore } from './core/vectorstore.js';
import { ToolRegistry, safeEval } from './core/tools.js';
import { Memory } from './core/memory.js';
import { Engine } from './core/engine.js';

export {
  Logger, log, math, ROLE, REACT, DEFAULTS,
  MockLLM, TransformersLLM, createLLM,
  HashEmbeddings, TransformersEmbeddings, createEmbeddings,
  VectorStore, ToolRegistry, safeEval,
  Memory, Engine,
};

// 一键装配：把世界顶级组件与自研系统层编排成一个可用引擎。
export function createStellar(cfg = {}) {
  const dim = cfg.embeddingDim || DEFAULTS.embeddingDim;
  const embeddings = cfg.embeddings || createEmbeddings(cfg.embeddingsSpec || { type: 'hash' });
  const vectorStore = cfg.vectorStore || new VectorStore(dim, cfg.vectorBackend || 'auto');
  const tools = cfg.tools || new ToolRegistry();
  const memory = cfg.memory || new Memory(embeddings, vectorStore);
  const llm = cfg.llm || createLLM(cfg.llmSpec || { type: 'mock' });
  const engine = new Engine({ llm, tools, memory, maxSteps: cfg.maxSteps, verbose: cfg.verbose });
  return { engine, memory, tools, embeddings, vectorStore, llm };
}
