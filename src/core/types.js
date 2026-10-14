// src/core/types.js
// 共享常量与契约。
export const ROLE = { SYSTEM: 'system', USER: 'user', ASSISTANT: 'assistant', TOOL: 'tool' };

// ReAct 协议标签（Yao et al., 2022 —— 世界顶级推理范式）。
export const REACT = {
  THOUGHT: 'Thought',
  ACTION: 'Action',
  ACTION_INPUT: 'Action Input',
  OBSERVATION: 'Observation',
  FINAL_ANSWER: 'Final Answer',
};

// 引擎默认上限，保障可控、可复现。
export const DEFAULTS = {
  maxSteps: 8,
  maxHistoryTokens: 4096,
  topK: 5,
  embeddingDim: 384,
};
