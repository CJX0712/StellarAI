// src/core/engine.js
// 推理编排引擎 —— 本系统的核心增量（集大成之上的"系统设计"）。
// 范式：ReAct（Yao et al., 2022），将推理(Thought)与行动(Action)交织，
// 经工具获取观测(Observation)，循环至 Final Answer。可插拔 LLM / 工具 / 记忆。

import { ROLE, REACT, DEFAULTS } from './types.js';
import { log } from '../utils/logger.js';

export class Engine {
  constructor({ llm, tools, memory, maxSteps = DEFAULTS.maxSteps, verbose = false }) {
    this.llm = llm;
    this.tools = tools;
    this.memory = memory;
    this.maxSteps = maxSteps;
    this.verbose = verbose;
  }

  _systemPrompt() {
    const toolLines = this.tools.list().map((t) => `- ${t.name}: ${t.description}`).join('\n');
    return [
      '你是使用 ReAct（推理+行动）范式的问题求解器。',
      '可用工具：',
      toolLines,
      '',
      '严格按以下格式输出（每次只输出一个 Thought/Action 块）：',
      'Thought: <你的思考>',
      'Action: <工具名>',
      'Action Input: <输入>',
      '——或当已能回答时——',
      'Thought: <思考>',
      'Final Answer: <最终答案>',
    ].join('\n');
  }

  _formatHistory() {
    return this.memory.getHistory().map((m) => {
      const tag = m.role === ROLE.USER ? 'User' : m.role === ROLE.TOOL ? 'Observation' : m.role;
      return `${tag}: ${m.content}`;
    }).join('\n');
  }

  // 解析 LLM 返回中的 ReAct 结构。
  _parse(text) {
    const fa = text.match(/Final\s*Answer\s*:\s*([\s\S]*)/i);
    if (fa) return { finalAnswer: fa[1].trim() };
    const action = text.match(/Action\s*:\s*([^\n]+)/i);
    const actionInput = text.match(/Action\s*Input\s*:\s*([\s\S]*?)(?=\n\s*(?:Thought|Action|Final\s*Answer)\s*:|$)/i);
    const thought = text.match(/Thought\s*:\s*([\s\S]*?)(?=\n\s*(?:Action|Final\s*Answer)\s*:|$)/i);
    if (action) {
      return {
        thought: thought ? thought[1].trim() : '',
        action: action[1].trim(),
        actionInput: actionInput ? actionInput[1].trim() : '',
      };
    }
    // 无结构输出：当作最终答案
    return { finalAnswer: text.trim() };
  }

  async run(query, { useMemory = true } = {}) {
    this.memory.addMessage(ROLE.USER, query);
    const trace = [];
    let context = '';
    if (useMemory) {
      const hits = await this.memory.recall(query, 3);
      if (hits.length) context = '相关记忆：\n' + hits.map((h) => `- ${h.text}`).join('\n');
    }
    let lastObservation = context;
    let answer = '';

    for (let step = 0; step < this.maxSteps; step++) {
      const prompt = [this._systemPrompt(), context ? '\n' + context : '', this._formatHistory(), 'Thought:'].join('\n');
      const raw = await this.llm.generate(prompt, { lastObservation, step, context });
      const parsed = this._parse(raw);
      if (this.verbose) log.info(`step ${step}`, JSON.stringify(parsed).slice(0, 200));

      if (parsed.finalAnswer !== undefined) { answer = parsed.finalAnswer; break; }

      // 执行工具
      const toolName = parsed.action;
      const res = await this.tools.run(toolName, parsed.actionInput);
      const observation = `[${toolName}] ${res.output}`;
      this.memory.addMessage(ROLE.TOOL, observation);
      lastObservation = observation;
      trace.push({
        step, thought: parsed.thought, action: toolName,
        actionInput: parsed.actionInput, observation: res.output, ok: res.ok,
      });
      if (!res.ok) {
        // 工具失败：让模型看到错误并自行调整
        continue;
      }
    }

    if (!answer) answer = lastObservation || '（未能在步数限制内得出结论）';
    this.memory.addMessage(ROLE.ASSISTANT, answer);
    // 长时记忆沉淀
    await this.memory.remember(`Q: ${query} A: ${answer}`);
    return { answer, trace, steps: trace.length };
  }
}
