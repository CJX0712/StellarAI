# 晨星·星核 StellarAI

> 站在世界顶级成果之上的**本地优先 · 多模态 AI 引擎**。
> 集成 [Hugging Face transformers.js](https://github.com/huggingface/transformers.js)（SOTA 模型本地推理）+ [hnswlib](https://github.com/nmslib/hnswlib)（Facebook AI 亿级向量 ANN 检索），自研 **ReAct 推理编排 / 工具调用框架 / 长短时记忆**，零 API key、零上传、可运行、可复现、可验证。

<p align="center">
  <img src="https://img.shields.io/badge/后端-transformers.js-blue" />
  <img src="https://img.shields.io/badge/检索-hnswlib-green" />
  <img src="https://img.shields.io/badge/范式-ReAct-orange" />
  <img src="https://img.shields.io/badge/测试-32%2F32%20全绿-7ee787" />
  <img src="https://img.shields.io/badge/署名-晨星-purple" />
  <img src="https://img.shields.io/badge/license-MIT-blue" />
</p>

---

## 一、设计哲学

| 原则 | 落地方式 |
|------|----------|
| **零重写** | 模型推理直接复用 Hugging Face 官方 `transformers.js`；向量检索直接复用 Facebook AI `hnswlib`。不重造轮子。 |
| **集大成** | 每一层都站在世界顶级成果之上：SOTA 模型（transformers.js）+ 世界顶级 ANN（hnswlib）+ 顶级推理范式（ReAct, Yao et al. 2022）。凡有更强现成方案，立即替换自写。 |
| **唯一目标** | 锻造架构最先进、性能最强、代码最优、文档最全的 AI 系统。 |
| **闭环交付** | 完整开源至 GitHub，可运行 / 可复现 / 可验证，署名 **晨星**。 |
| **自治执行** | 依赖缺失时自动优雅降级（向量检索 hnswlib → 精确检索；模型 transformers.js → 离线 hash 嵌入 + mock LLM），不停摆。 |

---

## 二、它站在哪些"世界顶级"之上

- **模型层** — [`@huggingface/transformers`](https://github.com/huggingface/transformers.js)：Hugging Face 官方，浏览器 / Node 端本地运行 BERT、GPT、CLIP、Whisper 等 SOTA 模型，基于 ONNX + WASM/WebGPU。
- **检索层** — [`hnswlib`](https://github.com/nmslib/hnswlib)：Facebook AI Similarity Search，分层可导航小世界图（HNSW），支持亿级向量近似最近邻，业界性能标杆。
- **推理范式** — **ReAct**（*Reasoning + Acting*, Yao et al., 2022）：将"思考"与"行动"交织，是当前 Agent 编排的世界主流范式。
- **数学底座** — 余弦相似度 / softmax / top-k 等取自经典数值方法，并在自检中以**不变量**交叉验证。

---

## 三、架构

```
┌──────────────────────────────────────────────────────────┐
│                       StellarAI 引擎                       │
├──────────────────────────────────────────────────────────┤
│  Engine（ReAct 编排 · 自研核心增量）                        │
│   Thought → Action → Observation → … → Final Answer        │
│      │            │                                         │
│      ▼            ▼                                         │
│  ┌──────────┐  ┌──────────────────────────────────────┐    │
│  │  Tools   │  │  Memory（短时长时记忆）                │    │
│  │ 注册/调用 │  │   短时：对话缓冲(Working Memory)     │    │
│  │ calculator│  │   长时：向量情景记忆(Episodic)       │    │
│  │ datetime │  └───────────────┬──────────────────────┘    │
│  │ knowledge│                  │                           │
│  └──────────┘                  ▼                           │
│                         ┌──────────────┐                  │
│                         │ Embeddings   │  transformers.js  │
│                         │ (SOTA 句向量)│  / hash(离线)     │
│                         └──────┬───────┘                  │
│                                ▼                          │
│                         ┌──────────────┐                  │
│                         │ VectorStore  │  hnswlib(世界顶级)│
│                         │ ANN / 精确    │  / exact(降级)    │
│                         └──────────────┘                  │
└──────────────────────────────────────────────────────────┘
                          ▲
                          │ 浏览器端
                   index.html（单文件 · 真模型推理）
```

---

## 四、特性

- ✅ **本地优先**：模型在你机器/浏览器里跑，零 API key、零数据上传、零费用。
- ✅ **多模态就绪**：文本生成、句向量嵌入、语义检索、情感分类；可扩展视觉/语音（CLIP/Whisper）。
- ✅ **ReAct 编排**：思考—行动—观察闭环，最长时记忆与工具协同求解。
- ✅ **工具框架**：内置安全计算器（递归下降求值，杜绝 `eval` 注入）、时间、知识检索；可任意扩展自定义工具。
- ✅ **长短时记忆**：对话缓冲 + 向量情景记忆，支持语义召回（RAG）。
- ✅ **优雅降级**：重依赖（hnswlib / transformers.js）缺失时自动回退，保证任意环境可运行。
- ✅ **可验证**：32/32 自检全绿，确定性、可复现（见下方"验证"）。
- ✅ **单文件演示**：`index.html` 内联 CSS/JS、零构建，打开即用真实本地 AI。

---

## 五、快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/CJX0712/StellarAI.git
cd StellarAI
npm install            # 轻量：重型依赖为可选，安装秒过
# 若需真实 SOTA 模型推理由 Node 端运行：
npm i @huggingface/transformers hnswlib-node
```

### 2. 跑自检（确定性，离线可验）

```bash
npm test
# ✅ 自检结果：32/32 通过 全绿
```

### 3. 浏览器演示（真实本地 AI）

直接用浏览器打开 **`index.html`**（或 `npm run demo:browser`）。首次会从 CDN 下载模型（约 20–300MB），之后浏览器缓存复用。包含：
- **语义检索**：all-MiniLM-L6-v2 句向量 + 余弦排序
- **文本生成**：distilgpt2 本地生成
- **情感分析**：distilbert-sst2 分类

### 4. Node 端示例

```bash
node examples/chat.js   # ReAct 推理：调用计算器工具
node examples/rag.js    # 记忆增强检索（RAG）流程
```

---

## 六、API 速览

```js
import { createStellar, MockLLM } from './src/index.js';

// 一键装配（离线后端：hash 嵌入 + 精确检索 + mock LLM）
const { engine, memory, tools } = createStellar({
  embeddingsSpec: { type: 'hash' },   // 生产换 { type:'transformers', model:'Xenova/all-MiniLM-L6-v2' }
  vectorBackend:  'exact',            // 生产换 'hnsw'（需 hnswlib-node）
  llmSpec:        { type: 'mock' },   // 生产换 { type:'transformers', model:'Xenova/distilgpt2' }
});

// 记忆增强
await memory.remember('StellarAI 集成 transformers.js 与 hnswlib。');
const hits = await memory.recall('这个引擎用了什么技术', 3);

// 工具
const r = await tools.run('calculator', '6*7');   // => '42'

// 编排（ReAct）
const out = await engine.run('帮我算 123×456');
console.log(out.answer, out.trace);
```

核心模块：`Engine` · `ToolRegistry` · `Memory` · `VectorStore` · `HashEmbeddings`/`TransformersEmbeddings` · `MockLLM`/`TransformersLLM`。

---

## 七、验证（可复现）

自检覆盖 8 大组、32 项断言，全部基于**确定性离线后端**（hash 嵌入 + mock LLM + 精确检索），不依赖模型下载即可证明系统正确性：

| 组 | 验证点 |
|----|--------|
| 数学基元 | dot / 余弦 / 归一化 / topK / argmax / softmax |
| 安全算术 | 运算符优先级、括号、幂、一元负号、**注入防护** |
| 向量检索 | k 近邻、最近邻正确、相似度降序 |
| 哈希嵌入 | 同句一致、异句更疏、已归一化 |
| 记忆系统 | 长时召回、短时记录 |
| ReAct 编排 | 工具调用、观测捕获、**最终答案含工具结果** |
| 容错 | 未知工具标记失败并恢复 |
| 端到端 | 装配一致性、calculator 6×7=42 |

---

## 八、目录结构

```
StellarAI/
├── index.html              # 单文件浏览器演示（真模型推理，零构建）
├── src/
│   ├── index.js            # 公共 API + createStellar 装配工厂
│   ├── core/
│   │   ├── engine.js       # ★ ReAct 推理编排（核心创新）
│   │   ├── tools.js        # 工具框架 + 安全计算器
│   │   ├── memory.js       # 长短时记忆
│   │   ├── vectorstore.js  # hnswlib / 精确检索（优雅降级）
│   │   ├── embeddings.js   # transformers.js / hash 嵌入
│   │   ├── llm.js          # transformers.js / mock LLM
│   │   └── types.js        # 共享契约
│   └── utils/
│       ├── math.js         # 向量与数值基元
│       └── logger.js       # 分级日志
├── test/run.js             # 自检套件（32/32 全绿）
├── examples/               # chat.js / rag.js 用法示例
├── package.json
├── LICENSE                 # MIT
└── README.md
```

---

## 九、路线图

- [ ] 多模态扩展：CLIP 图文检索、Whisper 语音识别
- [ ] 生产级向量索引持久化（hnswlib 落盘 / 增量构建）
- [ ] 流式输出（streaming）与工具并行调用
- [ ] 规划器（Plan-and-Execute）与自我反思（Reflexion）
- [ ] 评测基准（MMLU / GSM8K 本地化跑分）

---

## 十、许可证与署名

- **许可证**：MIT
- **作者**：晨星（Morning Star）
- **仓库**：https://github.com/CJX0712/StellarAI

> 集世界顶级之大成，铸本地优先之星核。✦
