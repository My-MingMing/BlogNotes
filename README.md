# LLM / Agent 笔记 — 主题阅读指南

> 本仓库已离线整理 **141 篇**（2025–2026）笔记，全部图片本地化，可断网阅读。  
> 完整按时间排列的清单见 **[完整目录.md](<完整目录.md>)**；本文件是按知识体系组织的**推荐阅读路线**。

这位博主的笔记有一个鲜明特点：**先用一个 36 篇的系列把大模型底层原理讲透，再逐个精读业界主流 Agent 框架的源码**。
因此它既适合补理论基础，也适合工程师对照真实代码库学习 Agent 系统设计。下面我按研究/工程视角把它重新组织成若干主题，
每个主题标注了**适合人群、前置知识和阅读顺序**。

---

## 🗺️ 总体学习路线

```
基础理论                  推理与部署优化              系统/底层
┌─────────────┐          ┌──────────────┐          ┌────────────┐
│ Transformer │  ──────▶ │ KVCache/量化  │ ──────▶ │ GPU Kernel │
│  原理(1-17) │          │ 投机解码/MoE  │          │ MPK/Kernel │
└─────────────┘          └──────────────┘          └────────────┘
       │                                                   │
       ▼                                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     Agent 框架源码精读                          │
│  OpenHands · Nanobot · Flink Agents · GUI-Agent · KernelAgent  │
└──────────────────────────────────────────────────────────────┘
       │                          │                        │
       ▼                          ▼                        ▼
  Agent 记忆系统            Agentic RL / OPD          Agent 架构理论
  MemOS · MemGen · ACE     OpenClaw-RL · Uni-Agent   17模式 · Agent OS
```

**建议三条路径**：
- 🟢 **入门补基础**：主题 1 → 2 → 3（吃透 Transformer 与推理优化）
- 🔵 **推理/部署工程师**：主题 2（KV Cache / 量化 / 投机解码）→ 主题 4（GPU Kernel）
- 🟣 **Agent 方向**：主题 5（先读 1~2 个框架源码）→ 6（记忆）→ 7（Agentic RL）→ 8（架构理论）

---

## 主题 1 · 大模型基础：Transformer 原理（必读地基）

> **适合**：所有人，尤其是想从零建立 LLM 心智模型的读者。**前置**：会 Python、懂基本神经网络即可。  
> 这是全博客最系统的部分——从注意力一路讲到工程实现，建议**按顺序通读**。

**1.1 核心架构（先读这一组，建立主干）**
- [探秘Transformer系列之（1）：注意力机制](<notes/2025-02-09_探秘Transformer系列之（1）：注意力机制/index.md>)
- [探秘Transformer系列之（2）---总体架构](<notes/2025-02-15_探秘Transformer系列之（2）---总体架构/index.md>)
- [探秘Transformer系列之（3）---数据处理](<notes/2025-02-18_探秘Transformer系列之（3）---数据处理/index.md>)
- [探秘Transformer系列之（6）--- token](<notes/2025-02-24_探秘Transformer系列之（6）--- token/index.md>)
- [探秘Transformer系列之（7）--- embedding](<notes/2025-02-27_探秘Transformer系列之（7）--- embedding/index.md>)
- [探秘Transformer系列之（10）--- 自注意力](<notes/2025-03-05_探秘Transformer系列之（10）--- 自注意力/index.md>)
- [探秘Transformer系列之（11）--- 掩码](<notes/2025-03-08_探秘Transformer系列之（11）--- 掩码/index.md>)
- [探秘Transformer系列之（12）--- 多头自注意力](<notes/2025-03-11_探秘Transformer系列之（12）--- 多头自注意力/index.md>)
- [探秘Transformer系列之（13）--- FFN](<notes/2025-03-14_探秘Transformer系列之（13）--- FFN/index.md>)
- [探秘Transformer系列之（14）--- 残差网络和归一化](<notes/2025-03-16_探秘Transformer系列之（14）--- 残差网络和归一化/index.md>)
- [探秘Transformer系列之（4）--- 编码器 & 解码器](<notes/2025-02-20_探秘Transformer系列之（4）--- 编码器 & 解码器/index.md>)
- [探秘Transformer系列之（5）--- 训练&推理](<notes/2025-02-22_探秘Transformer系列之（5）--- 训练&推理/index.md>)
- [探秘Transformer系列之（15）--- 采样和输出](<notes/2025-03-18_探秘Transformer系列之（15）--- 采样和输出/index.md>)
- [探秘Transformer系列之（16）--- 资源占用](<notes/2025-03-21_探秘Transformer系列之（16）--- 资源占用/index.md>)

**1.2 位置编码（单独拎出来，面试高频）**
- [探秘Transformer之（8）--- 位置编码](<notes/2025-03-01_探秘Transformer之（8）--- 位置编码/index.md>)
- [探秘Transformer系列之（9）--- 位置编码分类](<notes/2025-03-03_探秘Transformer系列之（9）--- 位置编码分类/index.md>)
- [探秘Transformer系列之（17）--- RoPE](<notes/2025-03-23_探秘Transformer系列之（17）--- RoPE/index.md>)
- [探秘Transformer系列之（23）--- 长度外推](<notes/2025-04-05_探秘Transformer系列之（23）--- 长度外推/index.md>)
> 17=RoPE，23=长度外推，是当下长上下文模型的基础。

**1.3 系列总览（索引贴）**
- [探秘Transformer系列之文章列表](<notes/2025-03-21_探秘Transformer系列之文章列表/index.md>)

## 主题 2 · 推理与部署优化（工程核心）

> **适合**：做推理加速、模型部署、降本的工程师。**前置**：主题 1。  
> 这部分把「为什么大模型推理慢/费显存、业界怎么解决」讲得很完整。

**2.1 注意力与显存优化（FlashAttention）**
- [探秘Transformer系列之（18）--- FlashAttention](<notes/2025-03-25_探秘Transformer系列之（18）--- FlashAttention/index.md>)
- [探秘Transformer系列之（19）----FlashAttention V2 及升级版本](<notes/2025-03-28_探秘Transformer系列之（19）----FlashAttention V2 及升级版本/index.md>)

**2.2 ⭐ KV Cache 优化（博主着墨最多，强烈推荐）**
- [探秘Transformer系列之（20）--- KV Cache](<notes/2025-03-30_探秘Transformer系列之（20）--- KV Cache/index.md>)
- [探秘Transformer系列之（24）--- KV Cache优化](<notes/2025-04-08_探秘Transformer系列之（24）--- KV Cache优化/index.md>)
- [探秘Transformer系列之（25）--- KV Cache优化之处理长文本序列](<notes/2025-04-10_探秘Transformer系列之（25）--- KV Cache优化之处理长文本序列/index.md>)
- [探秘Transformer系列之（26）--- KV Cache优化---分离or合并](<notes/2025-04-12_探秘Transformer系列之（26）--- KV Cache优化---分离or合并/index.md>)
- [探秘Transformer系列之（27）--- MQA & GQA](<notes/2025-04-14_探秘Transformer系列之（27）--- MQA & GQA/index.md>)
- [探秘Transformer系列之（28）--- DeepSeek MLA](<notes/2025-04-17_探秘Transformer系列之（28）--- DeepSeek MLA/index.md>)
> 主线：KV Cache 是什么(20) → 通用优化(24) → 长文本(25) → 分离/合并(26) → 用结构减小：MQA&GQA(27) → DeepSeek MLA(28)。
> 想专攻长上下文/省显存，这一组 + 1.2 位置编码就是完整拼图。

**2.3 投机解码 / 解码加速**
- [探秘Transformer系列之（30）--- 投机解码](<notes/2025-04-23_探秘Transformer系列之（30）--- 投机解码/index.md>)
- [探秘Transformer系列之（31）--- Medusa](<notes/2025-04-28_探秘Transformer系列之（31）--- Medusa/index.md>)
- [探秘Transformer系列之（32）--- Lookahead Decoding](<notes/2025-05-10_探秘Transformer系列之（32）--- Lookahead Decoding/index.md>)
- [探秘Transformer系列之（33）--- DeepSeek MTP](<notes/2025-05-17_探秘Transformer系列之（33）--- DeepSeek MTP/index.md>)
> 投机解码(30) 总览 → Medusa(31) → Lookahead(32) → DeepSeek MTP(33)。

**2.4 模型量化**
- [探秘Transformer系列之（34）--- 量化基础](<notes/2025-05-24_探秘Transformer系列之（34）--- 量化基础/index.md>)
- [探秘Transformer系列之（35）--- 大模型量化基础](<notes/2025-06-02_探秘Transformer系列之（35）--- 大模型量化基础/index.md>)
- [探秘Transformer系列之（36）--- 大模型量化方案](<notes/2025-06-08_探秘Transformer系列之（36）--- 大模型量化方案/index.md>)
> 量化基础(34/35) → 实际量化方案(36)，部署降本必备。

## 主题 3 · 模型结构与高效微调

> **适合**：关注模型扩展性、训练成本的读者。

**3.1 MoE 混合专家**
- [探秘Transformer系列之（21）--- MoE](<notes/2025-03-31_探秘Transformer系列之（21）--- MoE/index.md>)
- [探秘Transformer系列之（29）--- DeepSeek MoE](<notes/2025-04-20_探秘Transformer系列之（29）--- DeepSeek MoE/index.md>)
> MoE 原理(21) → DeepSeek MoE(29) 工业级实现。

**3.2 参数高效微调（PEFT）**
- [探秘Transformer系列之（22）--- LoRA](<notes/2025-04-03_探秘Transformer系列之（22）--- LoRA/index.md>)

## 主题 4 · GPU Kernel 与系统底层

> **适合**：CUDA/编译器/推理引擎方向，想理解「模型如何真正跑在 GPU 上」。**前置**：主题 2 有帮助。

**4.1 MPK（Mirage Persistent Kernel）源码——把模型编译成单个持久化 kernel**
- [MPK（Mirage Persistent Kernel）源码笔记（1）--- 基础原理](<notes/2025-10-23_MPK（Mirage Persistent Kernel）源码笔记（1）--- 基础原理/index.md>)
- [MPK（Mirage Persistent Kernel）源码笔记（2）--- 多层结构化图模型](<notes/2025-10-26_MPK（Mirage Persistent Kernel）源码笔记（2）--- 多层结构化图模型/index.md>)
- [MPK（Mirage Persistent Kernel）源码笔记（3）--- 系统接口](<notes/2025-10-29_MPK（Mirage Persistent Kernel）源码笔记（3）--- 系统接口/index.md>)
- [MPK（Mirage Persistent Kernel）源码笔记（4）--- 转译系统](<notes/2025-10-31_MPK（Mirage Persistent Kernel）源码笔记（4）--- 转译系统/index.md>)
- [MPK（Mirage Persistent Kernel）源码笔记（5）--- 执行引擎](<notes/2025-11-02_MPK（Mirage Persistent Kernel）源码笔记（5）--- 执行引擎/index.md>)

**4.2 PyTorch KernelAgent 源码——用 Agent 自动生成 GPU kernel（系统 × Agent 交叉）**
- [PyTorch KernelAgent 源码解读 ---（1）--- 原理](<notes/2026-05-11_PyTorch KernelAgent 源码解读 ---（1）--- 原理/index.md>)
- [PyTorch KernelAgent 源码解读 ---（2）--- 总体流程](<notes/2026-05-13_PyTorch KernelAgent 源码解读 ---（2）--- 总体流程/index.md>)
- [PyTorch KernelAgent 源码解读 ---（3）--- orchestrator](<notes/2026-05-16_PyTorch KernelAgent 源码解读 ---（3）--- orchestrator/index.md>)
- [PyTorch KernelAgent 源码解读 ---（4）--- ExtractorAgent](<notes/2026-05-18_PyTorch KernelAgent 源码解读 ---（4）--- ExtractorAgent/index.md>)
- [PyTorch KernelAgent 源码解读 ---（5）--- Dispatcher](<notes/2026-05-20_PyTorch KernelAgent 源码解读 ---（5）--- Dispatcher/index.md>)
- [PyTorch KernelAgent 源码解读 ---（6）--- Composer](<notes/2026-05-22_PyTorch KernelAgent 源码解读 ---（6）--- Composer/index.md>)

## 主题 5 · Agent 框架源码精读（博主下半场主线）

> **适合**：要从零搭 Agent、或读懂主流 Agent 框架的工程师。**前置**：了解 LLM tool-calling 即可。  
> 这几套都是**完整逐文件源码走读**，价值极高。建议**先挑一套读完**再横向对比，不要并行。

**5.1 OpenHands —— 编码类 Agent 的标杆（14 篇，最完整）**
- [AI Agent框架探秘：拆解 OpenHands（1）--- 核心理念](<notes/2026-01-19_AI Agent框架探秘：拆解 OpenHands（1）--- 核心理念/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（2）--- CodeAct论文](<notes/2026-01-21_AI Agent框架探秘：拆解 OpenHands（2）--- CodeAct论文/index.md>)
- [AI Agent 框架探秘：拆解 OpenHands（3）--- 启动](<notes/2026-01-27_AI Agent 框架探秘：拆解 OpenHands（3）--- 启动/index.md>)
- [AI Agent 框架探秘：拆解 OpenHands（4）--- 服务](<notes/2026-01-29_AI Agent 框架探秘：拆解 OpenHands（4）--- 服务/index.md>)
- [AI Agent 框架探秘：拆解 OpenHands（5）--- 交互&会话](<notes/2026-02-02_AI Agent 框架探秘：拆解 OpenHands（5）--- 交互&会话/index.md>)
- [AI Agent 框架探秘：拆解 OpenHands（6）--- 事件系统](<notes/2026-02-04_AI Agent 框架探秘：拆解 OpenHands（6）--- 事件系统/index.md>)
- [AI Agent 框架探秘：拆解 OpenHands（7）---  Agent](<notes/2026-02-23_AI Agent 框架探秘：拆解 OpenHands（7）--- Agent/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（8）---  CodeActAgent](<notes/2026-02-25_AI Agent框架探秘：拆解 OpenHands（8）--- CodeActAgent/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（9）---  AgentController](<notes/2026-02-27_AI Agent框架探秘：拆解 OpenHands（9）--- AgentController/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（10）---  Runtime](<notes/2026-03-02_AI Agent框架探秘：拆解 OpenHands（10）--- Runtime/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（11）---  Runtime主要组件](<notes/2026-03-05_AI Agent框架探秘：拆解 OpenHands（11）--- Runtime主要组件/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（12）--- Function call](<notes/2026-03-09_AI Agent框架探秘：拆解 OpenHands（12）--- Function call/index.md>)
- [AI Agent框架探秘：拆解 OpenHands（13）--- Memory](<notes/2026-03-11_AI Agent框架探秘：拆解 OpenHands（13）--- Memory/index.md>)
- [AI Agent 框架探秘：拆解 OpenHands（14）--- Microagents](<notes/2026-03-13_AI Agent 框架探秘：拆解 OpenHands（14）--- Microagents/index.md>)
> 推荐先读：核心理念(1) → CodeAct论文(2) → Agent(7) → CodeActAgent(8) → AgentController(9) → Runtime(10/11)。

**5.2 Nanobot / OpenClaw —— 轻量级 Agent 架构（10 篇，适合通读建立全貌）**
- [【OpenClaw】通过 Nanobot 源码学习架构---（1）总体](<notes/2026-03-30_【OpenClaw】通过 Nanobot 源码学习架构---（1）总体/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（2）外层控制逻辑](<notes/2026-04-01_【OpenClaw】通过 Nanobot 源码学习架构---（2）外层控制逻辑/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（3）AgentLoop](<notes/2026-04-03_【OpenClaw】通过 Nanobot 源码学习架构---（3）AgentLoop/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构 ---（4）SubAgent](<notes/2026-04-06_【OpenClaw】通过 Nanobot 源码学习架构 ---（4）SubAgent/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（5）Context](<notes/2026-04-08_【OpenClaw】通过 Nanobot 源码学习架构---（5）Context/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（6）Skills](<notes/2026-04-09_【OpenClaw】通过 Nanobot 源码学习架构---（6）Skills/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（7）Memory](<notes/2026-04-13_【OpenClaw】通过 Nanobot 源码学习架构---（7）Memory/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（8）Tools](<notes/2026-04-22_【OpenClaw】通过 Nanobot 源码学习架构---（8）Tools/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（9）周期性执行](<notes/2026-04-24_【OpenClaw】通过 Nanobot 源码学习架构---（9）周期性执行/index.md>)
- [【OpenClaw】通过 Nanobot 源码学习架构---（10）Heartbeat](<notes/2026-04-27_【OpenClaw】通过 Nanobot 源码学习架构---（10）Heartbeat/index.md>)

**5.3 Flink Agents —— 数据流 + Agent 的结合（7 篇，流式/大数据背景读者优先）**
- [【大数据 & AI】Flink Agents 源码解读 --- (1) ---  设计](<notes/2025-12-24_【大数据 & AI】Flink Agents 源码解读 --- (1) --- 设计/index.md>)
- [【大数据 & AI】Flink Agents 源码解读 --- (2) ---  核心架构](<notes/2025-12-29_【大数据 & AI】Flink Agents 源码解读 --- (2) --- 核心架构/index.md>)
- [【大数据 & AI】Flink Agents 源码解读 --- (3) --- Agent](<notes/2025-12-31_【大数据 & AI】Flink Agents 源码解读 --- (3) --- Agent/index.md>)
- [【大数据 & AI】Flink Agents 源码解读 --- (4) ---  AgentPlan](<notes/2026-01-05_【大数据 & AI】Flink Agents 源码解读 --- (4) --- AgentPlan/index.md>)
- [【大数据 & AI】Flink Agents 源码解读 --- (5) ---  ActionExecutionOperator](<notes/2026-01-08_【大数据 & AI】Flink Agents 源码解读 --- (5) --- ActionExecutionOperator/index.md>)
- [【大数据 & AI】Flink Agents 源码解读 --- (6) ---  ActionTask](<notes/2026-01-12_【大数据 & AI】Flink Agents 源码解读 --- (6) --- ActionTask/index.md>)
- [【大数据 & AI】Flink Agents 源码解读 --- (7) ---  AgentsExecutionEnvironment](<notes/2026-01-15_【大数据 & AI】Flink Agents 源码解读 --- (7) --- AgentsExecutionEnvironment/index.md>)

## 主题 6 · Agent 记忆系统（Memory）

> **适合**：研究长期记忆、上下文工程、RAG 进阶的读者。当前最前沿、最有研究价值的方向之一。

**6.1 MemOS —— 把记忆当操作系统来管理（7 篇，体系最完整）**
- [【Agent】MemOS 源码笔记---(1)--基本概念](<notes/2025-11-19_【Agent】MemOS 源码笔记---(1)--基本概念/index.md>)
- [【Agent】MemOS 源码笔记---(2)---TreeTextMemory](<notes/2025-12-01_【Agent】MemOS 源码笔记---(2)---TreeTextMemory/index.md>)
- [【Agent】MemOS 源码笔记---(3)---搜索](<notes/2025-12-08_【Agent】MemOS 源码笔记---(3)---搜索/index.md>)
- [【Agent】MemOS 源码笔记---(4)---KV Cache](<notes/2025-12-10_【Agent】MemOS 源码笔记---(4)---KV Cache/index.md>)
- [【Agent】MemOS 源码笔记---(5)---记忆分类](<notes/2025-12-15_【Agent】MemOS 源码笔记---(5)---记忆分类/index.md>)
- [【Agent】MemOS 源码笔记---(6)---MemScheduler -- 总体](<notes/2025-12-18_【Agent】MemOS 源码笔记---(6)---MemScheduler -- 总体/index.md>)
- [【Agent】MemOS 源码笔记---(7)---MemScheduler 细节](<notes/2025-12-22_【Agent】MemOS 源码笔记---(7)---MemScheduler 细节/index.md>)

**6.2 ACE —— Agentic Context Engineering（论文 + 源码）**
- [[Agent] ACE（Agentic Context Engineering）和Dynamic Cheatsheet学习笔记](<notes/2025-10-19_[Agent] ACE（Agentic Context Engineering）和Dynamic Cheatsheet学习笔记/index.md>)
- [[Agent] ACE（Agentic Context Engineering）源码阅读笔记---（1）基础模块](<notes/2025-11-04_[Agent] ACE（Agentic Context Engineering）源码阅读笔记---（1）基础模块/index.md>)
- [【Agent】 ACE（Agentic Context Engineering）源码阅读笔记 ---（2）--- 训练](<notes/2025-11-05_【Agent】 ACE（Agentic Context Engineering）源码阅读笔记 ---（2）--- 训练/index.md>)
- [【Agent】 ACE（Agentic Context Engineering）源码阅读笔记---（3）关键创新](<notes/2025-11-06_【Agent】 ACE（Agentic Context Engineering）源码阅读笔记---（3）关键创新/index.md>)

**6.3 MemGen —— 生成式隐式记忆**
- [【Agent】生成式隐式记忆 MemGen 源码解读](<notes/2025-11-10_【Agent】生成式隐式记忆 MemGen 源码解读/index.md>)

## 主题 7 · Agentic RL / 强化学习

> **适合**：关注用 RL 训练 Agent、On-Policy Distillation 的研究者。**前置**：主题 5 中任一框架 + RL 基础。

**7.1 OpenClaw-RL —— Agentic RL 与 On-Policy Distillation 源码（6 篇）**
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (1)---基础](<notes/2026-05-25_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (1)---基础/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (2)--- On-Policy Distillation](<notes/2026-05-27_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (2)--- On-Policy Distillation/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (3)--- 总体思考](<notes/2026-05-28_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (3)--- 总体思考/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (4)--- 架构](<notes/2026-05-30_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (4)--- 架构/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (5)--- 异步处理](<notes/2026-06-01_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (5)--- 异步处理/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (6)--- Rollout](<notes/2026-06-18_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (6)--- Rollout/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (7)--- Policy Serving](<notes/2026-07-14_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (7)--- Policy Serving/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (8)--- Environment](<notes/2026-07-16_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (8)--- Environment/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (9)--- Reward Judging](<notes/2026-07-20_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (9)--- Reward Judging/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (10)--- PRM](<notes/2026-07-22_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (10)--- PRM/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (11)--- 算法总体实现](<notes/2026-07-23_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (11)--- 算法总体实现/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (12)--- GRPO](<notes/2026-07-27_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (12)--- GRPO/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (13)--- OPD实现](<notes/2026-07-28_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (13)--- OPD实现/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (14)--- Teacher](<notes/2026-07-29_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (14)--- Teacher/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (15)--- Combine模式](<notes/2026-08-03_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (15)--- Combine模式/index.md>)
- [【Agentic RL / 强化学习 / OPD】OpenClaw-RL 源码阅读笔记 --- (16)--- AReal](<notes/2026-08-04_【Agentic RL 强化学习 OPD】OpenClaw-RL 源码阅读笔记 --- (16)--- AReal/index.md>)

**7.2 Uni-Agent · Miles —— 强化学习框架深度分析**
- [【Agentic RL / 强化学习框架】Uni-Agent 深度技术分析（1）--- 总体](<notes/2026-06-09_【Agentic RL 强化学习框架】Uni-Agent 深度技术分析（1）--- 总体/index.md>)
- [【强化学习框架】Uni-Agent 深度技术分析（2）--- 关键技术](<notes/2026-06-11_【强化学习框架】Uni-Agent 深度技术分析（2）--- 关键技术/index.md>)
- [【Agentic RL / 强化学习框架】Miles 项目技术分析---（1）--- 总体](<notes/2026-06-15_【Agentic RL 强化学习框架】Miles 项目技术分析---（1）--- 总体/index.md>)
- [【Agentic RL / 强化学习框架】Miles 项目技术分析---（2）--- 关键技术](<notes/2026-06-16_【Agentic RL 强化学习框架】Miles 项目技术分析---（2）--- 关键技术/index.md>)

**7.3 SERL / HIL-SERL —— 真机强化学习 · 具身智能（新增系列）**
- [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（1）全景篇](<notes/2026-06-22_【机器人 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（1）全景篇/index.md>)
- [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ---- （2）算法篇（SAC）](<notes/2026-06-23_【机器人 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ---- （2）算法篇（SAC）/index.md>)
- [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（3）算法篇（RLPD）](<notes/2026-06-25_【机器人 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（3）算法篇（RLPD）/index.md>)
- [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（4）算法篇（DrQ vs VICE）](<notes/2026-06-26_【机器人 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（4）算法篇（DrQ vs VICE）/index.md>)
- [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（5）工程篇](<notes/2026-06-28_【机器人 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（5）工程篇/index.md>)
- [【机器人 / 强化学习】HIL-SERL：人类在环驱动的具身智能进化框架](<notes/2026-06-29_【机器人 强化学习】HIL-SERL：人类在环驱动的具身智能进化框架/index.md>)
- [【机器人 / 强化学习】HIL-SERL 算法篇：DQN + SAC 混合架构的实现哲学](<notes/2026-06-30_【机器人 强化学习】HIL-SERL 算法篇：DQN + SAC 混合架构的实现哲学/index.md>)
- [【机器人 / 强化学习】HIL-SERL 算法篇：HG-DAgger 与 RLPD —— 从模仿到超越的训练双阶段](<notes/2026-07-01_【机器人 强化学习】HIL-SERL 算法篇：HG-DAgger 与 RLPD —— 从模仿到超越的训练双阶段/index.md>)
- [【机器人 / 强化学习】HIL-SERL 工程篇：人类在环的工程架构与物理设计](<notes/2026-07-03_【机器人 强化学习】HIL-SERL 工程篇：人类在环的工程架构与物理设计/index.md>)
- [【机器人 / 强化学习】LWD（Learning while Deploying）机器人持续进化的强化学习框架](<notes/2026-07-06_【机器人 强化学习】LWD（Learning while Deploying）机器人持续进化的强化学习框架/index.md>)
- [【机器人 / 强化学习】IQL（Implicit Q-Learning）：离线强化学习的隐式价值提取](<notes/2026-07-07_【机器人 强化学习】IQL（Implicit Q-Learning）：离线强化学习的隐式价值提取/index.md>)
- [【机器人 / 强化学习】DIVL：分布隐式价值学习](<notes/2026-07-09_【机器人 强化学习】DIVL：分布隐式价值学习/index.md>)
- [【机器人 / 强化学习】QAM：基于伴随匹配的 Q-learning 流策略优化](<notes/2026-07-11_【机器人 强化学习】QAM：基于伴随匹配的 Q-learning 流策略优化/index.md>)
- [【机器人 / 强化学习】QAM 与 DIVL：评价-执行闭环的完美配合](<notes/2026-07-13_【机器人 强化学习】QAM 与 DIVL：评价-执行闭环的完美配合/index.md>)
> 把 RL 从 LLM/Agent 延伸到真实机器人：SERL(1)全景 → (2)SAC → (3)RLPD → (4)DrQ vs VICE → (5)工程 → HIL-SERL 人在环。具身智能方向必读。

## 主题 8 · Agent 架构理论与设计模式（高屋建瓴）

> **适合**：架构师、技术决策者。读完前面的源码后再看这一组，会有「会当凌绝顶」的串联感。

- [Agent 17 种架构模式 分析 & 思考](<notes/2026-06-02_Agent 17 种架构模式 分析 & 思考/index.md>)
- [Agent OS ：五种驯服不确定性的范式](<notes/2026-06-04_Agent OS ：五种驯服不确定性的范式/index.md>)

**端侧 / Agent OS 形态**
- [端侧 Agent OS：手机架构与硬件协同设计](<notes/2026-06-08_端侧 Agent OS：手机架构与硬件协同设计/index.md>)

## 主题 9 · GUI / 端侧 Agent

> **适合**：做 GUI 自动化、手机/眼镜端智能体、Computer-Use 方向的读者。

**9.1 阶跃星辰 GUI-MCP（6 篇）**
- [【GUI-Agent】阶跃星辰 GUI-MCP 解读---(1)---论文](<notes/2026-03-16_【GUI-Agent】阶跃星辰 GUI-MCP 解读---(1)---论文/index.md>)
- [【GUI-Agent】阶跃星辰 GUI-MCP 解读---(2)---决策层](<notes/2026-03-18_【GUI-Agent】阶跃星辰 GUI-MCP 解读---(2)---决策层/index.md>)
- [【GUI-Agent】阶跃星辰 GUI-MCP 解读---(3)---执行层](<notes/2026-03-20_【GUI-Agent】阶跃星辰 GUI-MCP 解读---(3)---执行层/index.md>)
- [【GUI-Agent】阶跃星辰 GUI-MCP 解读---(4)---GUI-MCP 整体架构](<notes/2026-03-24_【GUI-Agent】阶跃星辰 GUI-MCP 解读---(4)---GUI-MCP 整体架构/index.md>)
- [【GUI-Agent】阶跃星辰 GUI-MCP 解读---(5)---命令解析和工具映射](<notes/2026-03-26_【GUI-Agent】阶跃星辰 GUI-MCP 解读---(5)---命令解析和工具映射/index.md>)
- [【GUI-Agent】阶跃星辰 GUI-MCP 解读---(6)---HITL(Human In The Loop)](<notes/2026-03-28_【GUI-Agent】阶跃星辰 GUI-MCP 解读---(6)---HITL(Human In The Loop)/index.md>)

**9.2 阿里通义 MAI-UI（2 篇）**
- [【GUI-Agent】阿里通义MAI-UI 代码阅读（1）--- 总体](<notes/2026-05-06_【GUI-Agent】阿里通义MAI-UI 代码阅读（1）--- 总体/index.md>)
- [【GUI-Agent】阿里通义MAI-UI 代码阅读（2）--- 实现](<notes/2026-05-08_【GUI-Agent】阿里通义MAI-UI 代码阅读（2）--- 实现/index.md>)

**9.3 端侧硬件 / 智能眼镜**
- [智能眼镜论文笔记](<notes/2025-11-13_智能眼镜论文笔记/index.md>)

## 主题 10 · 行业观察与其他

- [OpenAI Apps SDK：核心价值、竞争格局与发展猜想](<notes/2025-10-12_OpenAI Apps SDK：核心价值、竞争格局与发展猜想/index.md>)

---

> 📚 本指南共覆盖 138/141 篇。
>
> **未归类（补充阅读）**：
> - [【OpenClaw具身硬件】MiniClaw 阅读笔记---(1)基础](<notes/2026-08-11_【OpenClaw具身硬件】MiniClaw 阅读笔记---(1)基础/index.md>)
> - [【Agentic RL / 强化学习 / OPD】Hermes & OPD 源码阅读笔记](<notes/2026-08-10_【Agentic RL 强化学习 OPD】Hermes & OPD 源码阅读笔记/index.md>)
> - [VelesDB 深度解读 —— 融合向量、图与列存的端侧 AI 记忆引擎](<notes/2026-08-06_VelesDB 深度解读 —— 融合向量、图与列存的端侧 AI 记忆引擎/index.md>)

---
