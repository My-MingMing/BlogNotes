# 【OpenClaw具身硬件】MiniClaw 阅读笔记---(4) 文件读写

# 【OpenClaw具身硬件】MiniClaw 阅读笔记---(4) 文件读写

目录

* [【OpenClaw具身硬件】MiniClaw 阅读笔记---(4) 文件读写](#openclaw具身硬件miniclaw-阅读笔记---4-文件读写)
  + [0x00 概要](#0x00-概要)
  + [0x01 Read file & RAG](#0x01-read-file--rag)
    - [1.1 经典RAG的工作方式](#11-经典rag的工作方式)
    - [1.2 MimiClaw反过来：LLM是检索的发起者](#12-mimiclaw反过来llm是检索的发起者)
    - [1.3 为什么这么做？](#13-为什么这么做)
      * [动机1：MCU跑不动经典RAG的基础设施](#动机1mcu跑不动经典rag的基础设施)
      * [动机2：可解释性+可调试性](#动机2可解释性可调试性)
      * [动机3：LLM已经具备“读懂目录页+自主路由“的能力](#动机3llm已经具备读懂目录页自主路由的能力)
        + [判断1：在LLM足够强的时代，“检索“和“思考“的边界正在消融](#判断1在llm足够强的时代检索和思考的边界正在消融)
        + [判断2:“让LLM自己读“会带来质的不同](#判断2让llm自己读会带来质的不同)
    - [1.4 为什么能“工作“](#14-为什么能工作)
      * [机制1：上下文里始终有“地图“](#机制1上下文里始终有地图)
      * [机制2：纪律性的prompt协议](#机制2纪律性的prompt协议)
      * [机制3：文件系统作为“扁平知识库”](#机制3文件系统作为扁平知识库)
      * [和 Agentic-RAG/ Function-calling RAG 的关系](#和-agentic-rag-function-calling-rag-的关系)
    - [1.5 代价](#15-代价)
      * [隐含假设](#隐含假设)
      * [崩塌场景1：知识库规模超过“目录页极限“ 现象](#崩塌场景1知识库规模超过目录页极限-现象)
      * [崩塌场景2：文件名+标题不再具备“语义自描述性“ 现象](#崩塌场景2文件名标题不再具备语义自描述性-现象)
      * [崩塌场景3：换了一个不够强的模型](#崩塌场景3换了一个不够强的模型)
      * [崩塌场景4：延迟敏感场景（实时控制、对话感）现象](#崩塌场景4延迟敏感场景实时控制对话感现象)
      * [崩塌场景5：多用户共享/高并发知识更新](#崩塌场景5多用户共享高并发知识更新)
      * [崩塌场景6：知识是连续大段落，不是离散文档现象](#崩塌场景6知识是连续大段落不是离散文档现象)
      * [综合判断：什么时候该考虑 "补变"？](#综合判断什么时候该考虑-补变)
      * [元层面的崩塌：哲学本身的“成立条件“在变](#元层面的崩塌哲学本身的成立条件在变)
  + [0xFF 参考](#0xff-参考)

## 0x00 概要

MimiClaw 是**$5 芯片上的 AI 助理（OpenClaw）。没有 Linux，没有 Node.js，纯 C。**

用户在 TG 发一条消息，ESP32-S3 通过 WiFi 收到后送进 Agent 循环 — LLM 思考、调用工具、读取记忆 — 再把回复发回来。同时支持 **Anthropic (Claude)** 和 **OpenAI (GPT)** 两种提供商，运行时可切换。一切都跑在一颗 $5 的芯片上，所有数据存在本地 Flash。

![mimiclaw](images/img_001.png)

---

本篇会对MiniClaw 的思路和实现细节做进一步讨论。

"LLM自主read\_file替代RAG检索”=把检索的决策权从外部基础设施交还给 LLM，把检索基础设施从向量数据库简化为文件系统，把“准备上下文"这个动作从prompt阶段后移到推理阶段一用 frontier model 的语义理解力，换掉了所有embedding/index/rerank组件。

这是一个只有在大模型时代才成立的设计选择，也是MimiClaw能够把“完整Agent"塞进5美元芯片的关键技术杠杆之一。如果不是这个判断，MimiClaw就得在云端跑一套检索服务，整个“纯C/纯本地/ 0.5W"的产品故事会塌掉。

## 0x01 Read file & RAG

MiniClaw 用 LLM自主read\_file 替代RAG检索。它包含一个隐含的对照一经典 RAG是“系统替 LLM决定要看什么"，而MimiClaw反过来一"LLM自己决定要看什么"。

```
    /* Register read_file */
    mimi_tool_t rf = {
        .name = "read_file",
        .description = "Read a file from SPIFFS storage. Path must start with " MIMI_SPIFFS_BASE "/.",
        .input_schema_json =
            "{\"type\":\"object\","
            "\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Absolute path starting with " MIMI_SPIFFS_BASE "/\"}},"
            "\"required\":[\"path\"]}",
        .execute = tool_read_file_execute,
    };
    register_tool(&rf);
```

我们就来看看其思路和优劣。

### 1.1 经典RAG的工作方式

经典RAG（云端常见）由外部管线主导：

```
用户提问
	↓
[Embedding模型】把问题编码成向量[向量数据库]
	↓
检索top-K最相似段落
	↓
[Reranker/过滤]可选
	↓
把段落塞进 system prompt 或 user message
	↓
LLM接收"已经准备好的"上下文→生成回答
```

关键特征：LLM只是消费者。它在收到prompt之前就已经被“喂饱“了一检索过程对模型完全不可见，模型也无法对检索结果做主动反馈（除非套一层 agentic-RAG，但那已经是MimiClaw的方向了）

### 1.2 MimiClaw反过来：LLM是检索的发起者

MimiClaw没有上述任何一步外部检索。代替它的，是这样一个流程：

```
用户提问
	↓
LLM看到的system prompt只包含：
    - 静态指令+工具列表
    - 几份“目录页”（MEMORY摘要、Skills标题、文件路径约定）
LLM判断：“要回答这个问题，我需要看x文件“
	↓
LLM主动发起tool_use:read_file（path="/spiffs/skills/weather.md")
	↓
agent_loop 执行 tool，把文件内容作为 tool_result 塞回 messages
	↓
LLM拿到内容，再决定下一步（继续检索/回答/调其他工具）
```

关键反转：LLM不再是“被喂饱的消费者”，而是**"主动伸手取阅的读者“**。检索从“发生在LLM之前的预处理“变成了“发生在 LLM思考过程中的工具调用“。

### 1.3 为什么这么做？

三个层面的动机如下。

#### 动机1：MCU跑不动经典RAG的基础设施

回顾经典RAG的依赖项：

| 组件 | MCU上的代价 |
| --- | --- |
| Embedding模型（哪怕MiniLM量化） | 几MB模型权重 + 数百ms CPU推理，不可接受 |
| 向量数据库（FAISS/Annoy/hnsw） | 索引文件几十MB，PSRAM装不下 |
| 文档分块+索引构建管线 | 需要离线预处理服务器 |
| Reranker | 又一个模型 |

任意一项都超出ESP32-S3的资源预算。要么把这些放云端（增加网络依赖、隐私问题、服务器成本），要么放弃经典 RAG。MimiClaw选择了后者。

#### 动机2：可解释性+可调试性

经典RAG的一个老大难是“为什么这次检索结果不好“—是chunk太大？embedding模型不匹配？top-K截断了？re-ranker 出错？调试链路长。

MimiClaw这套机制下，每一次“检索“都是一个清晰可见的tool\_use日志：

```
[agent] tool_use:read_file(path="/spiffs/memory/MEMoRY.md")
[agent] tool_result: 1842 bytes
[agent] tool_use: read_file(path="/spiffs/skills/weather.md")
[agent] tool_result: 612 bytes
[agent] final answer generated
```

那个文件被读了、读到了什么、为什么读，都能从串口日志一行行看下来。调试RAG退化为调试一段对话。

#### 动机3：LLM已经具备“读懂目录页+自主路由“的能力

LLM的指令遵循能力强到可以替代很多原本需要专门管线的工作：

* 给它一个目录：“这里有5个skill，分别是X/Y/Z..."
* 它能根据用户问题自动判断：“用户问天气，我应该读weather.md"
* 然后发起对应的read\_file调用

这本质上是把“检索路由“这层职责从embedding相似度转交给了LLM 的语义理解。在大模型时代之前这是天方夜谭一以前的小模型理解不了“目录“的隐喻；但今天的frontier model，目录页就是它的索引。

MimiClaw做的不过是信任这种能力，并把它写进prompt协议里。

当然，这种做法是基于以下两个判断。

##### 判断1：在LLM足够强的时代，“检索“和“思考“的边界正在消融

* 经典RAG的预设是：检索是廉价的预处理，推理是昂贵的核心计算，所以要把检索做厚、做精、把“对的内容“喂进去。
* MimiClaw反过来预设：LLM 的推理已经强大到可以把“决定看什么“也包含进去。因此把检索作为推理的一部分（tool\_use），而不是推理之前的独立阶段。

这种范式在云端也有同样的趋势（Claude 的 agentic browsing、GPT 的 file\_search tool），Mimiclaw 只是因为资源约束被迫走得更彻底。

##### 判断2:“让LLM自己读“会带来质的不同

* 经典RAG：模型被动接受N段文本，无法判断“还需要再多一段”。
* MimiClaw模式：模型可以读完一份文件后觉得不够，继续read\_file第二份一这是一个真正的迭代过程

ReAct循环（最多10次工具调用）让LLM可以深度探索：

```
Q：我上周三跟你说了什么？
LLM: list_dir("/spiffs/sessions/") → 看到tg_12345.jsonl
LLM:read_file("/spiffs/sessions/tg_12345.jsonl") → 看到20条最近消息，没有上周三
LLM:read_file("/spiffs/memory/2026-04-22.md") → 找到当天日记 LLM：最终回答
```

这种“翻找 → 看一眼 → 决定再翻 → 给答案“的过程更接近人类查资料，而不是“系统给我5个段落，我从中挑”。信息密度低、覆盖面广的场景下，主动检索往往优于被动top-K一尤其是MimiClaw这种文件数量本来就少（几十到几百）的场景。

### 1.4 为什么能“工作“

光说"让LLM自己读“是不够的，MimiClaw至少有三件事在协同支撑这个范式：

#### 机制1：上下文里始终有“地图“

context\_builder.c把这些总是出现在system prompt里：

* 工具列表（包含read\_file、list\_dir等“探索性“工具）
* MEMORY.md内容（让LLM知道“我已经记得什么"）
* Skills目录页（让LLM知道“还有哪些文件值得看"）

* 路径约定（/spiffs/memory/，/spiffs/skills/，/spiffs/sessions/)

这构成了一份“知识地图“。LLM不需要embedding也能知道该去哪里找东西一一一因为地图本身就告诉了它。

#### 机制2：纪律性的prompt协议

回看system prompt里这些守则：

```
- Always read_file MEMORY.md before writing
- Use get_current_time before writing daily notes
- When a task matches a skill,read the full skill file for detailed instructions
- Read MEMoRY.md before answering questions about the user's preferences
```

这些是用自然语言写出的检索时机规则一什么时候应该主动read。它们替代了经典RAG里的“检索触发器逻辑"，但实现成本只是几行markdown。

#### 机制3：文件系统作为“扁平知识库”

SPIFFS上的所有文件都是markdown/JSONL，人类可读、模型可读、可枚举。LLM可以：

* 用list\_dir("/spiffs/skills/")发现资源
* 用read\_file加载特定资源
* 用write\_file/edit\_file修改资源

整个文件系统就是一个“模型可寻址的知识库”。它不用建索引，因为LLM自己能读懂"weather.md 里大概是天气相关内容“一一一文件名 + 标题 + 第一段就是天然的语义索引。

#### 和 Agentic-RAG/ Function-calling RAG 的关系

熟悉Agent架构的人会问：这不就是Agentic RAG/ReAct+retrieval tools吗？

是的，这正是Agentic RAG的一种极简形态。但MimiClaw把它推到一个更极端的位置：

| 维度 | 典型 Agentic RAG | MimiClaw |
| --- | --- | --- |
| 检索后端 | 仍然是向量库 / 关键词索引 | 就是文件系统 + 路径 |
| 工具复杂度 | 多个检索工具 + 重排 | 三五个文件操作工具 |
| 索引维护 | 需要离线管线更新 embedding | 零索引—LLM 写文件即 "更新索引" |
| 知识演化 | 重新跑 ingest pipeline | LLM 自己 write\_file 立即生效 |
| 可观测性 | 检索日志 + LLM 日志 | 只有 tool\_use 日志，统一视图 |

所以更准确的描述是：MimiClaw是“把Agentic RAG简化到只剩文件系统“的极致版本。

### 1.5 代价

为了平衡，必须承认这套哲学的代价：

| 代价 | 说明 |
| --- | --- |
| 延迟更高 | 每次 read\_file 都是一轮 LLM 调用，10 KB 文件 + ReAct 多轮 = 数秒到十几秒 |
| token 成本可能更高 | 主动探索会重复加载文件内容到 messages |
| 依赖模型能力 | 弱模型不会主动检索，会瞎答；只有 Claude/GPT 这种级别的模型才能稳定执行 |
| 大规模知识库无效 | 文件超过几百个时，目录页塞不下，LLM 无法发现所有可用资源 |
| 缺少跨文档融合 | 没有 reranker / fusion，靠 LLM 自己把多个 read\_file 结果 "在脑子里" 融合 |

所以，这种哲学的适用边界很清晰：适合知识库小（<1000 文件）、文件本身有清晰命名、运行 frontier-class LLM的场景。MimiClaw完美卡在这个区间--- 一台个人设备上的几十份 markdown，Anthropic/OpenAI 顶级模型，天作之合。

要是哪天有人想拿MimiClaw这套架构去管10万份企业文档，那才是真正需要回归经典 RAG的时刻。

#### 隐含假设

这套哲学依赖几个隐含假设 ----- 一旦假设失效，整个范式就会出问题。它的隐含假设大致有六条：

* 知识库规模小到能在目录页里列得完
* 文件命名+标题已经具备语义可识别性
* LLM模型足够强、能稳定自主路由
* 用户对"多轮tool\_use"的延迟与成本可接受
* 知识更新频率低/由LLM自己负责更新
* 知识结构是离散文档，不是连续大段落

每条假设崩塌时，触发的失败模式不同，需要的“补变“也不同。

#### 崩塌场景1：知识库规模超过“目录页极限“ 现象

当文件数从几十涨到几千，context\_builder.c里Skills段那块2KB缓冲再也装不下完整目录。要么截断（LLM 看不到后半部分文件），要么膨胀（systemprompt突破16KB上限）。

更隐蔽的失败：即使prompt还能装下，LLM在100+条目录里也找不准一注意力分散，召回率掉到不可用。

补变手段：从“全量目录“退化为“分层索引+检索”：

| 等级 | 做法 |
| --- | --- |
| 一级补变 | 把 skills 按目录分类（/spiffs/skills/home/, /spiffs/skills/work/），目录页只列类别而非文件 |
| 二级补变 | 引入 search\_files(query) 工具，做文件名 + 标题的关键词匹配（不是 embedding，是 BM25/正则）。LLM 不再看完整目录，而是先 search 再 read |
| 三级补变 | 上云端做向量检索—这时候已经回到经典 RAG，承认 MCU 不再适合作为单点处理器，需要外挂检索服务 |

#### 崩塌场景2：文件名+标题不再具备“语义自描述性“ 现象

LLM自主路由依赖文件名足以暗示内容。但当文件命名失去这种性质：

* 自动生成的文件名（note\_a8f3c2.md、UUID命名）
* 用代号/内部缩写（PROJ-X42-spec.md）
* 大量同质化标题（每天日记都叫日记，标题没区分度）

LLM看到目录就懵了一它只能乱试或完全放弃主动检索。

触发条件

* 系统自动归档机制把“语义命名“剥夺了
* 多语言混杂（中文文件名+英文LLM训练偏置）
* 极度专业领域（医学代码、化学命名）LLM训练时见过少

补变手段

* 强制语义命名：在write\_file工具的 prompt 里加约束："filename should describe content in 3-5words"
* 加目录段落描述：每个目录放一份INDEX.md，由LLM维护，列“这个目录里有什么“
* 文件首行强制为 #Human-readable Title—skill\_loader.c已经这么做了，可推广

核心思路：把“语义“从“文件名“扩展到”自我描述的元数据”，让目录页的每一项都自带摘要。

#### 崩塌场景3：换了一个不够强的模型

补变手段

* 缩短 ReAct链：在弱模型上把工具调用上限从 10砍到 3，避免迷路
* 预加载策略：在context\_builder.c里直接inline 一份“今日相关“内容（比如总是把今天的日记完整塞进去），不依赖 LLM主动读
* 简化目录页：每次只暴露3-5个最相关的 skill，由路径前缀或最近使用时间筛选
* 回退到经典 RAG：当模型置信度不足时，外挂一个 BM25/embedding 服务做硬路由

本质：弱模型场景下要把更多决策权拿回到prompt builder手里一philosophy从“信任LLM"逐步退回“系统辅助LLM"。

#### 崩塌场景4：延迟敏感场景（实时控制、对话感）现象

“看目录→读文件→综合→回答“是一个串行的多轮LLM调用。每一轮HTTPS+TLS+LLM推理都是3-10  
秒。如果回答需要读3个文件，总延迟可达30秒。

但有些场景完全不能等候：

* 智能家居语音助手（“开灯“必须1秒内响应）
* 紧急通知（火警、安防触发）
* 短回合社交聊天（用户期待<5秒响应感）

这些场景下 ReAct+read\_file的延迟模型直接出局。

补变手段

* 分级路由：在LLM调用前用一段轻量C代码做意图识别（关键词匹配），命中“控制类“指令直接走GPIO工具，不进ReAct 循环
* 预热缓存：把高频文件（MEMoRY.md、今日日记）在 system prompt 阶段就完整inline，LLM 不需要再 read 猪
* 异步预读：心跳触发时预读高概率文件并缓存到PSRAM，agent\_loop命中时省一次 read
* 流式响应（如果未来支持）：让LLM边读文件边输出“正在查记忆..“给用户填充感

这个崩塌场景指向一个更大的真相：Agentic RAG的“灵活性“和“延迟“是天敌。MimiClaw 的设计偏向前者，所以低延迟场景需要绕过整个Agent循环走快速通道。

#### 崩塌场景5：多用户共享/高并发知识更新

现象

当前架构假设"-个LLM主动维护一份记忆"，所以用 read\_fileMEMoRY.md之前必须 read，再edit\_file 写一这本质是乐观锁丢失版。

如果出现：

* 多个用户同时和同一个MimiClaw对话
* Heartbeat在写MEMORY.md时用户也在触发对话
* Cron任务回调和 TG消息并发处理

LLM看到的MEMORY.md可能已经被另一条路径改过了，但LLM用的是几秒前的快照来edit\_file，写回时静默覆盖了别人的修改。

触发条件

* 多用户家庭部署（爸爸妈妈孩子各自和同一台MimiClaw对话）
* IoT群部署（一个集群共用一份记忆）
* 高频心跳+高频对话叠加

补变手段

* 加文件级锁：在memory\_store.c里用 FreeRToSmutex 包 read-modify-write，但这只解决并发写
* 按用户分文件：MEMORY.md改为MEMORY\_<uSer\_id>.md，避免共享
* append-only模型：永远不edit，只append；定期由 LLM触发“压缩归档"，但归档窗口短，丢失少
* 真正的并发场景退到云端：embedded不适合做共享状态权威源，应该外挂一个KV服务

这个崩塌场景揭示了一个本质限制：MCU单设备适合做单用户的私人助理，不适合做多用户的共享大脑。

#### 崩塌场景6：知识是连续大段落，不是离散文档现象

MimiClaw的“文件即知识单元“假设最适合离散、可独立解读的小文档一一份skill、一天日记、一段GPIO 守则。但当知识形态是：

* 一本200页的产品手册（不能整个read，分块又丢上下文）
* 一份长会议录音转写（线性叙述，无章节）法律合同/学术论文（需要跨节引用）
* 大段聊天历史（信息分散在数百条对话里）

LLM没法用"一次read\_file“解决，又没有chunking+ranking的基础设施去定位段落。它只能盲读或放弃。

触发条件

* 用户希望 MimiClaw 帮自己 "读一份长文档"
* 客服场景需要查询长合同条款
* 意识形态本身就不能拆成小文件

补变手段

* 离线预切分：在文档进入 SPIFFS 之前先用 LLM 切成有标题的小章节，每章独立存为一个 .md（用 LLM 做离线 chunking 的语义版本）
* 加 read\_file\_section(path, section) 工具：让 LLM 能按 markdown 二级标题精读一段
* 页码 + 摘要外挂：每个长文档配一份 .toc.md，列出小节摘要 + 起止行号；LLM 先读 toc 再精读
* 真正长文场景需要 chunk + embedding：这时候彻底承认现行哲学不适用，回到经典 RAG

临界点：单文档 > 8 KB（超过 tool\_result 上限）就必须有切分机制；> 32 KB 基本必须 RAG 化。

#### 综合判断：什么时候该考虑 "补变"？

把上面六个场景归纳成一张决策表：

| 信号 | 该补变什么 | 紧迫程度 |
| --- | --- | --- |
| 文件数 > 200 | 加 search\_files + 分类目录 | 中 |
| 用户切到 mini/小模型 | inline 高频内容 + 缩短 ReAct | 高 |
| 出现实时控制需求 | 加快速通道绕过 Agent | 高 |
| 多用户场景 | 加锁 + 分用户记忆 | 看部署 |
| 处理长文档 | 离线切分 + section 工具 | 低（如果不做这个产品就不需要） |
| 文件名失去语义 | 加 INDEX.md + 强制命名约束 | 低（设计早期就避免） |

总体原则：先观察失败模式（用户问了什么、LLM 做了什么、最终给了什么答案），再判断对应哪种崩塌场景。不要预防性地引入复杂度—MimiClaw 当前的 "零 RAG 基础设施" 在 90% 个人场景里都够用，过早加 vector store反而把"5美元芯片纯本地"的故事毁了。

#### 元层面的崩塌：哲学本身的“成立条件“在变

MimiClaw的整套上下文工程哲学，本质是押注“frontier LLM的能力会持续增强、价格会持续下降”。

如果未来某天：

* frontier model价格反转上涨→多轮tool\_use成本不可承受
* 模型能力被监管限制（function-calling 被收窄、tool\_use被审计） → 主动检索不可靠
* 用户隐私法规要求所有处理本地化 → 不能依赖外部LLM API

那么“用frontier model 智能换基础设施复杂度“这笔账就需要重新算了一一一整个哲学的根基会被动摇，而不是某个具体崩塌场景。届时MimiClaw 可能需要切换到纯本地小模型+经典RAG 的范式，但那已经是另一个产品了。

这层“宏观崩塌“提醒我们：任何 Agent架构都是和当下LLM生态绑定的工程权衡，没有永恒正确。MimiClaw  
的优雅恰恰来自于它清醒地接受了这种时代依赖一它没装作自己是放之四海而皆准的方案，而是给定 2024-2026的LLM现状下的最优近似解。

## 0xFF 参考
