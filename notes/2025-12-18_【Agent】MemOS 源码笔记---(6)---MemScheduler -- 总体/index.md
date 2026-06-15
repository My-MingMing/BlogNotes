# 【Agent】MemOS 源码笔记---(6)---MemScheduler -- 总体

目录

* [【Agent】MemOS 源码笔记---(6)---MemScheduler -- 总体](#agentmemos-源码笔记---6---memscheduler----总体)
  + [0x00 摘要](#0x00-摘要)
  + [0x01 能力介绍](#0x01-能力介绍)
    - [1.1 需求](#11-需求)
    - [1.2 如何调度](#12-如何调度)
    - [1.3 MemScheduler 功能](#13-memscheduler-功能)
    - [1.4 执行示例](#14-执行示例)
      * [1.4.1 代码功能概述](#141-代码功能概述)
      * [1.4.2 案例：家庭助理场景中的记忆调度](#142-案例家庭助理场景中的记忆调度)
  + [0x02 MemScheduler 基本概念](#0x02-memscheduler-基本概念)
    - [2.1 核心组件](#21-核心组件)
    - [2.2 主要特点](#22-主要特点)
    - [2.3 BaseScheduler](#23-basescheduler)
      * [2.3.1 图例](#231-图例)
      * [2.3.2 定义](#232-定义)
    - [2.4 消息处理](#24-消息处理)
      * [2.4.1 消息类型](#241-消息类型)
      * [2.4.2 调度消息结构](#242-调度消息结构)
    - [2.5 函数实现](#25-函数实现)
  + [0xFF 参考](#0xff-参考)

## 0x00 摘要

记忆调度就像大脑的注意力机制，动态决定在合适的时刻调用合适的记忆。

在 MemOS 中，**记忆调度（Memory Scheduling）** 通过对【不同使用效率（参数>激活>工作>其他明文）的记忆】的相互调度，让模型能更高效、准确地获取用户所需的记忆。在对话和任务进行时，通过预测用户后续对话所需记忆并提前调入高效率记忆类型如激活记忆工作记忆，加速推理链路。

记忆调度的具体实现就是MemScheduler ，这是一个与 MemOS 系统并行运行的并发记忆管理系统，它协调 AI 系统中工作记忆、长时记忆和激活记忆之间的记忆操作。它通过事件驱动调度处理记忆检索、更新和压缩。该系统特别适合需要动态记忆管理的对话代理和推理系统。

备注：本文基于MemOS的文档和源码进行学习和解读，似乎其文档并没有跟上源码更新的速度，而且也有部分功能未实现或者未开源。

因为 MemScheduler 内容过多，因此分为两篇。

## 0x01 能力介绍

MemOS（Memory Operating System）是面向 Agent 的记忆操作系统，而 MemScheduler 是其中的 “调度器”，类比操作系统的 CPU 调度器，核心负责**所有记忆操作的统筹、调度、优先级管理和资源分配**。可以说，MemOS提供了基础的记忆管理框架，而MemScheduler则在此基础上增加了动态调度和自动化管理的能力，使得整个记忆系统更加智能化和高效。

### 1.1 需求

我们首先看看为什么需要调度。

在复杂的交互里，如果每次过来只靠简单的全局搜索，系统可能会：

* **太慢**：等到用户问完再临时搜索，首 Token 延迟高。
* **不准**：太多历史，反而淹没了关键信息，难以检索。

调度的作用就是让系统具备“即时准备、快速响应”的能力：

* **预加载**：在对话一开始，就加载用户的常用背景。
* **预测调用**：在用户还没输入完时，提前准备可能要用的记忆。

### 1.2 如何调度

可以结合任务语义、上下文、访问频率、生命周期等信息，动态安排记忆的调用与存储。

| 维度 | 说明 |
| --- | --- |
| 调什么？ | 参数记忆（长期知识与技能）  激活记忆（运行时的 KV 缓存与隐层状态） 明文记忆（外部可编辑的事实、用户偏好、检索片段） 支持`明文 ⇆ 激活 ⇆ 参数`的动态迁移，高频使用的明文片段可以提前编译成 KV 缓存；长期稳定的模板可以沉淀到参数中。 |
| 什么时候调？ | 上下文和高效记忆不足以支撑回答用户提问时进行记忆结构的优化 根据用户的意图和需求，对用户可能需要的记忆内容进行提前的准备 在连续提问过程中，调度记忆保持对话场景的高效准确 |
| 调给谁？ | 当前用户、特定角色代理（Agent）、或跨任务的共享上下文 |
| 调成什么样？ | 记忆会被打上热度、时效性、重要性等指标。调度器据此决定先加载谁、谁放冷存、谁需要归档。 |

使用 MemOS 云服务时，调度的作用可以从 `searchMemory` API 的表现中观测到：

* 它能快速返回相关记忆，避免上下文断裂。
* 返回的内容已经过调度器优化，确保结果既相关又不会过载模型输入。

### 1.3 MemScheduler 功能

核心职责如下。

* 记忆单元编排管理
  + 管理多个 MemCube 实例（即记忆容器）
  + 作为操作系统层级来处理和协调这些记忆单元
  + 提供创建、搜索、更新和删除记忆单元的方法
* 任务调度与自动化
  + 通过预定的操作实现自动化的内存管理任务
  + 处理各种内存操作，包括：
    - 添加新记忆（ADD\_LABEL）
    - 回答查询（ANSWER\_LABEL）
    - 处理查询（QUERY\_LABEL）
* 多用户支持
  + 设计用于多用户场景，并采用线程安全操作
  + 验证用户对特定记忆单元的访问权限
  + 为不同用户提供独立的上下文环境
* 记忆生命周期管理
  + 协调信息在不同组件之间的流动
  + 管理何时以及如何处理、存储和检索记忆
  + 与其他 MemOS 组件（如 MemReader 和大语言模型）集成

### 1.4 执行示例

官方文档中说，`examples/mem_scheduler/schedule_w_memos.py` 是一个演示脚本，展示如何使用 `MemScheduler` 模块。它说明了对话上下文中的记忆管理和检索。

#### 1.4.1 代码功能概述

此脚本演示了初始化和使用记忆调度器的两种方法：

1. **自动初始化**: 通过配置文件配置调度器
2. **手动初始化**: 显式创建和配置调度器组件

该脚本模拟用户和助手之间关于宠物的对话，演示记忆调度器如何管理对话历史并检索相关信息。

#### 1.4.2 案例：家庭助理场景中的记忆调度

**前段时间：用户忙着买房**

用户经常说：

* “帮我查一下 XX 小区的二手房均价。”
* “提醒我周六去看房。”
* “记录一下房贷利率的最新变化。”

MemOS系统操作

* 系统最初将这些条目都生成 **明文记忆**。
* 因为“买房”相关信息被频繁提及，调度器在后台判断它是近期的**核心主题**，于是将这些明文迁移为 **激活记忆**，让后续查询更快更直接。

**最近：用户房子买好了开始装修**

用户开始频繁提到：

* “周末要去看瓷砖。”
* “提醒我和装修公司确认水电改造。”
* “记一下下周家具送货时间。”

MemOS系统操作

* 系统继续生成新的 **明文记忆**。
* 调度器检测到“装修”已经成为新的高频主题，于是把这些条目迁移为 **激活记忆**。
* 同时，之前“买房”相关激活记忆不再被频繁使用，会被自动**降级回明文层**，以减少活跃占用。

**当前时刻：用户随口说——我感觉好多事堆在一起，你帮我理一下**

如果没有调度，系统只能全库检索，把所有可能相关的记忆拉出来：

* 看瓷砖（装修）
* 确认水电改造（装修）
* 家具送货（装修）
* 查房价（买房时期，过时）
* 看房（买房时期，过时）
* 买菜（生活琐事）
* 看电影（生活琐事）

有调度时，系统可以更快速返回

* 看瓷砖
* 确认水电改造
* 家具送货

  👉 **用户体感 UP**
* 响应更快（因为不需要全库检索）。
* 列出的就是自己最挂念的事 → 感觉助理“很懂我”

然而，实际上，在代码中只能找到这个例子：MemOS-main\examples\mem\_scheduler\memos\_w\_scheduler.py。

有兴趣的读者可以自行研究。

## 0x02 MemScheduler 基本概念

MemScheduler 实际上扮演着 MemOS 系统的“进程管理器”角色，确保按照系统的策略和用户需求正确地协调、调度和执行内存操作。

### 2.1 核心组件

`MemScheduler` 系统围绕几个关键组件构建：

* 消息处理: 通过带有标记processor的调度器处理传入消息
* 记忆管理: 管理不同的记忆类型 (工作、长时、用户)
* 检索系统: 基于上下文高效检索相关记忆项
* 监控: 跟踪记忆使用、频率和触发更新
* 调度器 (路由器): 通过检查来自 MemOS 系统的消息触发不同的记忆重分配策略。
* 日志记录: 维护记忆操作日志用于调试和分析

### 2.2 主要特点

* 模块化架构
  + 采用插件式架构，可通过配置进行定制
  + 通过工厂模式支持不同的实现方式
  + 允许自定义调度行为
* 任务调度
  + 自动处理记忆的添加、查询、更新等操作
  + 支持定时任务和事件驱动的任务
* 与 MOS 核心集成
  + 与主 MOS 功能协同工作
  + 可通过配置启用/禁用
  + 处理消息并协调内存操作
* 后台处理
  + 在后台运行计划任务
  + 支持启动和停止调度服务
  + 管理内存操作的资源分配

### 2.3 BaseScheduler

* 该类是 MemOS 的核心调度器基类，整合了消息队列、内存管理、任务分发、监控等功能，用于统筹内存立方体（MemCube）的调度逻辑。
* 核心能力包括工作内存替换、激活内存周期性更新、消息提交与消费、多线程并行任务分发，同时支持 RabbitMQ 消息队列和 Redis 存储集成。
* 支持动态上下文切换，通过配置灵活启用 / 禁用激活内存、并行分发等功能，适配不同场景需求。
* 特色是模块化设计（集成检索器、监控器、分发器等子模块），线程安全的上下文管理，优雅的启动 / 停止机制，以及完善的异常处理和资源清理，确保系统稳定运行。

#### 2.3.1 图例

具体架构如下。

```
+================================================+
|               BaseScheduler                    |
|  (整合消息队列、内存管理、任务分发的核心调度器)       |
+================================================+
| 初始化流程:                                      |
|  +------------------------+                    |
|  | 接收配置参数             |                     |
|  +------------------------+                     |
|  | 设置超参数 (top_k/窗口大小等) |                 |
|  +------------------------+                     |
|  | 初始化内部消息队列        |                     |
|  +------------------------+                     |
|  | 延迟初始化子模块          |                     |
|  | (检索器/监控器/分发器)    |                     |
|  +------------------------+                     |
+=================================================+
| 子模块初始化流程:                                  |
|  +------------------------+                     |
|  | 绑定LLM实例              |                     |
|  +------------------------+                     |
|  | 初始化监控器(通用/分发)    |                     |
|  +------------------------+                     |
|  | 初始化检索器             |                     |
|  +------------------------+                     |
|  | 加载认证配置             |                     |
|  +------------------------+                     |
|  | 初始化RabbitMQ连接       |                     |
|  +------------------------+                     |
+=================================================+
| 消息处理流程:                                     |
|  +------------------------+                     |
|  | 提交消息到队列            |                     |
|  | (支持单条/多条消息)       |                     |
|  +------------------------+                     |
|  | 消费线程循环检测队列       |                     |
|  | (按间隔批量取消息)        |                     |
|  +------------------------+                     |
|  | 分发器调度任务执行        |                     |
|  | (并行/串行模式可选)       |                     |
|  +------------------------+                     |
+=================================================+
| 工作内存替换流程:                                  |
|  +------------------------+                     |
|  | 获取用户查询历史          |                     |
|  +------------------------+                     |
|  | 重新排序内存(LLM参与)     |                     |
|  +------------------------+                     |
|  | 过滤不相关内存            |                     |
|  +------------------------+                     |
|  | 更新内存监控器           |                     |
|  +------------------------+                     |
|  | 替换文本内存中的工作集     |                     |
|  +------------------------+                     |
+=================================================+
| 激活内存更新流程:                                  |
|  +------------------------+                     |
|  | 检查更新间隔是否到达       |                     |
|  +------------------------+                     |
|  | 从监控器获取新内存        |                     |
|  +------------------------+                     |
|  | 组装文本并提取KV缓存      |                     |
|  +------------------------+                     |
|  | 清除旧缓存并添加新缓存     |                     |
|  +------------------------+                     |
|  | 持久化缓存到磁盘          |                     |
|  +------------------------+                     |
+=================================================+
| 启动/停止流程:                                    |
| 启动:                                            |
|  +------------------------+                     |
|  | 初始化分发器线程池        |                     |
|  +------------------------+                     |
|  | 启动消息消费线程          |                     |
|  +------------------------+                     |
| 停止:                                            |
|  +------------------------+                     |
|  | 标记运行状态为停止        |                     |
|  +------------------------+                     |
|  | 等待消费线程退出          |                     |
|  +------------------------+                     |
|  | 关闭分发器/监控器         |                     |
|  +------------------------+                     |
|  | 清空所有消息队列          |                     |
|  +------------------------+                     |
+=================================================+
```

#### 2.3.2 定义

代码如下。

```
class BaseScheduler(RabbitMQSchedulerModule, RedisSchedulerModule, SchedulerLoggerModule):
    """所有内存调度器的基类，整合消息队列、内存管理和任务分发功能"""

    def __init__(self, config: BaseSchedulerConfig):
        """使用给定配置初始化调度器"""
        super().__init__()
        self.config = config  # 存储调度器配置

        # 超参数配置（从配置中读取，无则使用默认值）
        self.top_k = self.config.get("top_k", 10)  # 检索时返回的Top-K结果数
        self.context_window_size = self.config.get("context_window_size", 5)  # 上下文窗口大小
        self.enable_activation_memory = self.config.get("enable_activation_memory", False)  # 是否启用激活内存
        self.act_mem_dump_path = self.config.get("act_mem_dump_path", DEFAULT_ACT_MEM_DUMP_PATH)  # 激活内存存储路径
        self.search_method = TreeTextMemory_SEARCH_METHOD  # 文本内存搜索方法
        self.enable_parallel_dispatch = self.config.get("enable_parallel_dispatch", False)  # 是否启用并行分发
        self.thread_pool_max_workers = self.config.get(
            "thread_pool_max_workers", DEFAULT_THREAD__POOL_MAX_WORKERS
        )  # 线程池最大工作线程数

        # 子模块初始化（延迟初始化，通过initialize_modules方法完成）
        self.retriever: SchedulerRetriever | None = None  # 检索器：用于内存检索和排序
        self.db_engine: Engine | None = None  # 数据库引擎：用于数据持久化
        self.monitor: SchedulerGeneralMonitor | None = None  # 通用监控器：监控调度器运行状态
        self.dispatcher_monitor: SchedulerDispatcherMonitor | None = None  # 分发器监控器：监控任务分发
        # 任务分发器：管理任务分发，支持并行/串行模式
        self.dispatcher = SchedulerDispatcher(
            max_workers=self.thread_pool_max_workers,
            enable_parallel_dispatch=self.enable_parallel_dispatch,
        )

        # 内部消息队列配置
        self.max_internal_message_queue_size = self.config.get(
            "max_internal_message_queue_size", 100
        )  # 内部消息队列最大容量
        self.memos_message_queue: Queue[ScheduleMessageItem] = Queue(
            maxsize=self.max_internal_message_queue_size
        )  # 调度消息队列
        self.max_web_log_queue_size = self.config.get("max_web_log_queue_size", 50)  # Web日志队列最大容量
        self._web_log_message_queue: Queue[ScheduleLogForWebItem] = Queue(
            maxsize=self.max_web_log_queue_size
        )  # Web日志存储队列
        self._consumer_thread = None  # 消息消费线程引用
        self._running = False  # 调度器运行状态标识
        self._consume_interval = self.config.get(
            "consume_interval_seconds", DEFAULT_CONSUME_INTERVAL_SECONDS
        )  # 消息消费间隔（秒）

        # 其他属性
        self._context_lock = threading.Lock()  # 上下文锁：保证线程安全
        self.current_user_id: UserID | str | None = None  # 当前用户ID
        self.current_mem_cube_id: MemCubeID | str | None = None  # 当前内存立方体ID
        self.current_mem_cube: GeneralMemCube | None = None  # 当前内存立方体实例
        self.auth_config_path: str | Path | None = self.config.get("auth_config_path", None)  # 认证配置文件路径
        self.auth_config = None  # 认证配置实例
        self.rabbitmq_config = None  # RabbitMQ配置实例

    def initialize_modules(
        self,
        chat_llm: BaseLLM,
        process_llm: BaseLLM | None = None,
        db_engine: Engine | None = None,
    ):
        """初始化调度器的所有子模块（检索器、监控器、分发器等）"""
        # 若未指定处理用LLM，默认使用聊天用LLM
        if process_llm is None:
            process_llm = chat_llm

        try:
            # 存储LLM实例
            self.chat_llm = chat_llm
            self.process_llm = process_llm
            self.db_engine = db_engine

            # 初始化通用监控器：监控内存和查询状态
            self.monitor = SchedulerGeneralMonitor(
                process_llm=self.process_llm, config=self.config, db_engine=self.db_engine
            )
            self.db_engine = self.monitor.db_engine  # 同步监控器的数据库引擎

            # 初始化分发器监控器：监控任务分发状态
            self.dispatcher_monitor = SchedulerDispatcherMonitor(config=self.config)
            # 初始化检索器：处理内存检索、排序和过滤
            self.retriever = SchedulerRetriever(process_llm=self.process_llm, config=self.config)

            # 若启用并行分发，初始化并启动分发器监控器
            if self.enable_parallel_dispatch:
                self.dispatcher_monitor.initialize(dispatcher=self.dispatcher)
                self.dispatcher_monitor.start()

            # 初始化认证配置
            if self.auth_config_path is not None and Path(self.auth_config_path).exists():
                # 从指定路径加载认证配置
                self.auth_config = AuthConfig.from_local_config(config_path=self.auth_config_path)
            elif AuthConfig.default_config_exists():
                # 从默认路径加载认证配置
                self.auth_config = AuthConfig.from_local_config()
            else:
                # 从环境变量加载认证配置
                self.auth_config = AuthConfig.from_local_env()

            # 若加载到认证配置，初始化RabbitMQ
            if self.auth_config is not None:
                self.rabbitmq_config = self.auth_config.rabbitmq
                self.initialize_rabbitmq(config=self.rabbitmq_config)

        except Exception as e:
            logger.error(f"Failed to initialize scheduler modules: {e}", exc_info=True)
            # 初始化失败时清理部分初始化的资源
            self._cleanup_on_init_failure()
            raise
```

### 2.4 消息处理

调度器通过带有专用processor的调度器来处理消息。

#### 2.4.1 消息类型

| 消息类型 | processor方法 | 描述 |
| --- | --- | --- |
| `QUERY_LABEL` | `_query_message_consume` | 处理用户查询并触发检索 |
| `ANSWER_LABEL` | `_answer_message_consume` | 处理答案并更新记忆使用 |

#### 2.4.2 调度消息结构

调度器使用以下格式（ScheduleMessageItem）处理来自其队列的消息：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| `item_id` | `str` | UUID (自动生成) 用于唯一标识 |
| `user_id` | `str` | 关联用户的标识符 |
| `mem_cube_id` | `str` | mem cube 的标识符 |
| `label` | `str` | 消息标签 (例如，`QUERY_LABEL`、`ANSWER_LABEL`) |
| `mem_cube` | `GeneralMemCube｜str` | mem cube 对象或引用 |
| `content` | `str` | 消息内容 |
| `timestamp` | `datetime` | 消息提交时间 |

同时调度器将按照以下结构（ScheduleLogForWebItem）发送调度消息。

| 字段 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| `item_id` | `str` | 唯一日志条目标识符 (UUIDv4) | 自动生成 (`uuid4()`) |
| `user_id` | `str` | 关联用户标识符 | (必需) |
| `mem_cube_id` | `str` | 链接的 mem cube ID | (必需) |
| `label` | `str` | 日志类别标识符 | (必需) |
| `from_memory_type` | `str` | 源记忆分区 可能的值： - `"LongTermMemory"` - `"UserMemory"` - `"WorkingMemory"` | (必需) |
| `to_memory_type` | `str` | 目标记忆分区 | (必需) |
| `log_content` | `str` | 详细日志消息 | (必需) |
| `current_memory_sizes` | `MemorySizes` | 当前记忆利用率 | ``` DEFAULT_MEMORY_SIZES = {   "long_term_memory_size": -1,   "user_memory_size": -1,   "working_memory_size": -1,   "transformed_act_memory_size": -1 } ``` |
| `memory_capacities` | `MemoryCapacities` | 记忆分区限制 | ``` DEFAULT_MEMORY_CAPACITIES = {   "long_term_memory_capacity": 10000,   "user_memory_capacity": 10000,   "working_memory_capacity": 20,   "transformed_act_memory_capacity": -1 } ``` |
| `timestamp` | `datetime` | 日志创建时间 | 自动设置 (`datetime.now`) |

### 2.5 函数实现

BaseScheduler 的函数如下：

```
    def replace_working_memory(
        self,
        user_id: UserID | str,
        mem_cube_id: MemCubeID | str,
        mem_cube: GeneralMemCube,
        original_memory: list[TextualMemoryItem],
        new_memory: list[TextualMemoryItem],
    ) -> None | list[TextualMemoryItem]:
        """重新排序后用新内存替换工作内存"""
        text_mem_base = mem_cube.text_mem
        # 仅支持TreeTextMemory类型的文本内存
        if isinstance(text_mem_base, TreeTextMemory):
            text_mem_base: TreeTextMemory = text_mem_base

            # 获取当前用户-内存立方体的查询监控器
            query_db_manager = self.monitor.query_monitors[user_id][mem_cube_id]
            # 与数据库同步获取最新查询历史
            query_db_manager.sync_with_orm()

            # 获取按时间排序的查询历史
            query_history = query_db_manager.obj.get_queries_with_timesort()
            # 用LLM处理并重新排序内存
            memories_with_new_order, rerank_success_flag = (
                self.retriever.process_and_rerank_memories(
                    queries=query_history,
                    original_memory=original_memory,
                    new_memory=new_memory,
                    top_k=self.top_k,)
            )

            # 根据查询历史过滤完全不相关的内存
            filtered_memories, filter_success_flag = self.retriever.filter_unrelated_memories(
                query_history=query_history,
                memories=memories_with_new_order,
            )

            # 处理过滤结果
            if filter_success_flag:
                memories_with_new_order = filtered_memories

            # 获取查询关键词集合
            query_keywords = query_db_manager.obj.get_keywords_collections()

            # 将工作内存转换为监控项
            new_working_memory_monitors = self.transform_working_memories_to_monitors(
                query_keywords=query_keywords,
                memories=memories_with_new_order,
            )

            # 若重新排序失败，将排序分数设为0
            if not rerank_success_flag:
                for one in new_working_memory_monitors:
                    one.sorting_score = 0

            # 更新工作内存监控器
            self.monitor.update_working_memory_monitors(
                new_working_memory_monitors=new_working_memory_monitors,
                user_id=user_id,
                mem_cube_id=mem_cube_id,
                mem_cube=mem_cube,
            )

            # 获取排序后的内存监控项，转换为工作内存
            mem_monitors: list[MemoryMonitorItem] = self.monitor.working_memory_monitors[user_id][
                mem_cube_id
            ].obj.get_sorted_mem_monitors(reverse=True)
            new_working_memories = [mem_monitor.tree_memory_item for mem_monitor in mem_monitors]

            # 替换文本内存的工作内存
            text_mem_base.replace_working_memory(memories=new_working_memories)

            # 记录工作内存替换日志
            self.log_working_memory_replacement(
                original_memory=original_memory,
                new_memory=new_working_memories,
                user_id=user_id,
                mem_cube_id=mem_cube_id,
                mem_cube=mem_cube,
                log_func_callback=self._submit_web_logs,
            )
        else:
            memories_with_new_order = new_memory

        return memories_with_new_order

    def update_activation_memory(
        self,
        new_memories: list[str | TextualMemoryItem],
        label: str,
        user_id: UserID | str,
        mem_cube_id: MemCubeID | str,
        mem_cube: GeneralMemCube,
    ) -> None:
        """
        更新激活内存：从新内存中提取KV缓存项，添加到KV缓存内存实例并持久化到磁盘。
        支持字符串列表和TextualMemoryItem列表两种输入格式。
        """
        # 检查新内存是否为空
        if len(new_memories) == 0:
            return

        # 提取文本内容（适配两种输入格式）
        if isinstance(new_memories[0], TextualMemoryItem):
            new_text_memories = [mem.memory for mem in new_memories]
        elif isinstance(new_memories[0], str):
            new_text_memories = new_memories
        else:
            return

        try:
            # 获取激活内存实例（支持KVCacheMemory和VLLMKVCacheMemory）
            if isinstance(mem_cube.act_mem, VLLMKVCacheMemory):
                act_mem: VLLMKVCacheMemory = mem_cube.act_mem
            elif isinstance(mem_cube.act_mem, KVCacheMemory):
                act_mem: KVCacheMemory = mem_cube.act_mem
            else:
                 return

            # 用模板组装新的文本内存（过滤空字符串）
            new_text_memory = MEMORY_ASSEMBLY_TEMPLATE.format(
                memory_text="".join(
                    [
                        f"{i + 1}. {sentence.strip()}\n"
                        for i, sentence in enumerate(new_text_memories)
                        if sentence.strip()  # 跳过空字符串
                    ]
                )
            )

            # 处理已有缓存（避免重复更新）
            original_cache_items: list[VLLMKVCacheItem] = act_mem.get_all()
            original_text_memories = []
            if len(original_cache_items) > 0:
                pre_cache_item: VLLMKVCacheItem = original_cache_items[-1]
                original_text_memories = pre_cache_item.records.text_memories
                original_composed_text_memory = pre_cache_item.records.composed_text_memory
                # 若新组装的文本与已有缓存一致，跳过更新
                if original_composed_text_memory == new_text_memory:
                    return
                # 清除旧缓存
                act_mem.delete_all()

            # 提取KV缓存并添加到激活内存
            cache_item = act_mem.extract(new_text_memory)
            cache_item.records.text_memories = new_text_memories
            cache_item.records.timestamp = datetime.utcnow()
            act_mem.add([cache_item])
            # 持久化激活内存到磁盘
            act_mem.dump(self.act_mem_dump_path)

            # 记录激活内存更新日志
            self.log_activation_memory_update(
                original_text_memories=original_text_memories,
                new_text_memories=new_text_memories,
                label=label,
                user_id=user_id,
                mem_cube_id=mem_cube_id,
                mem_cube=mem_cube,
                log_func_callback=self._submit_web_logs,
            )

        except Exception as e:
            logger.error(f"MOS-based activation memory update failed: {e}", exc_info=True)
            # 关键操作失败时可重新抛出异常，此处暂不中断执行

    def update_activation_memory_periodically(
        self,
        interval_seconds: int,
        label: str,
        user_id: UserID | str,
        mem_cube_id: MemCubeID | str,
        mem_cube: GeneralMemCube,
    ):
        try:
            # 检查是否达到更新间隔（首次更新或间隔时间已到）
            if (
                self.monitor.last_activation_mem_update_time == datetime.min
                or self.monitor.timed_trigger(
                    last_time=self.monitor.last_activation_mem_update_time,
                    interval_seconds=interval_seconds,
                )
            ):
                # 检查工作内存监控器中是否有可用内存
                if (
                    user_id not in self.monitor.working_memory_monitors
                    or mem_cube_id not in self.monitor.working_memory_monitors[user_id]
                    or len(self.monitor.working_memory_monitors[user_id][mem_cube_id].obj.memories)
                    == 0
                ):
                    return
                
                # 更新激活内存监控器（同步最新内存状态）
                self.monitor.update_activation_memory_monitors(
                    user_id=user_id, mem_cube_id=mem_cube_id, mem_cube=mem_cube
                )

                # 与数据库同步获取最新激活内存
                activation_db_manager = self.monitor.activation_memory_monitors[user_id][
                    mem_cube_id
                ]
                activation_db_manager.sync_with_orm()
                new_activation_memories = [
                    m.memory_text for m in activation_db_manager.obj.memories
                ]

                # 日志输出新内存数量及部分内容（最多前 5 条）
                for i, memory in enumerate(new_activation_memories[:5], 1):
                    logger.info(
                        f"Part of New Activation Memorires | {i}/{len(new_activation_memories)}: {memory[:20]}"
                    )

                # 执行激活内存更新
                self.update_activation_memory(
                    new_memories=new_activation_memories,
                    label=label,
                    user_id=user_id,
                    mem_cube_id=mem_cube_id,
                    mem_cube=mem_cube,
                )
                
                # 更新最后更新时间
                self.monitor.last_activation_mem_update_time = datetime.utcnow()
            else:
                # 未达到更新间隔，跳过更新

        except Exception as e:
            logger.error(f"Error in update_activation_memory_periodically: {e}", exc_info=True)

    def submit_messages(self, messages: ScheduleMessageItem | list[ScheduleMessageItem]):
        """提交一个或多个消息到消息队列"""
        # 处理单个消息（转换为列表）
        if isinstance(messages, ScheduleMessageItem):
            messages = [messages]  # transform single message to list

        # 验证消息类型并加入队列
        for message in messages:
            if not isinstance(message, ScheduleMessageItem):
                error_msg = f"Invalid message type: {type(message)}, expected ScheduleMessageItem"
                raise TypeError(error_msg)

            self.memos_message_queue.put(message)

    def _message_consumer(self) -> None:
        """持续检查消息队列并分发消息（运行在独立线程中）定期消费队列中的所有消息，避免忙等"""
        while self._running:  # Use a running flag for graceful shutdown
            try:
                # 批量获取队列中所有可用消息（线程安全）
                messages = []
                while True: # 基于运行状态标识实现优雅关闭
                    try:
                        # 非阻塞获取消息，无消息时抛出 Empty 异常
                        message = self.memos_message_queue.get_nowait()
                        messages.append(message)
                    except queue.Empty:
                        # 队列已空，退出循环
                        break

                if messages: # 若获取到消息，执行分发
                    try:
                        self.dispatcher.dispatch(messages)
                    except Exception as e:
                        logger.error(f"Error dispatching messages: {e!s}")
                    finally:
                        # 标记所有消息为已处理
                        for _ in messages:
                            self.memos_message_queue.task_done()

                # 短暂休眠，避免 CPU 占用过高
                time.sleep(self._consume_interval)  # Adjust interval as needed

            except Exception as e:
                time.sleep(self._consume_interval)  # 异常时同样休眠，避免循环报错

    def start(self) -> None:
        """
            启动调度器（初始化资源并启动消息消费线程）
            启动内容：1. 消息消费线程 2. 分发器线程池（若启用并行分发）
        """        
        if self._running:
            return

        # 启动消息消费线程
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._message_consumer,
            daemon=True,
            name="MessageConsumerThread",
        )
        self._consumer_thread.start()
```

## 0xFF 参考
