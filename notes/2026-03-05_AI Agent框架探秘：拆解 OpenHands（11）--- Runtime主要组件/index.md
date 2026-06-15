# AI Agent框架探秘：拆解 OpenHands（11）--- Runtime主要组件

目录

* [AI Agent框架探秘：拆解 OpenHands（11）--- Runtime主要组件](#ai-agent框架探秘拆解-openhands11-----runtime主要组件)
  + [0x00 概要](#0x00-概要)
  + [0x01 三大组件](#0x01-三大组件)
  + [0x02 数据流](#0x02-数据流)
  + [0x03 插件系统](#0x03-插件系统)
    - [3.1 sandbox\_plugins](#31-sandbox_plugins)
      * [sandbox\_plugins 的定义和作用](#sandbox_plugins-的定义和作用)
      * [具体插件功能](#具体插件功能)
      * [插件在系统中的使用](#插件在系统中的使用)
    - [3.2 Plugin 基类](#32-plugin-基类)
    - [3.3 JupyterPlugin](#33-jupyterplugin)
      * [核心特色](#核心特色)
      * [流程图](#流程图)
      * [代码](#代码)
    - [3.4 AgentSkillsPlugin](#34-agentskillsplugin)
      * [功能概述](#功能概述)
      * [核心特色](#核心特色-1)
      * [AgentSkillsRequirement](#agentskillsrequirement)
      * [框架注册与技能发现](#框架注册与技能发现)
      * [流程图](#流程图-1)
      * [代码](#代码-1)
  + [0x04 执行系统](#0x04-执行系统)
    - [4.1 调用](#41-调用)
    - [4.2action\_execution\_client.py](#42action_execution_clientpy)
    - [4.3 action\_execution\_server.py](#43-action_execution_serverpy)
    - [4.4 流程图](#44-流程图)
    - [4.5 代码](#45-代码)
  + [0x05 环境](#0x05-环境)
    - [5.1 调用](#51-调用)
    - [5.2 核心特色](#52-核心特色)
    - [5.3 流程图](#53-流程图)
    - [5.4 代码](#54-代码)
  + [0xFF 参考](#0xff-参考)

## 0x00 概要

本篇继续对 runtime 的解读，主要介绍 插件、执行系统和环境这三个组件。

因为本系列借鉴的文章过多，可能在参考文献中有遗漏的文章，如果有，还请大家指出。

## 0x01 三大组件

本篇要介绍的几个组件如下：

* ActionExecutor：在 Runtime 中执行动作的核心组件
  + ActionExecutor 初始化时会根据配置加载指定的插件。插件注册到 ActionExecutor 的插件字典。
  + 当接收到动作请求时，ActionExecutor 会调用相应的方法执行动作。
  + 对于浏览动作，ActionExecutor 会使用 BrowserEnv 来处理。
  + 如果涉及插件，ActionExecutor 会通过插件系统处理
* AgentSkillsPlugin：提供智能体技能功能的插件
  + AgentSkillsPlugin 是一个插件，继承自 Plugin 基类。
  + Runtime 初始化时，插件会被加载到插件字典中。 插件通过 PluginRequirement 机制被注册到系统中。
  + 特定动作触发时调用相应插件功能。
* BrowserEnv：浏览器环境封装，使用 BrowserGym 库。
  + ActionExecutor 在初始化时根据配置决定是否启用浏览器环境。
  + 当需要执行浏览相关的动作时，ActionExecutor 会调用 BrowserEnv 的方法。
  + BrowserEnv 运行在一个独立的多进程环境中。

## 0x02 数据流

Runtime 的数据流如下：

* Runtime 会发起动作请求 → ActionExecutor.run\_action()
* ActionExecutor 根据动作类型调用相应的处理方法；
* 如果涉及插件，通过插件系统处理；
* 如果涉及浏览器，调用 BrowserEnv 处理；
* 返回观察结果给智能体。

![Runtime-组件](images/img_001.png)

## 0x03 插件系统

Runtime会遇到如下问题：新增模块（如自定义工具、新 LLM 模型）时，需修改核心代码，扩展性差；多任务并发执行时，模块间交互频繁，易出现性能瓶颈；框架部署与运维复杂，难以适配不同环境（本地、云端、边缘端）。

因此，业界大多采用微服务架构或插件化设计，模块间通过标准化接口通信，新增功能只需开发插件并注册。

### 3.1 sandbox\_plugins

sandbox\_plugins 在 OpenHands 的 CodeActAgent 中起到了关键作用，主要用于定义和配置代理在沙箱环境中可以使用的工具和功能。这些插件是代理能够与环境交互并完成任务的基础工具集。

#### sandbox\_plugins 的定义和作用

在 CodeActAgent 类中，sandbox\_plugins 是一个类属性，定义了代理在沙箱环境中需要的插件：

```
sandbox_plugins: list[PluginRequirement] = [
    AgentSkillsRequirement(),
    JupyterRequirement(),
]
```

这些插件为代理提供了在沙箱环境中执行任务所需的工具和功能。

#### 具体插件功能

AgentSkillsRequirement 和 JupyterRequirement 是两个插件需求类。

* AgentSkillsRequirement：提供了一系列 Python 函数和工具，使代理能够执行各种操作，包括文件操作、目录浏览、代码执行等基本技能。需要在 JupyterRequirement 之前初始化，因为 Jupyter 需要使用这些函数。
* JupyterRequirement：提供了交互式 Python 解释器环境，允许代理执行 Python 代码，依赖于 AgentSkillsRequirement 提供的函数。

#### 插件在系统中的使用

从代码中可以看出，这些插件在多个地方被使用：

* 在 Runtime 初始化时：

```
# 在 agent_session.py 中
self.runtime = runtime_cls(
    plugins=agent.sandbox_plugins,
)
```

* 在 Runtime 中设置插件：

```
# 在 base.py 中
self.plugins = copy.deepcopy(plugins) if plugins is not None and len(plugins) > 0 else []
```

这些插件为代理提供了以下能力：

1. 执行 Bash 命令：通过 AgentSkills 中的命令执行功能
2. 执行 Python 代码：通过 Jupyter 插件提供 IPython 环境
3. 文件系统操作：读取、写入、编辑文件
4. 目录浏览：查看和导航文件系统
5. 其他实用工具：各种辅助函数和工具

我们接下来具体分析基类Plugin，AgentSkillsRequirement 和 JupyterPlugin

### 3.2 Plugin 基类

```
class Plugin:
    """Base class for a plugin.

    This will be initialized by the runtime client, which will run inside docker.
    """

    name: str

    @abstractmethod
    async def initialize(self, username: str) -> None:
        """Initialize the plugin."""
        pass

    @abstractmethod
    async def run(self, action: Action) -> Observation:
        """Run the plugin for a given action."""
        pass

@dataclass
class PluginRequirement:
    """Requirement for a plugin."""

    name: str
```

插件为：

```
ALL_PLUGINS = {
    'jupyter': JupyterPlugin,
    'agent_skills': AgentSkillsPlugin,
    'vscode': VSCodePlugin,
}
```

### 3.3 JupyterPlugin

`JupyterPlugin` 是 OpenHands 框架中的 Jupyter 内核插件，基于 `Plugin` 基类实现，核心职责是启动 Jupyter Kernel Gateway（内核网关）服务，提供 IPython 代码单元格的异步执行能力，支持代码运行、输出捕获（文本 / 图片）及 Python 解释器路径获取，是框架中集成交互式数据分析、代码调试等 Jupyter 相关功能的核心组件。

#### 核心特色

1. **跨平台适配**：兼容 Windows、Linux、macOS 系统，针对不同系统采用差异化的进程启动方式（Windows 用 `subprocess.Popen`，类 Unix 用 `asyncio.create_subprocess_shell`）。
2. **灵活的运行时支持**：区分本地运行时（LocalRuntime）与非本地运行时，适配不同部署场景（如沙箱环境、本地开发环境），自动处理工作目录与环境变量配置。
3. **端口自动分配**：在 40000-49999 端口范围内自动查找可用 TCP 端口，避免端口冲突。
4. **异步代码执行**：基于 `JupyterKernel` 封装异步代码执行逻辑，支持超时控制，能捕获文本输出与图片 URL 等结构化结果。
5. **环境隔离与兼容**：通过 `micromamba` 虚拟环境或本地环境变量确保依赖一致性，支持 Poetry 项目的路径配置，适配 OpenHands 框架的工程化部署。

#### 流程图

![11-1](images/img_002.png)

#### 代码

```
@dataclass
class JupyterRequirement(PluginRequirement):
    """Jupyter插件的依赖声明类，用于框架识别插件依赖。"""
    name: str = 'jupyter'  # 依赖名称，固定为'jupyter'

class JupyterPlugin(Plugin):
    """Jupyter插件，提供Jupyter Kernel Gateway启动与IPython代码执行能力。"""
    name: str = 'jupyter'  # 插件名称，固定为'jupyter'
    kernel_gateway_port: int  # Jupyter Kernel Gateway服务端口
    kernel_id: str  # Jupyter内核ID
    gateway_process: asyncio.subprocess.Process | subprocess.Popen  # 内核网关进程对象
    python_interpreter_path: str  # Python解释器路径

    async def initialize(
        self, username: str, kernel_id: str = 'openhands-default'
    ) -> None:
        """初始化Jupyter插件，启动Kernel Gateway服务，配置运行环境。

        参数:
            username: 执行用户名称（非本地运行时使用）
            kernel_id: Jupyter内核ID（默认：openhands-default）
        """
        # 在40000-49999端口范围内查找可用TCP端口，避免冲突
        self.kernel_gateway_port = find_available_tcp_port(40000, 49999)
        self.kernel_id = kernel_id
        # 判断是否为本地运行时（通过环境变量LOCAL_RUNTIME_MODE标记）
        is_local_runtime = os.environ.get('LOCAL_RUNTIME_MODE') == '1'
        # 判断是否为Windows系统
        is_windows = sys.platform == 'win32'

        if not is_local_runtime:
            # 非本地运行时：配置用户切换前缀与Poetry虚拟环境
            # 若启用SU_TO_USER，则添加"su - 用户名 -s "前缀（切换用户执行命令）
            prefix = f'su - {username} -s ' if SU_TO_USER else ''
            # 命令前缀：切换到代码仓库目录，配置环境变量，使用micromamba虚拟环境
            poetry_prefix = (
                'cd /openhands/code\n'
                'export POETRY_VIRTUALENVS_PATH=/openhands/poetry;\n'
                'export PYTHONPATH=/openhands/code:$PYTHONPATH;\n'
                'export MAMBA_ROOT_PREFIX=/openhands/micromamba;\n'
                '/openhands/micromamba/bin/micromamba run -n openhands '
            )
        else:
            # 本地运行时：无需用户切换，直接使用本地环境
            prefix = ''
            # 从环境变量获取代码仓库路径（本地运行时必须配置）
            code_repo_path = os.environ.get('OPENHANDS_REPO_PATH')
            if not code_repo_path:
                raise ValueError(
                    'OPENHANDS_REPO_PATH environment variable is not set. '
                    'This is required for the jupyter plugin to work with LocalRuntime.'
                )
            # 命令前缀：切换到代码仓库目录（本地环境依赖PATH确保环境正确）
            poetry_prefix = f'cd {code_repo_path}\n'

        if is_windows:
            # Windows系统：构建CMD格式的启动命令
            jupyter_launch_command = (
                f'cd /d "{code_repo_path}" && '  # 切换到代码仓库目录（/d参数支持跨盘符切换）
                f'"{sys.executable}" -m jupyter kernelgateway '  # 启动Jupyter Kernel Gateway
                '--KernelGatewayApp.ip=0.0.0.0 '  # 绑定所有网络接口
                f'--KernelGatewayApp.port={self.kernel_gateway_port}'  # 指定端口
            )

            # Windows系统使用同步subprocess.Popen启动进程（asyncio在Windows有兼容性限制）
            self.gateway_process = subprocess.Popen(  # type: ignore[ASYNC101] # noqa: ASYNC101
                jupyter_launch_command,
                stdout=subprocess.PIPE,  # 捕获标准输出
                stderr=subprocess.STDOUT,  # 标准错误重定向到标准输出
                shell=True,  # 使用shell执行命令
                text=True,  # 输出以文本模式返回
            )

            # Windows系统同步等待Kernel Gateway启动（读取输出直到包含'at'字符，标识服务就绪）
            output = ''
            while should_continue():
                if self.gateway_process.stdout is None:
                    time.sleep(1)  # 无输出时等待1秒
                    continue

                line = self.gateway_process.stdout.readline()  # 读取一行输出
                if not line:
                    time.sleep(1)
                    continue

                output += line
                if 'at' in line:  # 服务启动成功的标识（输出含"at"，如"Listening at..."）
                    break

                time.sleep(1)
        else:
            # 类Unix系统（Linux/macOS）：构建Bash格式的启动命令
            jupyter_launch_command = (
                f"{prefix}/bin/bash << 'EOF'\n"  # 切换到bash执行，EOF避免变量解析
                f'{poetry_prefix}'  # 环境配置前缀（虚拟环境/工作目录）
                f'"{sys.executable}" -m jupyter kernelgateway '  # 启动Kernel Gateway
                '--KernelGatewayApp.ip=0.0.0.0 '  # 绑定所有网络接口
                f'--KernelGatewayApp.port={self.kernel_gateway_port}\n'  # 指定端口
                'EOF'
            )

            # 类Unix系统使用asyncio创建异步子进程（避免阻塞事件循环）
            self.gateway_process = await asyncio.create_subprocess_shell(
                jupyter_launch_command,
                stderr=asyncio.subprocess.STDOUT,  # 标准错误重定向到标准输出
                stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
            )
            # 异步等待Kernel Gateway启动（读取输出直到包含'at'字符）
            output = ''
            while should_continue() and self.gateway_process.stdout is not None:
                line_bytes = await self.gateway_process.stdout.readline()  # 异步读取一行输出
                line = line_bytes.decode('utf-8')  # 字节转字符串
                output += line
                if 'at' in line:
                    break
                await asyncio.sleep(1)  # 等待1秒

        # 执行测试代码，获取当前Python解释器路径（验证环境正确性）
        _obs = await self.run(
            IPythonRunCellAction(code='import sys; print(sys.executable)')
        )
        self.python_interpreter_path = _obs.content.strip()  # 提取并保存解释器路径

    async def _run(self, action: Action) -> IPythonRunCellObservation:
        """内部方法：在Jupyter内核中执行代码单元格。

        参数:
            action: 待执行的动作（必须是IPythonRunCellAction类型）

        返回:
            IPythonRunCellObservation: 代码执行结果的观察值（含文本内容、图片URL等）
        """
        # 校验动作类型：仅支持IPythonRunCellAction
        if not isinstance(action, IPythonRunCellAction):
            raise ValueError(
                f'Jupyter plugin only supports IPythonRunCellAction, but got {action}'
            )

        # 初始化JupyterKernel（若未初始化）
        if not hasattr(self, 'kernel'):
            self.kernel = JupyterKernel(
                f'localhost:{self.kernel_gateway_port}',  # 内核网关地址（本地+端口）
                self.kernel_id  # 内核ID
            )

        # 若内核未初始化，执行初始化（建立连接）
        if not self.kernel.initialized:
            await self.kernel.initialize()

        # 异步执行代码，支持超时控制（超时时间从action获取）
        output = await self.kernel.execute(action.code, timeout=action.timeout)

        # 从结构化输出中提取文本内容与图片URL
        text_content = output.get('text', '')  # 文本输出（stdout/stderr）
        image_urls = output.get('images', [])  # 图片URL列表（如matplotlib绘图结果）

        # 返回封装后的观察结果
        return IPythonRunCellObservation(
            content=text_content,  # 文本内容
            code=action.code,  # 执行的代码
            image_urls=image_urls if image_urls else None,  # 图片URL（无则为None）
        )

    async def run(self, action: Action) -> IPythonRunCellObservation:
        """公开接口：执行IPython代码动作，返回观察结果。

        参数:
            action: 待执行的IPythonRunCellAction动作

        返回:
            IPythonRunCellObservation: 代码执行结果
        """
        # 调用内部_run方法执行代码，返回结果
        obs = await self._run(action)
        return obs
```

### 3.4 AgentSkillsPlugin

#### 功能概述

`AgentSkillsPlugin` 是 OpenHands 框架中管理智能体技能（Agent Skills）的核心插件，负责整合文件操作（`file_ops`）、文件读取（`file_reader`）、代码仓库操作（`repo_ops`）等基础技能模块，通过动态导入机制将分散的技能函数统一暴露给框架，同时提供插件依赖声明与文档自动生成能力，是智能体获取文件处理、仓库管理等核心操作能力的关键组件。

```
class AgentSkillsPlugin(Plugin):
    name: str = 'agent_skills'

    async def initialize(self, username: str) -> None:
        """Initialize the plugin."""
        pass

    async def run(self, action: Action) -> Observation:
        """Run the plugin for a given action."""
        raise NotImplementedError('AgentSkillsPlugin does not support run method')
```

#### 核心特色

1. **模块化技能整合**：通过动态导入机制，将 `file_ops`、`file_reader`、`repo_ops` 等独立模块的技能函数统一聚合，简化框架对技能的调用与管理。
2. **自动文档生成**：扫描所有导入的技能函数，提取函数签名与文档字符串（`__doc__`），自动生成标准化文档，提升开发可维护性。
3. **柔性依赖处理**：对 `repo_ops` 模块采用可选导入策略，导入失败时仅跳过该模块，不影响其他技能的正常使用，增强插件兼容性。
4. **极简初始化设计**：插件初始化逻辑为空实现，无需额外配置，聚焦于技能函数的聚合与暴露，降低使用门槛。
5. **明确的接口约束**：禁用 `run` 方法（抛出未实现异常），明确该插件的核心作用是技能聚合而非直接执行动作，避免误用。

#### AgentSkillsRequirement

AgentSkillsRequirement 是一个插件需求类，它定义了代理在沙箱环境中运行所需的基本技能集合，这些技能主要通过 Python 函数的形式提供，使代理能够执行各种操作：

* 为代理提供了与文件系统交互的基本能力
* 提供了执行命令和脚本的工具
* 为其他高级插件（如 Jupyter）提供了基础函数支持
* 确保代理能够在沙箱环境中完成大多数常见的开发任务

AgentSkillsRequirement 的主要功能如下：

* 文件系统操作
  + 提供读取、写入、编辑文件的能力
  + 支持目录浏览和文件管理操作
  + 允许代理查看和操作工作区中的文件
* 命令执行
  + 提供执行 shell 命令的能力
  + 允许代理在沙箱环境中运行 bash 命令
  + 支持与操作系统交互的各种操作
* 工具函数集合
  + 提供一系列实用的 Python 函数
  + 这些函数可以被其他插件（如 Jupyter）使用
  + 包括各种辅助功能，如字符串处理、数据操作等

在 CodeActAgent 中，AgentSkillsRequirement 被定义在 sandbox\_plugins 列表中：

```
sandbox_plugins: list[PluginRequirement] = [
    AgentSkillsRequirement(),
    JupyterRequirement(),
]
```

AgentSkillsRequirement 与其他组件的关系：

* 与 JupyterRequirement 的关系
  + AgentSkillsRequirement 必须在 JupyterRequirement 之前初始化
  + AgentSkillsRequirement 提供的 Python 函数会被 Jupyter 环境使用
  + 这种顺序确保了 Jupyter 可以访问所有必要的工具函数
* 与 Runtime 的关系
  + 在 LocalRuntime 和其他运行时环境中，这些插件会被加载和初始化

总的来说，AgentSkillsRequirement 是代理在 OpenHands 环境中执行任务的基础，它提供了一套核心函数，使代理能够与文件系统、命令行和运行环境进行交互。

#### 框架注册与技能发现

OpenHands 框架通过「插件注册机制」识别 `AgentSkillsPlugin`，并自动发现其聚合的所有具体 Skill 操作，步骤如下：

1. 插件注册与依赖声明

`AgentSkillsPlugin` 继承自框架的 `Plugin` 基类，通过 `AgentSkillsRequirement` 声明依赖，框架启动时会自动扫描并加载该插件：

```
@dataclass
class AgentSkillsRequirement(PluginRequirement):
    name: str = "agent_skills"  # 插件依赖名称，与插件名一致
    documentation: str = agentskills.DOCUMENTATION    

class AgentSkillsPlugin(Plugin):
    name: str = "agent_skills"  # 插件名称，框架通过该名称识别
```

2. 框架解析技能清单

框架加载 `AgentSkillsPlugin` 后，会读取其 `__all__` 变量和全局命名空间，提取所有 Skill 函数的关键信息：

* 函数名（如 `create_file`）：作为 Skill 的唯一标识；
* 函数签名（参数、返回值）：通过 `inspect.signature` 解析，用于智能体构造调用参数；
* 文档字符串（`__doc__`）：自动生成技能文档，供智能体参考使用。

3. 技能全局注册

框架将解析后的 Skill 信息注册到「全局技能注册表」中，形成 key-value 映射（key：Skill 函数名，value：Skill 函数对象 + 元数据），使智能体可通过函数名快速查找并调用对应 Skill。

4. 智能体调用具体 Skill 操作

智能体（Agent）通过框架提供的接口，从「全局技能注册表」中获取 `AgentSkillsPlugin` 聚合的具体 Skill，并触发执行

```
logger.debug('Initializing AgentSkills')
if 'agent_skills' in self.plugins and 'jupyter' in self.plugins:
    obs = await self.run_ipython(
        IPythonRunCellAction(
            code='from openhands.runtime.plugins.agent_skills.agentskills import *\n'
        )
    )
    logger.debug(f'AgentSkills initialized: {obs}')
```

#### 流程图

![11-1.5](images/img_003.png)

#### 代码

```
@dataclass
class AgentSkillsRequirement(PluginRequirement):
    """AgentSkillsPlugin 的依赖声明类，用于框架识别插件依赖。"""
    name: str = 'agent_skills'  # 依赖名称，固定为'agent_skills'
    documentation: str = agentskills.DOCUMENTATION  # 依赖文档（来自agentskills模块）

class AgentSkillsPlugin(Plugin):
    """智能体技能插件，负责聚合各类基础技能函数（文件操作、仓库操作等）。"""
    name: str = 'agent_skills'  # 插件名称，固定为'agent_skills'

    async def initialize(self, username: str) -> None:
        """初始化插件（空实现，无需额外配置）。"""
        pass

    async def run(self, action: Action) -> Observation:
        """执行插件动作（禁用该方法）。

        该插件的核心作用是聚合技能函数，而非直接执行动作，因此抛出未实现异常。
        """
        raise NotImplementedError('AgentSkillsPlugin does not support run method')

# 动态导入file_ops模块的所有技能函数，添加到当前模块全局命名空间
import_functions(
    module=file_ops,  # 源模块：文件操作模块（如创建/删除/修改文件）
    function_names=file_ops.__all__,  # 导入的函数列表（file_ops定义的所有公开函数）
    target_globals=globals()  # 目标命名空间：当前模块全局变量
)

# 动态导入file_reader模块的所有技能函数，添加到当前模块全局命名空间
import_functions(
    module=file_reader,  # 源模块：文件读取模块（如读取文本文件、解析JSON等）
    function_names=file_reader.__all__,  # 导入的函数列表
    target_globals=globals()
)

# 初始化__all__列表，包含已导入的所有技能函数（供外部模块导入）
__all__ = file_ops.__all__ + file_reader.__all__

# 可选导入repo_ops模块（代码仓库操作，如Git克隆、提交等）
try:
    from openhands.runtime.plugins.agent_skills import repo_ops

    # 动态导入repo_ops模块的所有技能函数
    import_functions(
        module=repo_ops,
        function_names=repo_ops.__all__,
        target_globals=globals()
    )

    # 将repo_ops的技能函数添加到__all__
    __all__ += repo_ops.__all__
except ImportError:
    # 若repo_ops模块不可用（如未安装依赖），则跳过导入，不影响其他功能
    pass

# 自动生成所有技能函数的标准化文档
DOCUMENTATION = ''
for func_name in __all__:
    # 从全局命名空间获取技能函数
    func = globals()[func_name]

    # 获取函数的文档字符串
    cur_doc = func.__doc__
    # 清理文档字符串：去除空行、统一去除每行缩进
    cur_doc = '\n'.join(filter(None, map(lambda x: x.strip(), cur_doc.split('\n'))))
    # 格式化文档：每行添加4个空格缩进，保证格式统一
    cur_doc = '\n'.join(map(lambda x: ' ' * 4 + x, cur_doc.split('\n')))

    # 提取函数签名（函数名+参数列表）
    fn_signature = f'{func.__name__}' + str(signature(func))
    # 将函数签名与格式化文档添加到总文档
    DOCUMENTATION += f'{fn_signature}:\n{cur_doc}\n\n'

# 单独添加file_editor技能函数（特殊处理，未包含在上述模块中）
from openhands.runtime.plugins.agent_skills.file_editor import file_editor  # noqa: E402
__all__ += ['file_editor']  # 将file_editor添加到__all__，供外部导入
```

## 0x04 执行系统

`ActionExecutor` 是 OpenHands 框架中运行于 Docker 沙箱内的核心动作执行组件，负责接收并执行来自后端的各类动作（如命令行执行、IPython 代码运行、浏览器操作等），生成对应的观察结果（Observation），同时管理插件生命周期、用户环境、工作目录及资源监控，是连接后端指令与沙箱执行环境的关键桥梁。

### 4.1 调用

在 OpenHands 项目中，ActionExecutor 类的调用与使用流程总结

1. 服务端：ActionExecutor 在 action\_execution\_server.py 中作为独立服务运行。
2. 客户端：各种运行时实现（如 LocalRuntime）通过 HTTP 请求与 ActionExecutor 通信。

执行步骤：

* Runtime（实际就是ActionExecutionClient的派生类）直接或间接调用 execute\_action() 方法；
  + class RemoteRuntime(ActionExecutionClient)
  + class LocalRuntime(ActionExecutionClient)
  + class DockerRuntime(ActionExecutionClient)
* 通过 HTTP POST 请求发送到 /execute\_action 端点；
* ActionExecutor 接收请求并执行相应操作；
* 将观察结果返回给客户端。

这种架构把操作执行与主应用分离，提供了更好的隔离性和安全性。

### 4.2action\_execution\_client.py

`action_execution_client.py` 文件包含`ActionExecutionClient`类，它实现了运行时接口。它是一个抽象实现，意味着仍需要通过具体实现来扩展才能使用。

ActionExecutionClient 通过HTTP调用与action\_execution\_server 交互以实际执行运行时操作。

ActionExecutionClient 在各种运行时实现中被使用。例如，在 LocalRuntime 中，操作会通过客户端发送到 ActionExecutor。

### 4.3 action\_execution\_server.py

ActionExecutor 在 openhands/runtime/action\_execution\_server.py 文件中充当核心组件。该文件将其实例化，并作为 FastAPI 应用程序的核心执行器。即，`action_executor.py` 文件包含`ActionExecutor`类，负责通过`/execute_action` HTTP端点接收的动作。它在HTTP响应中返回观察结果。

核心特色如下

1. **沙箱内环境管理**：初始化用户权限与工作目录，支持 Windows/Linux 跨平台环境适配，确保执行隔离性与安全性。
2. **插件化架构**：支持加载 VSCode、Jupyter 等各类插件，通过统一接口管理插件初始化与调用，灵活扩展执行能力。
3. **多环境协同**：集成 Bash/Windows PowerShell 命令行环境、浏览器环境（BrowserEnv）、Jupyter 交互式环境，满足多样化动作执行需求。
4. **异步初始化优化**：浏览器环境采用异步初始化，避免阻塞主流程，提升启动效率。
5. **资源与状态监控**：支持内存上限配置与内存监控，同步 Bash 与 Jupyter 工作目录，确保执行上下文一致性。
6. **异常处理与兼容性**：针对 Windows 系统、插件缺失等场景做特殊兼容处理，抛出明确异常并记录日志。

具体会：

在main中启动

```
if __name__ == '__main__':
    logger.debug(f'Starting action execution API on port {args.port}')
    # When LOG_JSON=1, provide a JSON log config to Uvicorn so error/access logs are structured
    log_config = None
    if os.getenv('LOG_JSON', '0') in ('1', 'true', 'True'):
        log_config = get_uvicorn_json_log_config()
    run(app, host='0.0.0.0', port=args.port, log_config=log_config, use_colors=False)
```

在 lifespan 函数中初始化

```
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global client, mcp_proxy_manager
        logger.info('Initializing ActionExecutor...')
        client = ActionExecutor(
            plugins_to_load,
            work_dir=args.working_dir,
            username=args.username,
            user_id=args.user_id,
            enable_browser=args.enable_browser,
            browsergym_eval_env=args.browsergym_eval_env,
        )
        await client.ainit()
        logger.info('ActionExecutor initialized.')

        # Check if we're on Windows
        is_windows = sys.platform == 'win32'

        # Initialize and mount MCP Proxy Manager (skip on Windows)
        if is_windows:
            logger.info('Skipping MCP Proxy initialization on Windows')
            mcp_proxy_manager = None
        else:
            logger.info('Initializing MCP Proxy Manager...')
            # Create a MCP Proxy Manager
            mcp_proxy_manager = MCPProxyManager(
                auth_enabled=bool(SESSION_API_KEY),
                api_key=SESSION_API_KEY,
                logger_level=logger.getEffectiveLevel(),
            )
            mcp_proxy_manager.initialize()
            # Mount the proxy to the app
            allowed_origins = ['*']
            try:
                await mcp_proxy_manager.mount_to_app(app, allowed_origins)
            except Exception as e:
                logger.error(f'Error mounting MCP Proxy: {e}', exc_info=True)
                raise RuntimeError(f'Cannot mount MCP Proxy: {e}')

        yield

        # Clean up & release the resources
        logger.info('Shutting down MCP Proxy Manager...')
        if mcp_proxy_manager:
            del mcp_proxy_manager
            mcp_proxy_manager = None
        else:
            logger.info('MCP Proxy Manager instance not found for shutdown.')

        logger.info('Closing ActionExecutor...')
        if client:
            try:
                client.close()
                logger.info('ActionExecutor closed successfully.')
            except Exception as e:
                logger.error(f'Error closing ActionExecutor: {e}', exc_info=True)
        else:
            logger.info('ActionExecutor instance not found for closing.')
        logger.info('Shutdown complete.')
```

在 /execute\_action 端点中被调用

```
    @app.post('/execute_action')
    async def execute_action(action_request: ActionRequest):
        assert client is not None
        try:
            action = event_from_dict(action_request.action)
            if not isinstance(action, Action):
                raise HTTPException(status_code=400, detail='Invalid action type')
            client.last_execution_time = time.time()
            observation = await client.run_action(action)
            return event_to_dict(observation)
        except Exception as e:
            logger.exception(f'Error while running /execute_action: {str(e)}')
            raise HTTPException(
                status_code=500,
                detail=f'Internal server error: {str(e)}',
            )
        finally:
            update_last_execution_time()
```

### 4.4 流程图

![11-3](images/img_004.png)

### 4.5 代码

`ActionExecutor`类的关键特性：

* 初始化用户环境和bash shell
* 插件管理和初始化
* 执行各种操作类型（bash命令、IPython单元、文件操作、浏览）
* 与BrowserEnv集成以进行网络交互

```
class ActionExecutor:
    """动作执行器（ActionExecutor）运行于 Docker 沙箱内，
    负责执行从 OpenHands 后端接收的动作，并生成对应的观察结果（Observation）。
    """

    def __init__(
        self,
        plugins_to_load: list[Plugin],
        work_dir: str,
        username: str,
        user_id: int,
        enable_browser: bool,
        browsergym_eval_env: str | None,
    ) -> None:
        """初始化动作执行器，配置执行环境、插件列表、用户信息等核心参数。

        参数:
            plugins_to_load: 待加载的插件列表
            work_dir: 初始工作目录路径
            username: 执行用户名称
            user_id: 执行用户ID
            enable_browser: 是否启用浏览器环境
            browsergym_eval_env: BrowserGym 评估环境名称（可选，启用浏览器时生效）
        """
        # 待加载的插件列表
        self.plugins_to_load = plugins_to_load
        # 初始工作目录（沙箱内路径）
        self._initial_cwd = work_dir
        # 执行用户名称
        self.username = username
        # 执行用户ID
        self.user_id = user_id

        # 初始化用户与工作目录（设置用户权限、创建工作目录，返回更新后的用户ID）
        _updated_user_id = init_user_and_working_directory(
            username=username, user_id=self.user_id, initial_cwd=work_dir
        )
        if _updated_user_id is not None:
            self.user_id = _updated_user_id  # 更新为实际生效的用户ID

        # 命令行会话（支持 Bash 或 Windows PowerShell）
        self.bash_session: BashSession | 'WindowsPowershellSession' | None = None  # type: ignore[name-defined]
        # 异步锁，确保动作执行的线程安全
        self.lock = asyncio.Lock()
        # 已加载的插件字典（key: 插件名称，value: 插件实例）
        self.plugins: dict[str, Plugin] = {}
        # 文件编辑器实例（基于工作目录根路径初始化）
        self.file_editor = OHEditor(workspace_root=self._initial_cwd)
        # 是否启用浏览器环境
        self.enable_browser = enable_browser
        # 浏览器环境实例（BrowserEnv）
        self.browser: BrowserEnv | None = None
        # 浏览器异步初始化任务（避免阻塞主流程）
        self.browser_init_task: asyncio.Task | None = None
        # BrowserGym 评估环境名称
        self.browsergym_eval_env = browsergym_eval_env

        # 合法性校验：未启用浏览器时，不允许设置 BrowserGym 评估环境
        if (not self.enable_browser) and self.browsergym_eval_env:
            raise BrowserUnavailableException(
                'Browser environment is not enabled in config, but browsergym_eval_env is set'
            )

        # 记录启动时间与最后执行时间
        self.start_time = time.time()
        self.last_execution_time = self.start_time
        # 初始化完成标记
        self._initialized = False
        # 已下载文件列表
        self.downloaded_files: list[str] = []
        # 下载目录路径（沙箱内）
        self.downloads_directory = '/workspace/.downloads'

        # 内存上限配置（从环境变量读取，可选）
        self.max_memory_gb: int | None = None
        if _override_max_memory_gb := os.environ.get('RUNTIME_MAX_MEMORY_GB', None):
            self.max_memory_gb = int(_override_max_memory_gb)
        else:
            logger.info('No max memory limit set, using all available system memory')

        # 初始化内存监控（根据环境变量决定是否启用）
        self.memory_monitor = MemoryMonitor(
            enable=os.environ.get('RUNTIME_MEMORY_MONITOR', 'False').lower()
            in ['true', '1', 'yes']
        )
        self.memory_monitor.start_monitoring()  # 启动内存监控

    async def _init_browser_async(self):
        """异步初始化浏览器环境（避免阻塞主流程）。"""
        if not self.enable_browser:
            logger.info('Browser environment is not enabled in config')
            return

        # Windows 系统不支持浏览器环境，输出警告
        if sys.platform == 'win32':
            logger.warning('Browser environment not supported on windows')
            return
        try:
            # 初始化浏览器环境（传入评估环境名称）
            self.browser = BrowserEnv(self.browsergym_eval_env)
            logger.debug('Browser initialized asynchronously')
        except Exception as e:
            logger.exception(f'Failed to initialize browser: {e}')
            self.browser = None  # 初始化失败则置空

    async def _init_plugin(self, plugin: Plugin):
        """初始化单个插件，并注册到插件字典。

        参数:
            plugin: 待初始化的插件实例
        """
        assert self.bash_session is not None, "命令行会话未初始化，无法加载插件"

        # VSCode 插件特殊处理：需要 Runtime ID 用于 Gateway API 的路径路由
        if isinstance(plugin, VSCodePlugin):
            runtime_id = os.environ.get('RUNTIME_ID')  # 从环境变量获取 Runtime ID
            await plugin.initialize(self.username, runtime_id=runtime_id)
        else:
            # 其他插件直接传入用户名初始化
            await plugin.initialize(self.username)

        # 将初始化后的插件注册到字典
        self.plugins[plugin.name] = plugin
        logger.debug(f'Initializing plugin: {plugin.name}')

        # Jupyter 插件特殊处理：同步命令行工作目录到 Jupyter 环境
        if isinstance(plugin, JupyterPlugin):
            # Windows 路径转义（将反斜杠替换为正斜杠）
            cwd = self.bash_session.cwd.replace('\\', '/')
            # 执行 IPython 代码，切换工作目录
            await self.run_ipython(
                IPythonRunCellAction(code=f'import os; os.chdir(r"{cwd}")')
            )

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        """执行 IPython 代码动作，返回执行结果观察值。

        参数:
            action: IPython 代码执行动作（包含待执行代码、是否补充额外信息等）

        返回:
            IPythonRunCellObservation: 代码执行结果的观察值（含输出内容、状态等）
        """
        assert self.bash_session is not None, "命令行会话未初始化，无法执行 IPython 动作"

        # 检查 Jupyter 插件是否已加载
        if 'jupyter' in self.plugins:
            _jupyter_plugin: JupyterPlugin = self.plugins['jupyter']  # 类型断言
            # 同步 Bash 与 Jupyter 的工作目录（确保执行上下文一致）
            jupyter_cwd = getattr(self, '_jupyter_cwd', None)  # 获取当前 Jupyter 工作目录
            if self.bash_session.cwd != jupyter_cwd:
                # Windows 路径转义
                cwd = self.bash_session.cwd.replace('\\', '/')
                # 生成切换工作目录的代码
                reset_jupyter_cwd_code = f'import os; os.chdir("{cwd}")'
                _aux_action = IPythonRunCellAction(code=reset_jupyter_cwd_code)
                # 执行工作目录切换
                _reset_obs: IPythonRunCellObservation = await _jupyter_plugin.run(
                    _aux_action
                )
                self._jupyter_cwd = self.bash_session.cwd  # 更新 Jupyter 工作目录缓存

            # 执行目标 IPython 代码
            obs: IPythonRunCellObservation = await _jupyter_plugin.run(action)
            obs.content = obs.content.rstrip()  # 去除输出内容末尾的空白字符

            # 若需要补充额外信息，添加工作目录与 Python 解释器路径
            if action.include_extra:
                obs.content += (
                    f'\n[Jupyter current working directory: {self.bash_session.cwd}]'
                )
                obs.content += f'\n[Jupyter Python interpreter: {_jupyter_plugin.python_interpreter_path}]'
            return obs
        else:
            # 未加载 Jupyter 插件时抛出异常
            raise RuntimeError(
                'JupyterRequirement not found. Unable to run IPython action.'
            )
```

## 0x05 环境

我们以 BrowserEnv 为例来看看环境如何实现。

`BrowserEnv` 是 OpenHands 框架中浏览器环境的核心封装类，负责创建独立的浏览器进程（基于 Playwright + BrowserGym），提供浏览器操作的标准化接口（如执行动作、检查存活状态、关闭环境），支持普通网页交互与评估场景（如 WebArena、MiniWoB），并通过进程间通信（Pipe）实现与主程序的高效数据传输，是框架中处理网页交互、数据抓取等浏览器相关任务的核心组件。

### 5.1 调用

ActionExecutor 有成员变量 self.browser: BrowserEnv

```
class ActionExecutor:
    """ActionExecutor is running inside docker sandbox.
    It is responsible for executing actions received from OpenHands backend and producing observations.
    """

    def __init__(
        self,
        plugins_to_load: list[Plugin],
        work_dir: str,
        username: str,
        user_id: int,
        enable_browser: bool,
        browsergym_eval_env: str | None,
    ) -> None:

        self.browser: BrowserEnv | None = None
            
    async def _init_browser_async(self):
        """Initialize the browser asynchronously."""
        if not self.enable_browser:
            logger.info('Browser environment is not enabled in config')
            return

        if sys.platform == 'win32':
            logger.warning('Browser environment not supported on windows')
            return

        logger.debug('Initializing browser asynchronously')
        try:
            self.browser = BrowserEnv(self.browsergym_eval_env)
            logger.debug('Browser initialized asynchronously')
        except Exception as e:
            logger.exception(f'Failed to initialize browser: {e}')
            self.browser = None
```

使用如下代码对浏览器环境进行使用。

```
    async def browse(self, action: BrowseURLAction) -> Observation:
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported or disabled.'
            )
        await self._ensure_browser_ready()
        return await browse(action, self.browser, self.initial_cwd)
        
    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        if self.browser is None:
            return ErrorObservation(
                'Browser functionality is not supported or disabled.'
            )
        await self._ensure_browser_ready()
        browser_observation = await browse(action, self.browser, self.initial_cwd)
        if not browser_observation.error:
            return browser_observation
        else:
            curr_files = os.listdir(self.downloads_directory)
            new_download = False
            for file in curr_files:
                if file not in self.downloaded_files:
                    new_download = True
                    self.downloaded_files.append(file)
                    break  

            if not new_download:
                return browser_observation
            else:
                # A new file is downloaded in self.downloads_directory, shift file to /workspace
                src_path = os.path.join(
                    self.downloads_directory, self.downloaded_files[-1]
                )
                # Guess extension of file using puremagic and add it to tgt_path file name
                file_ext = ''
                try:
                    guesses = puremagic.magic_file(src_path)
                    if len(guesses) > 0:
                        ext = guesses[0].extension.strip()
                        if len(ext) > 0:
                            file_ext = ext
                except Exception as _:
                    pass

                tgt_path = os.path.join(
                    '/workspace', f'file_{len(self.downloaded_files)}{file_ext}'
                )
                shutil.copy(src_path, tgt_path)
                file_download_obs = FileDownloadObservation(
                    content=f'Execution of the previous action {action.browser_actions} resulted in a file download. The downloaded file is saved at location: {tgt_path}',
                    file_path=tgt_path,
                )
                return file_download_obs
```

### 5.2 核心特色

1. **双模式支持**：兼容普通开放式浏览器场景（空白页启动，支持自由网页交互）与评估场景（集成 BrowserGym 生态，支持 WebArena、MiniWoB 等标准化任务评估）。
2. **进程隔离设计**：通过多进程（`multiprocessing`）创建独立浏览器进程，避免与主程序相互影响，提升稳定性与安全性。
3. **鲁棒的初始化与销毁**：内置重试机制（最多 5 次）处理浏览器启动失败，通过 `atexit` 注册自动关闭函数，确保资源释放。
4. **结构化数据处理**：将浏览器 DOM 转换为文本、截图转为 Base64 编码，保证观察结果（Observation）的可序列化与可传输性。
5. **下载与缓存支持**：配置浏览器下载路径，支持文件下载功能，适配网页数据抓取等场景。
6. **评估专用功能**：评估模式下自动注册 BrowserGym 任务、记录目标（Goal）与奖励（Reward），方便后续评估分析。

### 5.3 流程图

![11-5](images/img_005.png)

### 5.4 代码

初始化BrowserEnv时会`spawn`一个进程来启动浏览器，以便大模型在需要外部资料时通过工具获取，用到了强化学习库`gymnasium`来操作浏览器。

```
import gymnasium as gym

class BrowserEnv:
    def __init__(self, browsergym_eval_env: str | None = None):
        """初始化浏览器环境，创建独立进程与通信管道，支持普通交互与评估模式。

        参数:
            browsergym_eval_env: BrowserGym 评估环境名称（如 "browsergym/webarena"），
                                传入则启用评估模式，否则为普通开放式浏览器环境。
        """
        # 初始化HTML转文本转换器（用于提取网页文本内容）
        self.html_text_converter = self.get_html_text_converter()
        # 评估模式开关（是否启用BrowserGym评估）
        self.eval_mode = False
        # 评估目录（用于存储评估相关数据）
        self.eval_dir = ''

        # 评估模式配置：必须传入评估环境名称
        self.browsergym_eval_env = browsergym_eval_env
        self.eval_mode = bool(browsergym_eval_env)  # 有评估环境则启用评估模式

        # 初始化浏览器环境进程：设置多进程启动方式为"spawn"（跨平台兼容）
        multiprocessing.set_start_method('spawn', force=True)
        # 创建进程间通信管道（双向通信：浏览器端 <-> 代理端）
        self.browser_side, self.agent_side = multiprocessing.Pipe()

        # 启动浏览器环境（带重试机制）
        self.init_browser()
        # 注册进程退出时的自动关闭函数（确保资源释放）
        atexit.register(self.close)

    def get_html_text_converter(self) -> html2text.HTML2Text:
        """创建并配置HTML转文本转换器，定义网页内容的处理规则。"""
        html_text_converter = html2text.HTML2Text()
        # 配置规则：不忽略链接（保留链接文本），忽略图片
        html_text_converter.ignore_links = False
        html_text_converter.ignore_images = True
        # 图片使用alt文本替代（提升文本可读性）
        html_text_converter.images_to_alt = True
        # 禁用自动文本换行（保持原始网页文本结构）
        html_text_converter.body_width = 0
        return html_text_converter

    @retry(
        wait=wait_fixed(1),  # 重试间隔1秒
        stop=stop_after_attempt(5) | stop_if_should_exit(),  # 最多重试5次或进程退出时停止
        retry=retry_if_exception_type(BrowserInitException),  # 仅对浏览器初始化异常重试
    )
    def init_browser(self) -> None:
        """启动浏览器进程，失败则重试（最多5次），最终失败抛出异常。"""
        logger.debug('Starting browser env...')
        try:
            # 创建浏览器进程，目标函数为browser_process（独立进程中运行）
            self.process = multiprocessing.Process(target=self.browser_process)
            self.process.start()
        except Exception as e:
            logger.error(f'Failed to start browser process: {e}')
            raise  # 抛出异常触发重试

        # 检查浏览器进程是否存活（超时200秒）
        if not self.check_alive(timeout=200):
            self.close()  # 进程未存活则关闭资源
            raise BrowserInitException('Failed to start browser environment.')

    def browser_process(self) -> None:
        """浏览器进程核心逻辑：初始化BrowserGym环境，处理动作请求，返回观察结果。"""
        if self.eval_mode:
            # 评估模式：初始化BrowserGym评估环境
            assert self.browsergym_eval_env is not None
            logger.info('Initializing browser env for web browsing evaluation.')
            # 补全评估环境名称前缀（确保符合BrowserGym规范）
            if not self.browsergym_eval_env.startswith('browsergym/'):
                self.browsergym_eval_env = 'browsergym/' + self.browsergym_eval_env
            # 根据评估环境类型导入对应的BrowserGym任务（注册为gym环境）
            if 'visualwebarena' in self.browsergym_eval_env:
                import browsergym.visualwebarena  # noqa F401 注册visualwebarena任务
                import nltk
                nltk.download('punkt_tab')  # 下载NLTK依赖数据
            elif 'webarena' in self.browsergym_eval_env:
                import browsergym.webarena  # noqa F401 注册webarena任务
            elif 'miniwob' in self.browsergym_eval_env:
                import browsergym.miniwob  # noqa F401 注册miniwob任务
            else:
                raise ValueError(
                    f'Unsupported browsergym eval env: {self.browsergym_eval_env}'
                )
            # 创建评估环境（标记所有元素，超时100000秒）
            env = gym.make(self.browsergym_eval_env, tags_to_mark='all', timeout=100000)
        else:
            # 普通模式：创建开放式浏览器环境
            env = gym.make(
                'browsergym/openended',  # 开放式任务类型
                task_kwargs={'start_url': 'about:blank', 'goal': 'PLACEHOLDER_GOAL'},  # 空白页启动
                wait_for_user_message=False,  # 不等待用户消息
                headless=True,  # 无头模式（无GUI界面，节省资源）
                disable_env_checker=True,  # 禁用环境检查（提升启动速度）
                tags_to_mark='all',  # 标记所有DOM元素（便于交互）
                timeout=100000,  # 超时时间
                pw_context_kwargs={'accept_downloads': True},  # 允许文件下载
                pw_chromium_kwargs={'downloads_path': '/workspace/.downloads/'},  # 下载文件保存路径
            )
        # 重置环境，获取初始观察结果与信息
        obs, info = env.reset()

        logger.info('Successfully called env.reset')
        # 评估模式专用：记录目标与图片URL（用于后续评估）
        self.eval_goal = None
        self.goal_image_urls = []
        self.eval_rewards: list[float] = []
        if self.eval_mode:
            self.eval_goal = obs['goal']
            # 处理目标对象中的文本与图片URL
            if 'goal_object' in obs:
                obs['goal_object'] = list(obs['goal_object'])
                if len(obs['goal_object']) > 0:
                    self.eval_goal = obs['goal_object'][0]['text']  # 取第一个目标文本
                # 收集目标中的图片URL
                for message in obs['goal_object']:
                    if message['type'] == 'image_url':
                        image_src = message['image_url']
                        if isinstance(image_src, dict):
                            image_src = image_src['url']  # 处理嵌套URL格式
                        self.goal_image_urls.append(image_src)
            logger.debug(f'Browsing goal: {self.eval_goal}')
        logger.info('Browser env started.')

        # 循环处理请求（进程退出时终止）
        while should_continue():
            try:
                # 检查是否有来自代理端的请求（超时0.01秒，非阻塞）
                if self.browser_side.poll(timeout=0.01):
                    unique_request_id, action_data = self.browser_side.recv()

                    # 处理关闭请求：关闭环境并退出进程
                    if unique_request_id == 'SHUTDOWN':
                        logger.debug('SHUTDOWN recv, shutting down browser env...')
                        env.close()
                        return
                    # 处理存活检查请求：返回ALIVE状态
                    elif unique_request_id == 'IS_ALIVE':
                        self.browser_side.send(('ALIVE', None))
                        continue

                    # 评估模式专用请求：获取目标信息
                    if action_data['action'] == BROWSER_EVAL_GET_GOAL_ACTION:
                        self.browser_side.send(
                            (
                                unique_request_id,
                                {
                                    'text_content': self.eval_goal,  # 目标文本
                                    'image_content': self.goal_image_urls,  # 目标图片URL列表
                                },
                            )
                        )
                        continue
                    # 评估模式专用请求：获取奖励列表
                    elif action_data['action'] == BROWSER_EVAL_GET_REWARDS_ACTION:
                        self.browser_side.send(
                            (
                                unique_request_id,
                                {'text_content': json.dumps(self.eval_rewards)},  # 奖励列表JSON字符串
                            )
                        )
                        continue

                    # 处理普通浏览器动作请求
                    action = action_data['action']
                    # 执行动作，获取结果（观察值、奖励、终止状态等）
                    obs, reward, terminated, truncated, info = env.step(action)

                    # 评估模式：记录奖励
                    if self.eval_mode:
                        self.eval_rewards.append(reward)

                    # 处理网页文本内容：DOM对象转字符串后转为纯文本
                    html_str = flatten_dom_to_str(obs['dom_object'])
                    obs['text_content'] = self.html_text_converter.handle(html_str)
                    # 处理观察结果序列化（确保可通过管道传输）
                    # 1. 标记元素截图转为Base64 URL
                    obs['set_of_marks'] = image_to_png_base64_url(
                        overlay_som(
                            obs['screenshot'], obs.get('extra_element_properties', {})
                        ),
                        add_data_prefix=True,
                    )
                    # 2. 网页截图转为Base64 URL
                    obs['screenshot'] = image_to_png_base64_url(
                        obs['screenshot'], add_data_prefix=True
                    )
                    # 3. 转换为Python原生类型（numpy数组转普通整数）
                    obs['active_page_index'] = obs['active_page_index'].item()
                    obs['elapsed_time'] = obs['elapsed_time'].item()
                    # 将结果发送回代理端
                    self.browser_side.send((unique_request_id, obs))
            except KeyboardInterrupt:
                logger.debug('Browser env process interrupted by user.')
                try:
                    env.close()
                except Exception:
                    pass
                return

    def step(self, action_str: str, timeout: float = 120) -> dict:
        """在浏览器环境中执行动作，返回序列化的观察结果。

        参数:
            action_str: 待执行的浏览器动作（如点击、输入、导航等）
            timeout: 超时时间（默认120秒）

        返回:
            dict: 浏览器观察结果（含网页文本、截图、元素信息等）
        """
        # 生成唯一请求ID（用于匹配响应）
        unique_request_id = str(uuid.uuid4())
        # 发送动作请求到浏览器进程
        self.agent_side.send((unique_request_id, {'action': action_str}))
        start_time = time.time()
        # 循环等待响应
        while True:
            # 检查进程是否退出或超时
            if should_exit() or time.time() - start_time > timeout:
                raise TimeoutError('Browser environment took too long to respond.')
            #            # 检查是否有响应（非阻塞，超时0.01秒）
            if self.agent_side.poll(timeout=0.01):
                response_id, obs = self.agent_side.recv()
                # 匹配请求ID，返回对应的观察结果
                if response_id == unique_request_id:
                    return dict(obs)

    def check_alive(self, timeout: float = 60) -> bool:
        """检查浏览器进程是否存活。

        参数:
            timeout: 超时时间（默认60秒）

        返回:
            bool: 存活返回True，否则返回False
        """
        # 发送存活检查请求
        self.agent_side.send(('IS_ALIVE', None))
        # 等待响应（超时时间内）
        if self.agent_side.poll(timeout=timeout):
            response_id, _ = self.agent_side.recv()
            # 响应ID为"ALIVE"表示进程存活
            if response_id == 'ALIVE':
                return True
            logger.debug(f'Browser env is not alive. Response ID: {response_id}')
        return False

    def close(self) -> None:
        """关闭浏览器环境，释放进程与通信资源。"""
        # 若进程已终止，直接返回
        if not hasattr(self, 'process') or not self.process.is_alive():
            return
        try:
            # 发送关闭请求到浏览器进程
            self.agent_side.send(('SHUTDOWN', None))
            # 等待进程终止（最多5秒）
            self.process.join(5)
            # 若进程仍存活，强制终止
            if self.process.is_alive():
                logger.error(
                    'Browser process did not terminate, forcefully terminating...'
                )
                self.process.terminate()
                self.process.join(5)
                # 若仍存活，强制杀死进程
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(5)
            # 关闭通信管道
            self.agent_side.close()
            self.browser_side.close()
        except Exception as e:
            logger.error(f'Encountered an error when closing browser env: {e}')
```

## 0xFF 参考

<https://docs.all-hands.dev/openhands/usage/architecture/runtime>
