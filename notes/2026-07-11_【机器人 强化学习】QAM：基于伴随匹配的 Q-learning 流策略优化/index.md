# 【机器人 / 强化学习】QAM：基于伴随匹配的 Q-learning 流策略优化

# 【机器人 / 强化学习】QAM：基于伴随匹配的 Q-learning 流策略优化

目录

* [【机器人 / 强化学习】QAM：基于伴随匹配的 Q-learning 流策略优化](#机器人--强化学习qam基于伴随匹配的-q-learning-流策略优化)
  + [0x00 概要](#0x00-概要)
  + [1.1 核心问题：Flow 策略的优化困境](#11-核心问题flow-策略的优化困境)
    - [1.1 为什么 Flow 策略难以用 RL 优化？](#11-为什么-flow-策略难以用-rl-优化)
    - [1.2 三种候选方案](#12-三种候选方案)
  + [0x02 核心思想：Adjoint Matching](#0x02-核心思想adjoint-matching)
    - [2.1 一句话理解](#21-一句话理解)
    - [2.2 伴随变量的直觉](#22-伴随变量的直觉)
    - [2.3 一个比喻](#23-一个比喻)
    - [2.4 数学框架](#24-数学框架)
  + [0x03 网络架构](#0x03-网络架构)
    - [3.1 三大核心网络](#31-三大核心网络)
      * [Velocity Network（\(v\_\theta\)）：导航员](#velocity-network导航员)
      * [Critic（Q）：价值网络](#criticq价值网络)
      * [Target Q 网络](#target-q-网络)
    - [3.2 Slow / Fast 双 Actor 设计](#32-slow--fast-双-actor-设计)
    - [3.3 QAM 的伴随匹配损失](#33-qam-的伴随匹配损失)
  + [0x04 QAM 作者实现](#0x04-qam-作者实现)
    - [4.1 训练主循环](#41-训练主循环)
    - [4.2 Critic Loss 流程](#42-critic-loss-流程)
    - [4.3 Actor Loss 流程](#43-actor-loss-流程)
    - [4.4 Flow Matching 部分（Slow Actor）](#44-flow-matching-部分slow-actor)
    - [4.5 Adjoint Matching 部分（Fast Actor）](#45-adjoint-matching-部分fast-actor)
    - [4.6 动作采样流程](#46-动作采样流程)
    - [4.7 Flow 动作生成](#47-flow-动作生成)
    - [4.8 六大核心特性](#48-六大核心特性)
      * [Slow/Fast 双 Actor 结构](#slowfast-双-actor-结构)
      * [Adjoint Matching 注入 Critic 梯度](#adjoint-matching-注入-critic-梯度)
      * [Pessimistic Critic Ensemble](#pessimistic-critic-ensemble)
      * [Best-of-N 动作选择](#best-of-n-动作选择)
      * [可插拔的 FQL One-Step Policy](#可插拔的-fql-one-step-policy)
      * [可选 Edit Policy Refinement](#可选-edit-policy-refinement)
  + [0x05 QAM 与相关方法的对比](#0x05-qam-与相关方法的对比)
    - [5.1 QAM vs IQL](#51-qam-vs-iql)
    - [5.2 QAM vs RECAP](#52-qam-vs-recap)
    - [5.3 QAM vs QGF](#53-qam-vs-qgf)
    - [5.4 通俗版理解](#54-通俗版理解)
  + [0x06 总结](#0x06-总结)

## 0x00 概要

有了 DIVL 提供的价值函数，下一步是策略提升——如何让 VLA 策略输出更高价值的动作？这就是 QAM（Q-learning with Adjoint Matching）要解决的问题。它巧妙地将 Flow Matching 的表达能力与 Q-learning 的优化目标结合，同时解决了扩散模型在强化学习中通过时间反向传播（BPTT）带来的不稳定问题。

## 1.1 核心问题：Flow 策略的优化困境

### 1.1 为什么 Flow 策略难以用 RL 优化？

现代视觉-语言-动作（VLA）模型常采用 Flow Matching 或 Diffusion 架构生成动作。这类模型表达能力强，可以生成平滑、多峰、符合物理规律的连续动作序列。但问题在于：**动作并非单步直接输出，而是通过多步去噪/ODE 积分逐步迭代生成。**

这意味着整个计算图是一条长链式结构——每一步计算都依赖上一步的结果。如果我们想用 Q 函数梯度来优化最终生成的动作，梯度就必须沿着整条生成链路反向传播。由此引发两大问题：

1. **计算开销大**：多步迭代链路需要全程保留所有中间状态，显存占用极高
2. **数值不稳定**：长链路反向传播极易出现梯度消失、梯度爆炸、更新方差过大，最终导致训练崩溃

这是基于扩散/流匹配的强化学习策略在真机场景中难以收敛的核心原因。

### 1.2 三种候选方案

面对"如何优化 flow policy"这个问题，有三种可能的路径：

**方案 A：优势加权回归（AWR）。** 需要计算 \(\log \pi(a \mid s)\)，但 flow 多步去噪过程下难以获得解析密度，高维 action chunk 下归一化常数不可积。

**方案 B：直接反向传播 \(\nabla\_\theta Q(s, x\_1(x\_0;\theta))\)。** 需要穿过整条 ODE 反传，计算昂贵且数值不稳定，对大型 VLA 模型不实际。

**方案 C：QAM（采用此方案）。** 将轨迹级优化转化为流路径上的局部回归，无需穿过 ODE 反传，只需终点梯度。这正是我们要深入讲解的方案。

## 0x02 核心思想：Adjoint Matching

### 2.1 一句话理解

**不让梯度"穿越"整个生成过程（太贵太不稳定），而是在终点拿到 Critic 的"修改意见"后，沿着生成路径反向传递这份意见，逐步微调每一步的生成方向。**

传统方式与 QAM 的对比：

```
传统做法（BPTT，不可行）:
噪声 → [ODE 积分 N 步] → 动作 → ∇_a Q → [反向穿过 N 步 ODE] → 更新 θ

QAM 做法（高效稳定）:
1. 用冻结的参考策略 π_β 跑一条前向流轨迹 {a_w}（无需梯度）
2. 在终点 a_1 处计算 Critic 梯度: g̃_1 = -∇_a[Q(s, a_1)/λ]
3. 沿参考轨迹反向积分伴随方程，得到每个时间点 w 的局部目标 g̃_w
4. 用 ||f_θ(s,a_w,w) - f_β(s,a_w,w) + σ_w·g̃_w||² 做局部回归更新 θ
```

### 2.2 伴随变量的直觉

在控制理论中，伴随变量（Adjoint）可以理解为**终点目标函数对轨迹中间状态的敏感度**。

在 QAM 框架下，优化目标是提升动作对应的 Q 价值。终点动作处的 Critic 梯度 \(\nabla\_a Q(s, a\_{\text{final}})\) 的含义是：对最终动作做微小偏移时，Q 价值的变化趋势。但最终动作由完整的流生成轨迹推导而来，我们需要求解一个关键问题：

> 为了让最终动作向高 Q 值方向优化，流生成过程中每一个时间步的向量场需要做出怎样的调整？

伴随法正是用来求解该问题的工具。它会将终点处的 Q 梯度，按照伴随方程反向映射到流轨迹的每一个时间步，生成逐步的局部监督信号。至此，原本复杂的端到端 RL 优化问题，被转化为简单的**逐歩监督回归任务**：在每个时间步，让当前向量场去匹配经过 Q 梯度修正后的目标向量场。

### 2.3 一个比喻

我们可以用流水线上的工头来理解 QAM：

* **直接端到端反传**就像拉一根很长的绳子，希望绳子末端的反馈能准确传到每一个中间节点。绳子越长，抖动越大，越难控制。
* **QAM**则像是在流水线的每个工位安排一个工头。每个工头只告诉当前这一步的向量场应该往哪里微调，不需要整条链路一起承担不稳定的反向传播。

### 2.4 数学框架

QAM 的目标是让策略 \(\pi\) 逐渐靠近一个更好的分布：

\[\pi(a \mid s) \propto \pi\_{\text{behavior}}(a \mid s) \cdot \exp(Q(s, a))
\]

为了实现这个目标，QAM 定义的目标速度场是：

\[v\_{\text{target}} = (x\_1 - x\_0)\_{\text{保持现状（模仿行为）}} + \alpha \cdot \nabla\_a Q(s, x\_t)\_{\text{寻找更好（Q 的指引）}}
\]

这里的 \(\alpha\) 控制了"模仿行为"和"追求 Q"之间的平衡。

QAM 没有把 RL 梯度硬塞进整个生成链条，而是用伴随匹配把全局优化问题拆成 flow 每一步的局部回归问题。论文中特别说明：\(f\_\beta\) 保持固定，作为离线 RL 前由行为克隆初始化得到的参考流；离线和在线训练过程中优化的是 \(f\_\theta\)。

关键注意点：QAM 不是部署推理时"每一帧实时调用 critic 梯度"。论文强调的是**训练阶段**：把 critic gradient 转换为 flow 模型的 step-wise supervision。训练后的策略在部署时仍然是前向生成动作，只是其动作分布已经被价值函数引导过。

## 0x03 网络架构

### 3.1 三大核心网络

#### Velocity Network（\(v\_\theta\)）：导航员

这是 QAM 的执行灵魂。在传统 RL（如 PPO/SAC）中，策略网络是一个"瞬移器"——输入状态 \(s\)，直接跳出动作 \(a\)。但在 QAM 这种基于 Flow Matching 的算法中，策略网络是一个"导航员"：

* **输入**：\((s, x\_t, t)\)——状态、当前位置、当前时间刻度
* **输出**：速度矢量，表示当前位置的切线方向
* **作用**：通过 Adjoint Matching 学习如何在 Q 的指引下生成最优的速度场

在数学上，动作的生成遵循一个常微分方程：

\[\frac{dx\_t}{dt} = v\_\theta(x\_t, t, s)
\]

输入 \((s, x\_t, t)\)：需要知道现在的环境 \(s\)、自己现在在哪 \(x\_t\)、以及现在是第几秒 \(t\)。输出 \(v\)：返回一个和动作维度一样的矢量，代表当前位置的切线方向。

在推理时，Velocity Network 被反复调用：

```
x = torch.randn_like(action)   # 从噪声开始
dt = 1.0 / num_steps
for i in range(num_steps):
    t = i * dt
    v = v_theta(x, t, s)       # 问导航员
    x = x + v * dt             # 迈一小步
# 循环结束后的 x 就是最终动作 a
```

#### Critic（Q）：价值网络

Q 网络负责评估动作的好坏。输入 \((s, a)\)，输出 Q 值。它的作用是为 Adjoint Matching 提供梯度信息 \(\nabla\_a Q\)。

#### Target Q 网络

Target Q 是 Q 的一个慢更新副本（软更新/Polyak Averaging）。为什么需要它？如果在计算更新目标时使用正在频繁变动的 Q，会导致学习目标不断"逃跑"，引发模型震荡甚至发散。使用平滑变化的 Target 网络能极大提高训练稳定性。

### 3.2 Slow / Fast 双 Actor 设计

QAM 代码中最有特色的设计是**双流 Actor 架构**：

| Actor | 训练目标 | Loss 来源 | 作用 | 特点 |
| --- | --- | --- | --- | --- |
| **actor\_slow**（慢流） | Behavior Cloning | flow\_loss 匹配 \(x\_1 - x\_0\) | 学习数据集的先验，保证轨迹大致形状 | 稳定、保守 |
| **actor\_fast**（快流） | RL 策略提升 | adj\_loss（Q 梯度） | 在 actor\_slow 的基础上给出修正矢量 | 激进、追求卓越 |

这种"双流"设计本质是一种残差设计，解决了强化学习中一个经典的矛盾：**如何既保持数据中的稳定行为，又追求更高的奖励？**

两者的结合方式由一个 `residual` 参数控制：

* **Residual 模式（推荐）**：生成的总速度 \(v = v\_{\text{slow}} + v\_{\text{fast}}\)。就像老司机在开车，旁边坐着一个精算的导航员，不断在老司机的方向盘上施加微小的力，让车开得更完美。
* **Non-residual 模式**：actor\_fast 尝试直接去匹配一个已经融合了 Q 梯度的总目标速度。

### 3.3 QAM 的伴随匹配损失

下面这段代码是 QAM 损失函数的核心实现，Adjoint Matching 对应的就是将 Q 梯度直接修正目标速度向量的这一步：

```
def qam_loss(v_net, q_net, s, x_1, eta=1.0):
    batch_size = s.shape[0]

    # 1. 采样 t 和噪声 x_0
    t = torch.rand(batch_size, device=s.device)   # t ~ U(0, 1)
    x_0 = torch.randn_like(x_1)                   # x_0 ~ N(0, I)

    # 2. 构造中间点 x_t（线性插值）
    t_expanded = t.view(-1, 1)
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

    # 3. 计算 Q 对 x_t 的梯度（Adjoint 项）
    x_t.requires_grad_(True)
    q_val = q_net(s, x_t)
    grad_a_q = torch.autograd.grad(
        outputs=q_val.sum(), inputs=x_t, create_graph=True
    )[0]

    # 4. 构造目标速度（核心：匹配）
    # 基础速度 + Q 梯度修正 = 目标速度
    v_base = x_1 - x_0
    v_target = v_base + eta * grad_a_q

    # 5. 预测速度并计算 MSE Loss
    v_pred = v_net(s, x_t, t)
    loss = F.mse_loss(v_pred, v_target.detach())

    return loss
```

这段代码的精髓在于：

* **"Adjoint"（伴随）**：对应 `grad_a_q`。没有让梯度通过 ODE 链条反传，而是把 Q 的梯度作为一个伴随信号直接提取出来。
* **"Matching"（匹配）**：对应 `F.mse_loss(v_pred, v_target)`。Velocity 网络不再是去最大化 Q，而是去匹配那个已经被 Q 梯度修正过的理想速度 `v_target`。

## 0x04 QAM 作者实现

qam.py 实现了一个 **Q-learning + flow policy + adjoint matching** 的强化学习 agent：它用 `actor_slow` 学习数据动作分布，用 `critic` 学习价值函数，用 `actor_fast` 通过 critic 梯度进行 Q 引导的 flow 修正，并支持 best-of-n、action chunking、one-step 蒸馏和 edit policy 等增强机制。

总体流程图如下：

![QAM-1-总体流程图](images/img_001.png)

### 4.1 训练主循环

训练入口是 `update()` 和 `batch_update()`：

* 单步更新 和 批量 scan 更新 最终都会调用 `_update()`：

`total_loss()` 把 critic loss 和 actor loss 相加：

```
loss = critic_loss + actor_loss
```

训练时，critic loss 和 actor loss 协同工作，就像"老师"和"学生"：

![QAM-训练主循环](images/img_002.png)

**Critic Loss** (老师在备课)：

* 训练 Q 函数，使当前 Q 接近 TD target。使用保守 Q 学习策略——\(Q\_{\text{target}} = \text{mean}(Q) - \rho \cdot \text{std}(Q)\)——缓解 OOD 动作带来的 Q 过估计。
* 它唯一的任务是把 \(Q(s, a)\) 的数值算准。

**Actor Loss** (学生在考试)：

* 它调用已经练好的 critic 网络。核心行: `grad_fn = jax.grad(...)`。学生通过对老师 (Critic) 求导, 拿到梯度 adj (伴随状态)。然后, 学生 (actor\_fast) 去匹配这个梯度方向。
* 包含 flow matching loss（slow actor 的基础模仿学习）和 adjoint matching loss（fast actor 的 Q 梯度引导优化），以及可选的 one-step actor loss 和 edit policy loss。

### 4.2 Critic Loss 流程

`critic_loss()` 的目标是训练 Q 函数，使当前 Q 接近 TD target。

![QAM-critic_loss](images/img_003.png)

流程说明如下：

| 步骤 | 操作 | 说明 |
| --- | --- | --- |
| A | batch 输入 | 批量输入数据 |
| B | action\_chunking? | 判断是否需要动作分块 |
| C | reshape actions 为动作块 | 将动作重塑为块结构 |
| D | 取 actions[..., 0, :] | 取第一个动作 |
| E | sample\_actions 采样 next\_actions | 采样下一步动作 |
| F | clip next\_actions 到 [-1, 1] | 裁剪动作值范围 |
| G | target\_critic 计算 next\_qs | 目标网络计算 Q 值 |
| H | next\_q = mean - ρ \* std | 保守 Q 值估计 (CQL/类似方法) |
| I | 计算 target\_q | TD 目标：reward + γ^H × mask × next\_q |
| J | critic 计算当前 q | 当前网络计算 Q 值 |
| K | MSE(q, target\_q) × valid | 均方误差损失，乘以有效掩码 |
| L | 返回 critic\_loss 和统计信息 | 输出损失和诊断信息 |

### 4.3 Actor Loss 流程

`actor_loss()` 包含几部分：

1. slow actor 的 behavior cloning flow matching loss。
2. fast actor 的 adjoint matching loss。
3. 可选的 FQL one-step actor loss。
4. 可选的 edit policy loss。

整体结构：

![QAM-actor_loss](images/img_004.png)

### 4.4 Flow Matching 部分（Slow Actor）

这部分是 slow actor 的基础模仿学习目标，它做的事情是：

1. 从标准高斯采样 `x_0`。
2. 取数据集动作作为 `x_1`。
3. 随机采样时间 `t`。
4. 在线性插值路径上得到 `x_t`。
5. 真实速度是 `x_1 - x_0`。
6. 训练 `actor_slow(obs, x_t, t)` 预测这个速度。

核心公式如下：

```
x_0 ~ N(0, I)                   # 采样噪声
x_1 = dataset_action            # 数据集动作
t ~ Uniform(0, 1)               # 采样时间
x_t = (1 - t) * x_0 + t * x_1   # 线性插值
vel = x_1 - x_0                 # 真实速度
pred = actor_slow(obs, x_t, t)  # 预测速度
flow_loss = MSE(pred, vel)      # 回归
```

这个部分让 `actor_slow` 学会从噪声流向数据动作分布。

### 4.5 Adjoint Matching 部分（Fast Actor）

`adj_matching()` 是代码中最核心的函数，它利用 critic 对动作的梯度构造 adjoint state，再反向传播到 flow 轨迹中，用来训练 `actor_fast`。

**正向 Flow Rollout**：从噪声动作开始，按 `flow_steps` 做积分。

**Critic 梯度初始化 Adjoint**：完成 flow rollout 后，用 critic 对最终动作求梯度：

```
grad_fn = jax.grad(...)
adj = -grad_fn(obs, xs[-1]) * self.config["inv_temp"]
```

**反向传播 Adjoint**：使用 VJP（Vector-Jacobian Product）从最后一步向前反推 adjoint：

```
for i in reversed(range(flow_steps)):
    def fn(xi):
        return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)
    vjp = jax.vjp(fn, xs[i])[1](adj)[0]
    adj = adj + h * vjp
```

这确保了 Q 的指引信号能够穿过整个 Flow 动力学系统，精准地分配到每一个时间步上。

### 4.6 动作采样流程

推理或构造 target 时会调用 `sample_actions()`

![QAM-4-sample](images/img_005.png)

`sample_actions()` 会一次生成 `best_of_n` 个候选动作，然后用 critic 选 Q 最大的那个：

```
q = self.network.select("critic")(observations, actions).mean(axis=0)
indices = jnp.argmax(q, axis=-1)
...
actions = ...
```

这是一个简单但有效的 inference-time policy improvement。

### 4.7 Flow 动作生成

`compute_flow_actions()` 从噪声开始，按 `flow_steps` 做 Euler 积分：

```
actions = noises
for i in range(flow_steps):
    t = ...
    vels = sum([network(observations, actions, t) for network in networks])
    actions = actions + vels / flow_steps
actions = clip(actions)
```

这里的 `model` 参数可以是：

* `"slow"`：只用 `actor_slow`
* `"fast"`：只用 `actor_fast`
* `"slow,fast"`：两个速度场相加

### 4.8 六大核心特性

QAM 的六大核心特性如下：

#### Slow/Fast 双 Actor 结构

如前所述，actor\_slow 负责模仿，actor\_fast 负责优化。这种分离设计将"保持数据分布"和"追求更高奖励"这两个目标解耦，结构清晰。

#### Adjoint Matching 注入 Critic 梯度

不是简单地最大化 \(Q(s, \text{actor}(s))\)，而是先通过 flow 生成动作轨迹，在终点计算 critic 梯度，沿 flow 时间反向传播 adjoint，最后训练 fast vector field 匹配 adjoint 信号。这比普通 deterministic policy gradient 更适合 flow-based policy。

#### Pessimistic Critic Ensemble

Critic 使用 ensemble 并采用 \(\text{mean}(Q) - \rho \cdot \text{std}(Q)\) 作为 target Q。这是一种保守 Q 学习策略，适合离线 RL，能缓解 OOD 动作带来的 Q 过估计。

#### Best-of-N 动作选择

采样时先生成多个候选动作，再用 critic 选 Q 最高的那个。这相当于在推理阶段做轻量级 policy improvement。

#### 可插拔的 FQL One-Step Policy

当 `fql_alpha > 0` 时，模型训练 `one_step_actor`，把 flow policy 蒸馏成一步策略。优点：推理更快、保留 flow policy 的动作质量、额外通过 Q loss 提高动作价值。

#### 可选 Edit Policy Refinement

当 `edit_scale > 0` 时，模型会在 flow action 基础上学习一个小修正：`edited_action = flow_action + edit * edit_scale`。这让策略可以在保持 flow policy 主体输出的同时，进行局部 Q 提升。

## 0x05 QAM 与相关方法的对比

### 5.1 QAM vs IQL

| 维度 | IQL | QAM |
| --- | --- | --- |
| 策略形式 | 单步高斯/确定性策略 | 多步 Flow/Diffusion 生成策略 |
| Q 引导方式 | Advantage 加权行为克隆 | Critic gradient + Adjoint Matching |
| 优化信号 | **标量权重**：这个动作好多少 | **矢量方向**：动作往哪里改更好 |
| 表达能力 | 相对有限 | 能表达多峰、连续、长程动作分布 |
| 计算开销 | 较低 | 较高，需要 flow 积分和伴随计算 |

通俗说：IQL 是"这个动作得分高，多模仿它"。QAM 是"当前动作如果往这个方向偏一点，分数会更高；请让生成过程每一步都朝这个方向调整"。这就是 QAM 比 advantage 加权 BC 更适合复杂生成式 VLA 策略的原因。

### 5.2 QAM vs RECAP

| 维度 | RECAP | QAM / LWD |
| --- | --- | --- |
| 改进信号 | 二值 improvement indicator | 连续 critic action gradient |
| 策略更新 | Advantage-conditioned imitation / MLE | Adjoint Matching 回归目标 |
| 优势 | 稳定、易训练 | 能利用连续方向信息，适合 flow policy |
| 风险 | 标签粗粒度、依赖阈值 | 依赖 critic 梯度质量、计算更复杂 |

RECAP 像老师说："这张卷子是优秀答案，请按这种风格写。"QAM 像老师直接握着学生的笔说："这一笔往左偏一点，分数会更高。"两者不是谁取代谁，而是不同工程取舍。

### 5.3 QAM vs QGF

QGF（Q-Guided Flow）是 Sergey Levine 团队的成果。两者的根本区别在于策略优化发生的时机：

| 维度 | QAM | QGF |
| --- | --- | --- |
| 策略优化时机 | 训练时 | 测试/推理时 |
| 是否更新策略参数 | 是，RL 训练中更新 actor | 否，测试时不再训练 |
| 对 Q 梯度的使用 | Adjoint Matching 构造稳定目标 | 采样时直接用 \(\nabla\_a Q\) 引导生成 |
| 训练复杂度 | 更高（actor+critic 联合优化） | 更低（actor 可用纯 BC 稳定训练） |
| 稳定性 | 解决"穿过多步去噪反传不稳定" | 规避训练期 actor-critic 不稳定 |
| 典型优势 | 理论上可学到更"内生"的高价值策略 | 简单、便宜、可扩展 |

**工程选择建议：**

1. 想要端到端 RL 学出最终策略、追求上限 → 偏 QAM
2. 想要训练稳定、实现简单、算力更友好 → 偏 QGF
3. 可组合使用：先用 QAM 训练出更好 critic/actor，再在推理期叠加 QGF 式 guidance

### 5.4 通俗版理解

> 风谷城每年都有一场比赛：谁能骑着"流风马"穿过迷雾峡谷，谁就能拿到全城的粮仓钥匙。城里两位年轻驯马师——阿井（QAM）和阿舟（QGF）——选择了不同的路径。

**阿井的做法（QAM）**：  
他每次训练都带着裁判老秦（Q 函数）。老秦不只在终点打分，还会在每个弯道说："往左一点会更值钱。"可流风马动作连着动作，直接反向传播经常算晕。于是阿井请来"回声匠"（Adjoint Matching），把老秦的方向建议翻译成一系列稳定的练习目标。他不是等到最后才改，而是每一步都学"该往哪边更好"。慢慢地，阿井和马在训练里就越骑越聪明。

**阿舟的做法（QGF）**：  
他先让马老老实实学名骑手的录像（BC 预训练），把基本功练稳。同时单独训练了一个"价值罗盘"（Critic）。到比赛那天，阿舟不再重新训练马，只在骑行当下拿出罗盘："往这边多拽一点，价值更高。"马边跑边被推着修正路线，省钱、省时。

**优劣**：在新赛道上，阿舟的方法因不用临时重练，快速适配。阿井在长期训练充分的赛道上，上限依然很高。

## 0x06 总结

QAM 是离线强化学习和生成式策略融合的典范之作。它的核心创新在于：

1. **避免 BPTT 的梯度问题**：通过 Adjoint Matching，将长链路的梯度反向传播转化为监督学习问题
2. **融合 Q 的指引**：在保持数据稳定性的同时，利用 Q 函数引导策略向更高奖励方向探索
3. **双流设计**：通过 actor\_slow（模仿）和 actor\_fast（优化）的平衡，既保持稳定性又追求卓越
4. **精细的数学设计**：通过伴随动力学，在轨迹的起止点强制模仿，中间阶段允许适度探索

QAM 证明了即使在离线设定下，我们也可以通过巧妙的统计方法将生成式模型的表达能力与强化学习的目标导向能力结合，为大规模 VLA 模型在真实机器人上的 RL 训练奠定了可行的工程基础。
