# 【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（3）算法篇（RLPD）

目录

* [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（3）算法篇（RLPD）](#机器人--强化学习serl让真机强化学习从难用走向可复现的强化学习框架-----3算法篇rlpd)
  + [0x00 概要](#0x00-概要)
  + [0x01 基础 & 背景](#0x01-基础--背景)
    - [1.1 总体流程图](#11-总体流程图)
    - [1.2 面临的问题](#12-面临的问题)
  + [0x02 RLPD 基础](#0x02-rlpd-基础)
    - [2.1 RLPD 的三大支柱](#21-rlpd-的三大支柱)
    - [2.2 Prior Data 策略](#22-prior-data-策略)
    - [2.3 算法流程图](#23-算法流程图)
    - [2.4 代码细节](#24-代码细节)
  + [0x03 BC](#0x03-bc)
    - [3.1 BCAgent](#31-bcagent)
      * [3.1.1 BCAgent 在SERL中的作用](#311-bcagent-在serl中的作用)
      * [3.1.2 BCAgent 核心组件](#312-bcagent-核心组件)
      * [3.1.3 BCAgent 编码器架构](#313-bcagent-编码器架构)
      * [3.1.4 BCAgent 损失函数](#314-bcagent-损失函数)
      * [3.1.5 BCAgent 动作采样](#315-bcagent-动作采样)
    - [3.2 bc.py 流程图](#32-bcpy-流程图)
    - [3.3 bc.py 技术细节](#33-bcpy-技术细节)
  + [0x04 High UTD & 稳定性机制](#0x04-high-utd--稳定性机制)
    - [4.1 High UTD：把每条真机样本反复研读](#41-high-utd把每条真机样本反复研读)
      * [4.1.1 理解 UTD（Update-to-Data Ratio）](#411-理解-utdupdate-to-data-ratio)
      * [4.1.2 为什么 SERL 需要高 UTD？](#412-为什么-serl-需要高-utd)
      * [4.1.3 极致的采样效率](#413-极致的采样效率)
      * [4.1.4 实现](#414-实现)
      * [4.1.5 UTD 降为 1 的后果：从"特训班"降级为"自习室"](#415-utd-降为-1-的后果从特训班降级为自习室)
      * [4.1.6 High UTD 的副作用与应对](#416-high-utd-的副作用与应对)
    - [4.2 稳定性机制](#42-稳定性机制)
  + [0x05 Layer Normalization](#0x05-layer-normalization)
    - [5.1 当前层归一化实现分析](#51-当前层归一化实现分析)
    - [5.2 层归一化的技术细节和优势](#52-层归一化的技术细节和优势)
    - [5.3 实现方案](#53-实现方案)
    - [5.4 总结](#54-总结)
  + [0x06 Soft Update & REDQ](#0x06-soft-update--redq)
    - [6.1 Soft Update 的力量](#61-soft-update-的力量)
    - [6.2 REDQ：Critic Ensemble 抑制过估计](#62-redqcritic-ensemble-抑制过估计)
      * [6.2.1 为什么需要 Critic Ensemble？](#621-为什么需要-critic-ensemble)
      * [6.2.2 REDQ论文算法](#622-redq论文算法)
      * [6.2.3 SERL 的实现](#623-serl-的实现)
        + [架构细节](#架构细节)
        + [具体实现](#具体实现)
        + [与论文的区别](#与论文的区别)
    - [6.3 时间戳对齐](#63-时间戳对齐)
  + [0x07 SAC vs RLPD](#0x07-sac-vs-rlpd)
    - [7.1 相同之处](#71-相同之处)
    - [7.2 不同之处](#72-不同之处)
    - [7.3 通俗解释](#73-通俗解释)
  + [0xFF 参考](#0xff-参考)

## 0x00 概要

SERL 的核心使命是：在真实世界中，让机器人在 20-40 分钟内学会高精度的机械操作。它通过集成 SAC、RLPD、DrQ 和 VICE，将原本需要数百万次尝试的 RL，压缩到了人类演示水平的量级。

RLPD（Reinforcement Learning with Prior Data）是一种基于 off-policy 的 actor-critic强化学习算法，借鉴了 soft-actor-critic 等时序差分算法的成功经验，但为满足上述需求做出了一些关键修改，它其实就是 SAC + 之前的数据（Prior Data）+ 极高的更新频率（High UTD）。

注：

* **本系列的最终目标是“通过一系列相关项目/算法的解读，来深入学习/分析/反推 LWD（Learning while Deploying）这篇论文的机理和可能实现”**。之所以从SERL入手，是因为 SERL，HIL-SERL，SOP（没有开源）都是罗剑岚博士的一系列论文，可以从中管窥作者的思路脉络。
* 本文依然是从工程/论文进行反推，还请读者不吝指出问题，多谢。

## 0x01 基础 & 背景

### 1.1 总体流程图

SERL 的总体流程图如下，其中：

* SAC（Soft Actor-Critic）是算法底座 ，是整个系统的"引擎"。
* RLPD (RL with Prior Data) 是性能加速器，是 SERL 实现"20 分钟学会"的关键，它通过"暴力更新"和"不忘初心"来榨取每一条数据的价值。
* High UTD：迫使模型对有限的样本进行 "深度研读"，从而在极短的物理时间内捕捉到成功的信号。

![SERL-总体-流程](images/img_001.png)

### 1.2 面临的问题

在许多场景中，强化学习的优异表现依赖于与环境进行大量的在线交互，这通常通过使用模拟器来实现。然而，在实际问题中，常常面临样本获取成本高昂的情况。此外，奖励信号稀疏，且高维的状态和动作空间往往使这一问题更加严重。

一些先前的研究致力于通过预训练利用这些数据，而其他方法则在在线训练时引入约束，以应对分布转移问题。然而，每种方法都有其缺点，例如需要额外的训练时间和超参数，或者在行为策略之外的提升有限。

## 0x02 RLPD 基础

RLPD关注的是，是否可以在 在线学习时，直接应用现有的离策略方法以充分利用离线数据。在每一步训练中，RLPD 在先验(离线)数据和on-policy数据之间等概率采样，以形成一个训练批次，即“对称采样”，即每个批次有50%的数据来自（在线）回放缓冲区，另外50%来自离线数据缓冲区（先验数据）。

我们对比如下：

* 原生：纯靠自己试
* SERL：在 SAC 的 Batch 里塞进人类演示数据，相当于给 SAC 考试时递了一张带有参考答案的纸，让它不用从头瞎猜

那"RLPD"这四个字母贵在哪里？秘密不在"怎么算"（Loss Function），而在"喂什么"（Batch Composition）。RLPD 50/50 抽样 强行保证每个 Batch 都有 128 条 Demo。这相当于给机器人装了一个"强制记忆模块"，让它每一秒钟都在看正确答案。

### 2.1 RLPD 的三大支柱

1. High UTD（Update-to-Data）：

* SERL 每跑一步环境，就进行 20 次网络更新。这种"暴力刷题"让每一条物理数据都被反复压榨，是 20 分钟收敛的硬件级保证。

2. Layer Normalization（LN）：

* 问题：在普通的 SAC 中, 网络通常就是纯 MLP。但在 RLPD 中, 由于我们要做 High UTD (高频更新), Q 值的估算非常容易发散。
* 解决方案：因此，RLPD 会在 Critic 网络 (甚至 Actor 网络) 的每一个隐藏层之后, 都加上 LayerNorm。
* 直观理解：高 UTD 会产生巨大的梯度冲击。LN 就给高速赛车装上强力悬挂，确保每一层神经元的输出在 20 倍更新频率下依然稳定，不至于"飞出去"。

3. 50/50 混合采样：

* 做法：在 update 过程中, 我们不再只从 replay\_buffer 抽数据, 而是将演示数据（Demo）与在线数据（Online）强行等比例混合，确保智能体在探索新坑时，始终能看到正确路标。
  + 抽 128 个在线数据 (自己跑出来的)。
  + 抽 128 个先验数据 (人类演示的 Demo)。
  + 把它们拼成一个 256 的 Batch 进行训练。
* 为什么有效?：演示数据告诉智能体"正确的路长啥样", 在线数据告诉智能体"这里的坑别踩"。两者结合, 收敛速度能提升数倍。

### 2.2 Prior Data 策略

Prior Data 策略决定了演示数据与在线数据的黄金采样配比。

真机强化学习最怕"冷启动"— 机器人像没头苍蝇一样乱撞。SERL 引入了 RLPD（Reinforcement Learning with Prior Data）机制：

**数据混洗**：在训练的每一个 Batch 中，系统会强制性地混合：

1. **50% 的演示数据**：这是由人类老师录制好的"标准答案"
2. **50% 的在线数据**：这是机器人自己折腾出来的"实战记录"

**算法价值**：这种混合采样确保了模型在进化的每一秒，都在不断对比"正确答案"与"自己的尝试"。它解决了强化学习初期的探索困境，让机器人即使在完全没拿高分的情况下，也能通过模仿专家数据来迅速建立起任务的初步认知。

我们可以这样理解：

* prior data 像"老师给的标准答案"
* online replay 像"学生自己的练习记录"
* 每次训练都同时看标准答案和自己的错题，策略就不会偏离任务太远

这也是 SERL 样本效率高的关键之一。它不是让机器人从零开始乱试，而是在 demonstrations 的引导下进行强化学习微调。论文中明确写到，每次更新使用 sample-based approximation，其中 half of the samples drawn from prior data，half drawn from replay buffer。

### 2.3 算法流程图

下面是根据 rlpd.py 的代码逻辑整理的 RLPD 训练流程逻辑图。这个图展示了从数据采样到网络更新的完整路径, 特别标注了 RLPD 相对于普通 SAC 的核心改进点 (如 BC Loss 和 Pessimistic Backup)。

![SERL-RLPD-流程](images/img_002.png)

### 2.4 代码细节

RLPD 关键组件说明

* Ensemble Q (10 Qs): 使用 10 个 Q 网络而非 2 个, 增加评估的多样性。
* rho (Pessimism): 通过 "均值 - rho \* 标准差" 来实现对不确定区域的惩罚。
* BC Loss: 在 Actor 更新中加入行为克隆, 强迫智能体初始阶段不要偏离演示数据。
* High UTD (Scan): JAX 的 `lax.scan` 允许在一个硬件循环内执行 20 次上述流程。
* LayerNorm (代码内部) 在所有隐藏层后强制执行归一化, 支撑高频更新。

结合 rlpd.py 代码的深度解读:

1. 关于 rho 的计算:

   ```
   next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
   next_q = next_qs.mean(axis=0) - next_qs.std(axis=0) * self.config["rho"]
   ```

   这是 RLPD 的精髓。普通的 SAC 是 min(Q1, Q2), 而这里是用标准差来量化"不确定性"。如果 10 个 Q 网络对某个状态动作意见不统一 (std 大), next\_q 就会被压得很低。
2. 关于 BC Loss：

   ```
   bc_loss = -(dist.log_prob(jnp.clip(batch_actions, -1 + 1e-5, 1 - 1e-5)) * batch["valid"][..., -1]).mean() * self.config["bc_alpha"]
   ```

   这行代码在告诉 Actor: "不管 Q 值怎么说, 你输出的动作最好和 Buffer 里的真实动作 (演示数据) 接近一些"。这对机器人任务极其关键, 因为它防止了机器人在训练初期因为乱甩而撞坏硬件。
3. 关于 High UTD:

   ```
   @jax.jit
   def batch_update(self, batch):
       agent, infos = jax.lax.scan(self._update, self, batch)
   ```

   这里使用了 JAX 的 scan 原语。这比 Python 的 for 循环快得多, 它能把 20 次更新编译成一个高效的 GPU 算子。
4. 演示数据（Prior Data）：  
   SERL 所基于的 RLPD 算法中，作者发现最简单、最有效的办法就是一视同仁，比如：

   * 从 Replay Buffer（自己跑的数据）取 128 个。
   * 从 Demo Buffer（演示数据）取 128 个。
   * 凑成一个 256 的 Batch，直接丢进 Loss 函数。
   * 意义：不需要复杂的权重计算，这种简单的"对半开"采样就能极大地提升效率。
5. "看未来"与"看现在"的逻辑

   * rlpd.py 中计算 next\_q 用的是 target\_critic, 而计算 actor\_loss 时用的是 critic (当前网络)。
   * Target Critic (看未来): 用于计算 `r + γ Q_{target}`。由于 Q\_{target} 更新得很慢 (Soft Update), 它提供了一个稳定的地基, 防止 Q 值计算产生正反馈螺旋 (即自己把自己估高)。
   * Critic (看现在): 用于 Actor 的更新。Actor 问: "我现在的动作好吗? "。由于 Critic 正在被最快地训练, 它能给 Actor 提供最及时的反馈。
   * 矛盾解决: 这就是"评估要稳 (Target), 改进要快 (Current)"的权衡。

## 0x03 BC

在 SERL 的复现中, 通常的步骤是:

1. 收集 20 个 Demo。
2. 跑 bc.py 进行预训练, 让机器人学会"手往哪放"。
3. 跑 rlpd.py 进行正式训练, 利用 BC 训练好的模型作为起点, 通过 50/50 采样快速进化。

在 SERL 的完整流程中，bc.py 的代码非常关键, 它揭示了 SERL 系统中 Behavioral Cloning (BC) 环节是如何运作的。

### 3.1 BCAgent

BCAgent 负责预训练冷启动，是**纯监督学习**的行为克隆实现，架构最为简洁。SERL先用 BC 模仿 Demo，让机器人学会"手往哪放"，再开启 RL 寻找"怎么抓取"。

#### 3.1.1 BCAgent 在SERL中的作用

1. **冷启动**：为RL算法提供初始策略
2. **演示数据利用**：从专家演示中学习
3. **安全基线**：在RL训练初期提供安全策略

**核心优势**：BCAgent 的简洁性使其成为从演示到强化学习的理想桥梁，通过监督学习快速获得可用的策略，然后可以在此基础上进行RL微调。

BCAgent 的特性如下：

| 特性 | BCAgent | 说明 |
| --- | --- | --- |
| **网络数量** | 仅1个Policy网络 | 无Critic，无Temperature |
| **tanh\_squash** | False | 不使用tanh压缩 |
| **输出分布** | MultivariateNormalDiag | 标准高斯分布 |
| **训练目标** | 最小化MSE + 负对数似然 | 监督学习 |

#### 3.1.2 BCAgent 核心组件

**唯一的 Policy 网络**

```
network_kwargs["activate_final"] = True
networks = {
    "actor": Policy(
        encoder_def,              # 视觉编码器
        MLP(**network_kwargs),    # 默认 [256, 256]
        action_dim=actions.shape[-1],
        tanh_squash_distribution=False,  # 关键差异
    )
}
```

| 组件 | 输入 | 网络结构 | 输出 | 特点 |
| --- | --- | --- | --- | --- |
| **Policy** | 图像观测 | 编码器+MLP[256,256] | 动作分布(μ,σ) | 纯监督学习 |

#### 3.1.3 BCAgent 编码器架构

**"small" 编码器**：

```
encoders = {
    image_key: SmallEncoder(
        features=(32, 64, 128, 256),
        kernel_sizes=(3, 3, 3, 3),
        strides=(2, 2, 2, 2),
        padding="VALID",
        pool_method="avg",
        bottleneck_dim=256,
        spatial_block_size=8,
    )
}
```

**"resnet" 编码器**：

```
encoders = {
    image_key: resnetv1_configs["resnetv1-10"](
        pooling_method="spatial_learned_embeddings",
        num_spatial_blocks=8,
        bottleneck_dim=256,
    )
}
```

**"resnet-pretrained" 编码器**：

```
pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
    pre_pooling=True,
)
encoders = {
    image_key: PreTrainedResNetEncoder(
        pooling_method="spatial_learned_embeddings",
        num_spatial_blocks=8,
        bottleneck_dim=256,
        pretrained_encoder=pretrained_encoder,
    )
}
```

#### 3.1.4 BCAgent 损失函数

```
def loss_fn(params, rng):
    # 前向传播
    dist = self.state.apply_fn(
        {"params": params},
        batch["observations"],
        temperature=1.0,
        train=True,
        rngs={"dropout": key},
        name="actor",
    )
    
    pi_actions = dist.mode()              # 预测动作
    log_probs = dist.log_prob(batch["actions"])  # 对数概率
    
    # 多重损失
    mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)  # MSE损失
    actor_loss = -(log_probs).mean()                     # 负对数似然
    
    return actor_loss, {
        "actor_loss": actor_loss,
        "mse": mse.mean(),
    }
```

#### 3.1.5 BCAgent 动作采样

```
def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
    dist = self.state.apply_fn(
        {"params": self.state.params},
        observations,
        temperature=temperature,
        name="actor",
    )
    if argmax:
        actions = dist.mode()  # 确定性采样
    else:
        actions = dist.sample(seed=seed)  # 随机采样
    return actions
```

### 3.2 bc.py 流程图

BC Agent (模仿学习) 核心流程图如下。BC 关键组件说明：

1. ResNet-10: SERL 的默认视觉骨架，用于从原始像素中提取物理特征。
2. Random Crop: 极其重要的 Trick，通过对画面进行 ±4 像素的裁剪来模拟环境扰动。
3. TanhNormal: 动作分布模型，确保输出的动作符合机械臂的物理范围。
4. Pre-training: BC 在 SERL 中扮演“冷启动”的角色，将 RL 的搜索空间缩减到目标附近。

![SERL-bc.py 流程图](images/img_003.png)

### 3.3 bc.py 技术细节

* 核心逻辑: update 函数

  + 不仅仅是回归: 虽然它计算了 mse, 但实际更新用的是 actor\_loss = -log\_probs.mean()。这是一种概率视角下的模仿: 让策略在演示数据给出的状态下, 输出演示动作的概率尽可能大。
  + 没有 Critic：这里完全没有 Q 网络。BC 纯粹是"看着答案抄答案", 不需要奖励信号。
* 视觉处理: 数据增强 (Data Augmentation) data\_augmentation\_fn：

  + 这是 DrQ-v2 风格的随机裁剪。
  + 为什么重要?: 在机器人任务中, 摄像头画面可能会有轻微抖动。通过对演示图片进行随机裁剪, 可以让策略学会"忽略"这种位移, 提高鲁棒性。这也是 SERL 能在 20 分钟内学会任务的秘诀之一。
* 网络架构: ResNet-10

  + 在 create 方法中, 它定义了三种编码器。SERL 默认推荐的是 resnet:

  ```
  elif encoder_type == "resnet":
   encoders = {
       image_key: resnetv1_configs["resnetv1-10"](...)
   }
  ```

  + ResNet-10 比传统的 CNN 效果好得多, 因为它能提取更深层的特征, 同时又不像 ResNet-50 那样运算缓慢。
* 为什么会有 mse 却不用它更新?

  + 在 update 函数的 loss\_fn 中, 作者计算了 mse = ((pi\_actions - batch["actions"]) \*\* 2).sum(-1), 但返回值的第一项 (真正的 Loss) 是 actor\_loss = -(log\_probs).mean()。
  + 原因: MSE 只关心均值对不对, 而 log\_prob 关心的是整个概率分布。如果人类演示同一个动作时有微小的偏差, log\_prob 能更好地捕获这种"容错性"。
* EncodingWrapper 的作用

  这个包装器能把视觉图像和机械臂自身的状态 (关节角度、末端坐标) 揉在一起。这意味着机器人不仅知道自己"看到了什么", 还知道自己"现在手在哪"。

  ```
  encoder_def = EncodingWrapper(..., use_proprio=use_proprio, enable_stacking=True, ...)
  ```
* 冷启动与热切换：如果复现SERL，一般会把 bc.py 练出来的模型会作为 RLPDAgent.create 时的初始参数 (或权重)。这相当于把原本需要几百万次尝试才能学会的动作, 压缩成了几千步的模仿。
* bc\_loss 的数据生效为（padding="VALID"）。这是因为不能对在线数据做 BC Loss？

  + 在线数据 (智能体自己乱跑出来的) 非常"乱"。
  + 物理后果: 如果强迫智能体去模仿这些乱七八糟的动作, 就像是让一个正在学走路的孩子去模仿自己摔跤的动作。这会导致策略陷入低水平的循环。

## 0x04 High UTD & 稳定性机制

High UTD 的意义：它强迫神经网络在极短的时间内"吃透"每一张图片。

### 4.1 High UTD：把每条真机样本反复研读

#### 4.1.1 理解 UTD（Update-to-Data Ratio）

UTD（Update-to-Data Ratio）表示每采集一条环境数据，算法进行多少次梯度更新。

* 传统 RL 常用 UTD=1：采一步，训一步。
* SERL / RLPD 使用更高 UTD（通常为 20 甚至更高）：采集一条昂贵的真机数据后，learner 会多次从 buffer 中采样并更新网络。

#### 4.1.2 为什么 SERL 需要高 UTD？

我们可以把 High UTD 理解成：真机数据太贵，所以每一帧都要反复研读，不能看一遍就扔。

没有 UTD 的后果：普通的 SAC 每采样一个数据才更新一次。对于机器人这种高维度（ResNet 图像）且数据量极小（只有 2.5 小时数据）的任务：

* 收敛太慢：你可能需要练上 10 天半个月。
* 不稳定性：由于视觉特征（ResNet）需要海量更新才能稳定，如果更新频率太低，视觉头会一直处于"模糊"状态，无法提取有效的位姿信息。

#### 4.1.3 极致的采样效率

High UTD 将数据的价值榨取到了极致：

* 20x 的复习强度： 利用 High UTD（Update-to-Data）策略，机器人每在现实中走一步，后台 Learner 就会对现有数据进行 20 到 40 次的高频更新。
* REDQ 保驾护航： 为了防止这种高强度学习产生幻觉，系统利用 10–20 个 Critic 组成 "陪审团（Ensemble）"，通过取最小值的方式压制过估计偏差。
* 成果：
  + 数据压榨逻辑：真机采集的数据中隐藏着极其细微的物理交互特征（如手爪与工件的摩擦）。通过 High UTD，模型被迫对有限的样本进行"深度研读"，从而在极短的物理时间内捕捉到成功的信号。
  + 这让原本需要几周的训练过程，被压缩到了喝杯咖啡的时间。
* 工程代价：High UTD 对 Learner 的算力提出了严苛要求。这要求 JAX 必须在毫秒级时间内完成多轮反向传播，以确保学习速度始终领先于采样速度。

#### 4.1.4 实现

```
    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["SACAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.
        """
        batch_size = batch["rewards"].shape[0]
        assert (
            batch_size % utd_ratio == 0
        ), f"Batch size {batch_size} must be divisible by UTD ratio {utd_ratio}"
        minibatch_size = batch_size // utd_ratio
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        def scan_body(carry: Tuple[SACAgent], data: Tuple[Batch]):
            (agent,) = carry
            (minibatch,) = data
            agent, info = agent.update(
                minibatch, pmap_axis=pmap_axis, networks_to_update=frozenset({"critic"})
            )
            return (agent,), info

        def make_minibatch(data: jnp.ndarray):
            return jnp.reshape(data, (utd_ratio, minibatch_size) + data.shape[1:])

        minibatches = jax.tree_map(make_minibatch, batch)

        (agent,), critic_infos = jax.lax.scan(scan_body, (self,), (minibatches,))

        critic_infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), critic_infos)
        del critic_infos["actor"]
        del critic_infos["temperature"]

        # Take one gradient descent step on the actor and temperature
        agent, actor_temp_infos = agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"actor", "temperature"}),
        )
        del actor_temp_infos["critic"]

        infos = {**critic_infos, **actor_temp_infos}

        return agent, infos
```

#### 4.1.5 UTD 降为 1 的后果：从"特训班"降级为"自习室"

结论：如果把 UTD 降为 1，效果会大幅变差，甚至完全学不会。

把 cta\_ratio（UTD 比率）从 20 降到 1，导致效果变差的原理主要有三点：

1. **导师还没谱，学生瞎改（Critic Lag）**  
   在 SAC 中，Actor（学生）的更新是基于 Critic（导师）给出的 Q 值梯度的。

* UTD = 20：每跑一步，Critic 都要"刷题" 20 次。这让 Critic 能迅速消化新产生的数据，把 Q 值（身价）算得非常准。当 Actor 来问"我该怎么改"时，Critic 给出的方向是极其精准的。
* UTD = 1：Critic 只能练一次就给建议。对于精密插件任务，由于视觉特征（像素）极其复杂，Q 值需要海量的更新才能捕捉到"插头对准插座"那一瞬间的剧烈价值波动。此时 Critic 的评估可能还是"模糊的"。
* 后果：Actor 顺着错误的梯度方向去改，只会越练越废。

2. **视觉特征提取的滞后（Encoder Training）**  
   SERL 使用的是从像素开始的端到端学习。

* 原理：ResNet 需要大量的反向传播才能从杂乱的背景中认出"插孔"。
* 对比：20 倍的更新频率意味着视觉编码器（Encoder）的学习速度提高了 20 倍。如果 UTD=1，你的机器人可能练了一整天，ResNet 还没看清物体的轮廓。

3. **数据压榨率（Sample Reuse）**  
   机器人数据是"昂贵的"（需要电机动，需要时间）。

* UTD = 20：每一条真实的物理轨迹，都会被拿出来反复揉搓 20 遍。
* UTD = 1：这条数据用一次就扔了，就像"富二代"在浪费极其稀有的成功样本。
* 后果：在数据极其稀疏的"精密操作"中，UTD=1 会导致机器人根本无法在有限的 1 小时训练内通过自发尝试撞到正确答案。

#### 4.1.6 High UTD 的副作用与应对

但 High UTD 也有副作用。对同一批数据反复训练，critic 容易过拟合和过估计，最终让策略崩溃。因此，SERL 还需要配套的稳定性机制。

### 4.2 稳定性机制

High UTD 是发动机，但发动机太猛就需要刹车系统。SERL 通过多种机制的协同，实现了在极高样本效率下的稳定训练。

**整体稳定性保障**：

1. **Critic Ensemble / REDQ**：多个 critic 像陪审团，随机子采样取最小值，压制 High UTD 带来的估值爆炸
2. **Critic LayerNorm**：让高频更新不至于数值失控，支持更高 UTD ratios
3. **Soft Update**：让目标网络缓慢跟随，维持 Bellman 目标平稳，保证策略更新平滑
4. **RLPD 50/50 采样**：demo 数据作为锚点，防止策略偏离专家分布
5. **DrQ 数据增强**：random\_crop 提供最重要的视觉正则化
6. **Actor encoder stop-gradient**：防止 actor loss 破坏视觉表征

这些机制不是孤立工作的，而是协同配合。SERL 的工程价值在于不是单独实现某个技巧，而是把一整套相互配合的稳定性机制整合起来，使得高 UTD 这种"激进"的训练策略能够在真机上稳定运行，形成一套可工作的系统。SAC 的巧妙之处恰恰在于它如何利用"不确定性"来获得最终的稳定。

我们接下来选择部分机制进行解读。

## 0x05 Layer Normalization

论文中提到，regularizing the critic with layer normalization allows for higher UTD ratios and thus more efficient training。也就是说，SERL 并不是单纯把 UTD 拉高，而是通过 critic 正则化让高频更新不至于数值失控。

即，为了抗住 20 倍的更新强度而不崩盘，SERL 在 Critic 网络中引入了（LayerNorm）。这在传统 SAC 中是不常见的，但在高 UTD 的 RLPD 算法中至关重要。

### 5.1 当前层归一化实现分析

从 MLP可以看到已有的层归一化支持：

```
class MLP(nn.Module):
    use_layer_norm: bool = False  # 层归归一化开关

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x) # 线性变换
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                # 正则化层（可选）
                if self.dropout_rate is not None and self.dropout_rate > 0: 
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:  # 关键：层归一化应用
                    x = nn.LayerNorm()(x) # 标准化层输出 	
                x = activations(x) # 激活函数
        return x
```

### 5.2 层归一化的技术细节和优势

**Critic 网络的特点**：

1. **输入方差大**：观测编码和动作拼接导致输入分布不稳定
2. **梯度爆炸风险**：深度网络容易出现梯度问题
3. **Ensemble 训练**：多个 Critic 网络需要稳定的训练动态

**层归一化的具体好处**：

* **稳定训练**：减少内部协变量偏移
* **提高学习率**：可以使用更大的学习率
* **加速收敛**：减少训练震荡
* **改善泛化**：对输入扰动更鲁棒

### 5.3 实现方案

SACAgent 创建时启用层归一化.

```
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
```

针对 DrQAgent 的实现

```
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
```

VICE 中的层归一化（已实现）

```
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        vice_network_kwargs={
            "activations": nn.leaky_relu,
            "use_layer_norm": True,
            "hidden_dims": [
                256,
            ],
            "dropout_rate": 0.1,
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
```

### 5.4 总结

对 Critic 进行层归一化正则化的关键是：

1. **配置启用**：在 `critic_network_kwargs` 中设置 `use_layer_norm: True`
2. **正则化组合**：配合 Dropout 和权重衰减获得最佳效果
3. **超参数调整**：层归一化后可以使用更大的学习率
4. **针对性优化**：根据任务类型（视觉/状态）调整正则化强度
5. **性能监控**：添加统计信息验证层归一化的实际效果

在 SERL 框架中，这种实现方式既保持了代码的简洁性，又充分利用了 Flax/JAX 的模块化优势，是提高 Critic 网络训练稳定性和性能的有效手段。

## 0x06 Soft Update & REDQ

Soft Update 让目标网络始终缓慢追踪当前 Q 值，保持贝尔曼目标的平稳性。在 REDQ 的高 UTD 场景下尤为重要。

### 6.1 Soft Update 的力量

在机器人控制中，动作的连续性决定了硬件的寿命。SERL 坚持使用 Soft Update（软更新）维护目标网络：

* **平滑公式**：`θ(target) = τ θ_online + (1−τ) θ{target}`。其中 τ 通常设为极其微小的 0.005。
* **硬件意义**：与直接拷贝权重的 Hard Update 不同，Soft Update 让目标值（Target）以一种近乎流体的方式缓慢漂移。这反映到机器人身上，就是动作的进化是"渐进"的，不会因为模型权重的突跳导致机械臂产生瞬时的冲击电流或抖动。

Soft update的核心实现如下：

```
def target_update(self, tau: float) -> "JaxRLTrainState":
    """
    Performs an update of the target params via polyak averaging. The new
    target params are given by:

        new_target_params = tau * params + (1 - tau) * target_params
    """
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), self.params, self.target_params
    )
    return self.replace(target_params=new_target_params)
```

这个方法在 SACAgent.update 中被调用：

```
# Update target network (if requested)
if "critic" in networks_to_update:
    new_state = new_state.target_update(self.config["soft_target_update_rate"])
```

**原理分析：** Soft Update采用Polyak averaging的方式缓慢更新目标网络。这种方法的核心思想是让目标网络以平滑的方式跟踪主网络，而不是周期性地完全复制。这种平滑跟踪有助于：

* 减少训练过程中的方差
* 提高算法的稳定性
* 防止因目标网络剧烈变化导致的训练震荡

### 6.2 REDQ：Critic Ensemble 抑制过估计

REDQ 模式：支持把 Q 网络增加到 10 个以上，并从中随机抽 2 个来计算 Target。这是另一种对抗高估问题的强力方法。

* 原生：2 个 Critic
* SERL：10–20 个 Critic
* 作用：支持 High UTD（UTD=20）。如果没有这么多 Critic 压阵，SAC 会在疯狂更新中产生严重的数值爆炸。

#### 6.2.1 为什么需要 Critic Ensemble？

High UTD 虽然能加速学习，但会带来致命副作用：**Q 值过估计（Overestimation Bias）**。模型会因为反复研读少量样本而变得极端自信，单Q网络容易高估未见过的状态一动作对的价值，最终导致策略崩溃。

SERL 引入了 REDQ（Randomized Ensembled Double Q-learning）风格的机制来解决这个问题。我们可以将其理解为一种"陪审团机制"：

* **Critic Ensemble（陪审团）**：同时训练 10 到 20 个独立运行的 Critic 网络
* **随机子集采样（Randomized Subset）**：在计算目标 Q 值时，并不看所有人的意见，而是随机抽取 2 个 Critic
* **取最小值（In-sample Min）**：在抽出的子集中取分数的最小值，min操作天然抑制0OD区域的过高估计

通俗地说：如果十个裁判里随机抽出的几个裁判中，有一个觉得这个动作危险，那我们就保守一点。这种"悲观主义"巧妙地抵消了 High UTD 带来的"狂热乐观"，使训练在极高强度下依然稳如磐石。

#### 6.2.2 REDQ论文算法

算法如下：

![REDQ算法](images/img_004.png)

**训练时的行为**：

1. ‌**随机采样**‌：从 N 个（默认 N=10）Critic 网络中随机无放回地选取 M 个（默认 M=2）子集索引。
2. ‌**前向传播**‌：‌**仅**‌将这 M 个网络的参数用于计算目标状态动作值 Q(s′,a′)*Q*(*s*′,*a*′)。
3. ‌**取最小值**‌：对这 M 个输出值取最小值作为 Target Q 值的一部分（即 min⁡i∈subsetQi(s′,a′)。
4. ‌**损失计算**‌：虽然 Target 只用了 M 个网络，但在计算 Critic 损失时，‌**所有 N 个网络**‌都会根据同一个 Target 进行梯度更新，以维持集成多样性 。

**这种设计既保持了ensemble的容量优势，又通过子采样降低了计算成本和过拟合风险。**

1. **计算效率**：只计算 K 个网络的前向传播，而非全部 N 个
2. **正则化效果**：随机子采样引入额外噪声，提高泛化能力
3. **过拟合缓解**：避免始终使用相同的"最好"网络

REDQ 论文证明：`min(2 from 10)` 的效果接近 `min(10)`，但计算量减少 5 倍。

#### 6.2.3 SERL 的实现

![SERL-REDQ](images/img_005.png)

##### 架构细节

```
# drq.py:124-125 和 launcher.py:165-166
critic_ensemble_size=10,    # 10 个独立 Q 网络
critic_subsample_size=2,    # 计算 target 时只随机选 2 个
```

##### 具体实现

```
def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
    # ...前期准备代码...
    
    # 1. 计算所有ensemble成员的Q值
    target_next_qs = self.forward_target_critic(
        batch["next_observations"],
        next_actions,
        rng=rng,
    )  # shape: (critic_ensemble_size, batch_size)

    # 2. 如果配置了子采样，则随机选择指定数量的网络
    if self.config["critic_subsample_size"] is not None:
        rng, subsample_key = jax.random.split(rng)
        subsample_idcs = jax.random.randint(
            subsample_key,
            (self.config["critic_subsample_size"],),  # 通常是2
            0,
            self.config["critic_ensemble_size"],  # 通常是10
        )
        target_next_qs = target_next_qs[subsample_idcs]  # 只保留选中的网络

    # 3. 在(子采样后的)ensemble成员中取最小值
    target_next_min_q = target_next_qs.min(axis=0)  # shape: (batch_size,)
    
    # ...后续使用target_next_min_q计算TD目标...
```

##### 与论文的区别

SERL 先计算所有10个网络的Q值，然后随机选择2个网络进行子采样，最后在这2个中选择最小值。

**详细执行流程**：

1. **前向传播所有ensemble**：首先计算所有10个critic网络对(next\_state, next\_action)的Q值
2. **随机子采样**：从10个网络中随机选择2个网络的索引（`subsample_idcs`）
3. **提取子样本**：`target_next_qs[subsample_idcs]` 只保留这2个网络的Q值
4. **取最小值**：在这2个网络的Q值中取最小值作为bootstrapping目标

**REDQ为什么这样设计**：

1. **前向传播成本低**：注释中明确提到"Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)"，因为没有梯度计算，计算所有10个网络的代价相对较小
2. **子采样减少复杂度**：后续操作（梯度计算等）只在2个网络上进行，大大减少了计算复杂度
3. **随机性的好处**：每次更新随机选择不同的网络组合，相当于隐式的bagging，增加了训练的随机性和泛化能力
4. **过拟合缓解**：通过随机子采样和取最小值的方式，能够有效缓解critic过拟合问题

这种设计在保持计算效率的同时，充分利用了ensemble的多样性，是REDQ算法的核心创新点之一。

### 6.3 时间戳对齐

代码中完全没有观测 - 动作的时间戳对齐机制。但这是刻意的设计取舍：

**实际的时序设计**：

```
step(action) 被调用
    ├─1. 计算目标位姿 + 安全裁剪        ~0.1ms
    ├─2. _send_gripper_command()     ~600ms (夹爪动作时)
    ├─3. _send_pos_command()         ~5ms (HTTP POST)
    ├─4. time.sleep(1/hz - elapsed)  补齐到 100ms 控制周期
    ├─5. _update_curipos()           ~5ms (HTTP POST /getstate)
    └─6. _get_obs()
        ├ get_im()                   ~10~30ms (从 VideoCapture 队列取最新)
        └ 组装 state (用步骤5的数据)
```

**为什么不做精确对齐？**

1. **相对偏差可接受**：10Hz 控制周期 = 100ms 间隔，相机 30fps = 33ms 间隔，最坏情况图像延迟 33ms，相对控制周期只有 1/3 偏差
2. **硬件限制**：精确对齐需要硬件触发同步，当前架构不支持
3. **算法容错性**：SAC/DrQ 学习的是随机策略，天然对观测噪声有一定容忍度

SERL 在系统设计上做出了取舍，专注于整体架构的简洁性和可维护性，而非追求极致的硬件同步精度。

## 0x07 SAC vs RLPD

### 7.1 相同之处

RLPD 的 Critic 更新和 SAC 没有区别。

* 在 SAC 中：Target = r + γ min(Q₁, Q₂)(s', π(s'))
* 在 RLPD 中：Target = r + γ min(Q₁, Q₂)(s', π(s'))

对于 Batch 里的每一个样本（无论是来自 Demo 还是 Online），Critic 都在做同一件事：Loss = (Q(s, a) - Target)²，其中 Target = r\_{vice} + γ Q(s', a')。

注意：

* 这里的 a 就是当时实际发生的那个动作。
* 如果是 Online 数据，a 是机器人自己做的。
* 如果是 Demo 数据，a 是人做的。
* Critic 并不在乎这个动作是谁做的，它只负责给这个"动作-状态"对打分。

### 7.2 不同之处

这里的"特殊处理"不在公式里，而是在数据的质量上。RLPD 并没有给 Critic 写一个"针对 Demo 的特殊公式"，它的特殊在于强行喂给 Critic 50% 的高质量样本。秘密不在"怎么算"（Loss Function），而在"喂什么"（Batch Composition）。

* Online 样本：机器人的动作可能是乱挥（`a_{random}`），它对应的 r\_{vice} 大概率是 0。
* Demo 样本：人类的动作是精准的（`a_{demo}`），它对应的 r\_{vice} 大概率是 1（或者接近 1 的高分）。

所有数据（无论谁做的）都必须经过 VICE 的安检。VICE 说它是成功，它才是成功。

* 当 VICE（裁判）坏了，始终给 0 分时，会发生什么？
  + 不公平的待遇：原本该拿 1 分的 a\_{demo} 现在只能拿 0 分。在线试错 被 VICE 判为 0。人类演示 也被 VICE 判为 0。
  + 后果：Critic 学到的是："不管是人类做的那个精妙动作，还是我刚才那个乱挥的动作，身价全都是 0"。Critic 只能接受着这个事实 —— "这个世界没有奖赏，做什么都是徒劳"。Q.网络随之萎缩到 0。

### 7.3 通俗解释

想象你和机器人都在学数学：

* 普通 SAC（像自学）：
  + 机器人每天自己做 256 道题。因为是新手，错题率 99%。
  + 机器人看着这些错题（\(r=0\)），吃力地总结经验。
  + 弱点：因为从来没见过正确解法（\(r=1\)），它可能要练 1 万年才能偶尔撞对一次。
* RLPD（像有标准答案的刷题）：
  + 机器人每天做 128 道自己的题（Online），另外读 128 道满分卷子（Demo）。
  + 它计算 Critic Loss 时，虽然公式一样，但那 128 道满分卷子带来的 Target 里的 r 是 1。
  + 强项：它时刻被提醒"什么是正确的"。Critic 会迅速学到："哦！人类做的那些动作才是值钱的，我自己瞎搞的那些不值钱"。

## 0xFF 参考
