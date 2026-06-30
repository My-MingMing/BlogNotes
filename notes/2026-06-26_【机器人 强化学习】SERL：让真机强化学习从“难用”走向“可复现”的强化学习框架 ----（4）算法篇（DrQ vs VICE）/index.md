# 【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（4）算法篇（DrQ vs VICE）

目录

* [【机器人 / 强化学习】SERL：让真机强化学习从“难用”走向“可复现”的强化学习框架 ----（4）算法篇（DrQ vs VICE）](#机器人--强化学习serl让真机强化学习从难用走向可复现的强化学习框架-----4算法篇drq-vs-vice)
  + [0x00 概要](#0x00-概要)
  + [0x01 奖励与自动化：从手写 reward 到 learned reward](#0x01-奖励与自动化从手写-reward-到-learned-reward)
    - [1.1 Success Classifier：用少量图像训练自动裁判](#11-success-classifier用少量图像训练自动裁判)
    - [1.2 VICE：把奖励分类器也纳入训练](#12-vice把奖励分类器也纳入训练)
  + [0x02 DrQ vs VICE](#0x02-drq-vs-vice)
    - [2.1 DrQ Agent (Data-regularized Q-learning)](#21-drq-agent-data-regularized-q-learning)
    - [2.2 VICE Agent (Variational Inverse Control with Events)](#22--vice-agent-variational-inverse-control-with-events)
    - [2.3 Agent体系](#23-agent体系)
      * [2.3.1 SACAgent 核心机制](#231-sacagent-核心机制)
      * [2.3.2 继承体系](#232-继承体系)
    - [2.4 深度对比表](#24-深度对比表)
  + [0x03 DrQ：让视觉 RL 更稳健](#0x03-drq让视觉-rl-更稳健)
    - [3.1 数据增强的作用](#31-数据增强的作用)
    - [3.2 视觉编码器体系](#32-视觉编码器体系)
      * [3.2.1 EncodingWrapper（特征融合） 设计](#321-encodingwrapper特征融合-设计)
      * [3.2.2 视觉 - 本体感知 Fusion 详解](#322-视觉---本体感知-fusion-详解)
      * [3.2.3 多任务接口条件](#323-多任务接口条件)
    - [3.3 算法实现](#33-算法实现)
      * [3.3.1 DrQAgent 网络架构](#331-drqagent-网络架构)
      * [3.3.2 编码器选择](#332-编码器选择)
      * [3.3.3 `async_drq_sim.py`代码细节](#333-async_drq_simpy代码细节)
      * [3.3.4 drq.py 逻辑流程图](#334-drqpy-逻辑流程图)
      * [3.3.5 drq.py 代码拆解](#335-drqpy-代码拆解)
  + [0x04 VICE](#0x04-vice)
    - [4.1 背景 & 原理](#41-背景--原理)
      * [4.1.1 动机：为什么要搞 VICE？](#411-动机为什么要搞-vice)
      * [4.1.2 变分推断（Variational Inference）](#412-变分推断variational-inference)
        + [贝叶斯推断的困境](#贝叶斯推断的困境)
        + [变分推断的解决方案](#变分推断的解决方案)
      * [4.1.3 VICE 原理](#413-vice-原理)
      * [4.1.4 对抗性进化：猫鼠游戏](#414-对抗性进化猫鼠游戏)
    - [4.2 网络架构](#42-网络架构)
    - [4.3 奖励（reward）的计算](#43-奖励reward的计算)
    - [4.4 训练逻辑](#44-训练逻辑)
    - [4.5 流程图](#45-流程图)
    - [4.6 工程细节](#46-工程细节)
      * [4.6.1 分类即奖励（vice\_reward）](#461-分类即奖励vice_reward)
      * [4.6.2 策略 & 效果](#462-策略--效果)
      * [4.6.3 工程上的"防崩盘"设计](#463-工程上的防崩盘设计)
  + [0xFF 参考](#0xff-参考)

## 0x00 概要

本篇主要介绍 DrQ 和 VICE，

* 机器人必须通过像素看世界，DrQ 解决了视觉特征提取的泛化难题。
* VICE 解决了"真实世界没有代码奖励"的问题，让机器人拥有了"成就感"。

![SERL-DrQ-VICE](images/img_001.png)

注：

* **本系列的最终目标是“通过一系列相关项目/算法的解读，来深入学习/分析/反推 LWD（Learning while Deploying）这篇论文的机理和可能实现”**。之所以从SERL入手，是因为 SERL，HIL-SERL，SOP（没有开源）都是罗剑岚博士的一系列论文，可以从中管窥作者的思路脉络。
* 本文依然是从工程/论文进行反推，还请读者不吝指出问题，多谢。

## 0x01 奖励与自动化：从手写 reward 到 learned reward

真实机器人任务的 reward specification 是 SERL 重点解决的问题之一。

某些任务可以用机器人状态直接定义，例如 PCB 插入可以根据末端或物体位置设计奖励；在这种情况下，奖励函数可以由一个二元分类器提供，该分类器接收状态观测 s 并输出一个二元” 事件”e发生的概率，对应于任务的成功完成。奖励随后由 r(s) = log p(e|s) 给出。

但多数任务需要从图像中判断是否完成。比如"电缆是否正确卡入槽位""物体是否被放到正确 bin 里"，往往需要视觉判定，手工设计一个好的奖励函数本身就是一个耗时且需要领域专长的任务。

SERL 支持三类奖励方式：

1. **手写奖励**：适用于状态足以判断成功的任务
2. **二分类成功判别器**：用成功/失败图像训练 classifier
3. **VICE**：把 reward classifier 与 RL 过程结合，在训练中利用策略产生的负样本更新分类器

![SERL-reward](images/img_002.png)

### 1.1 Success Classifier：用少量图像训练自动裁判

用户可以采集少量成功与失败图片，训练一个轻量视觉分类器。该分类器在训练过程中输出成功概率或二值 reward。即，人只需要定义成功 / 失败样本，之后系统就能自动在训练中判断 reward。这就是自动裁判（Success Classifier）:

Success Classifier 是离线训练专用，是简单的端到端结构：

* 架构： 基于轻量级视觉分类器，利用少量正负样本图像快速预训练。
* 职责： 在训练过程中实现 Reward 自动化，使系统具备无人值守的自我优化能力。

* 流程: 先收集正/负样本 pkl → BCE loss 训练 ResNet-10 冻结 + MLP head → 保存 checkpoint → 加载到 env wrapper 中作为固定奖励函数

这里需要注意：SERL 不是完全不需要 reward，而是提供了工具让 reward specification 更容易、更自动化。SERL 的目标不是消除 reward 设计，而是降低 reward engineering 的成本。

### 1.2 VICE：把奖励分类器也纳入训练

VICE 将奖励学习看成类似 GAN 的过程：策略像 generator，不断产生新样本；reward classifier 像 discriminator，判断哪些状态是成功事件。这样，随着策略分布变化，classifier 可以用策略产生的新负样本继续训练，避免只在初始数据上过拟合。

这种设计让奖励系统能够自适应策略的进化。初始阶段，classifier 在人类标注的样本上训练；随着策略能力的提升，classifier 可以从策略产生的经验中学习更细粒度的成功模式，实现奖励的持续改进。

VICE 是在线训练， 支持 return\_encoded / classify\_encoded 模式

* serl-main/serl\_launcher/serl\_launcher/agents/continuous/vice.py
* VICEAgent 继承自 DrQAgent, 分类器嵌入 agent 内部, 与 RL 训练交替在线更新
* 额外正则化: Mixup + Label Smoothing + Gradient Penalty

## 0x02 DrQ vs VICE

在具身智能和强化学习的语境下，DrQ 和 VICE 是两套针对不同痛点的经典 Agent（代理）架构。

简单来说：DrQ 是为了 "看得更清"，而 VICE 是为了 "罚得更准"。

* VICE（上帝之眼）：解决"是否到达"，通过分类器提供平滑、客观的奖励信号。
* 视觉鲁棒性（DrQ）
  + 随机裁剪：对 s 和 \(s'\) 使用不同偏移，强迫网络学习物理特征而非像素坐标。
  + 共享编码器：Actor 和 Critic 共用 ResNet-10，加速特征收敛。

这样可以视觉与硬件包装：眼手合一。

### 2.1 DrQ Agent (Data-regularized Q-learning)

用途：解决 "从图像直接学习动作（Pixel-to-Action）" 时的样本效率和泛化问题。

* 它的痛点： 如果直接把原始图片丢给 RL 模型，机器人很容易 "只看局部"，或者因为光线变了一点就完全不会动了。
* 核心特色：图像增强（Data Augmentation）+ 一致性。

  + 做法：机器人每看到一张图，DrQ 会在后台生成它的几个 "微调版"（比如随机裁剪 4 个像素、稍微改变亮度）。
  + Q-Learning的一致性： DrQ 强制要求模型对这几张 "微调版" 照片预测出的 Q 值必须是一样的。
* 优点：

  + 极强的稳定性： DrQ 像是在训练时给机器人戴上了各种滤镜，强迫模型学会看透事物的本质特征。
  + 极高的样本效率： 即使数据量很少，通过图像增强，模型也能学到稳健的特征。

### 2.2 VICE Agent (Variational Inverse Control with Events)

用途： 解决 "自动定义奖励（Reward）" 的问题，特别是针对那些难以用数值描述的 "事件"。

* 它的痛点： 你怎么告诉机器人 "盖子盖紧了"？你很难写代码去算角度。
* 核心特色：反向学习（Inverse RL）+ 事件判定。

  + 做法： 开发者给机器人看一堆 "成功的快照"。（比如 20 张盖好盖子的照片）。VICE 会通过这些照片训练一个判别器（Classifier）。
  + 特色奖励： 在机器人自己尝试时，VICE 会根据它当前的画面 "多像" 那堆成功快照，自动产生一个 θ 到 1 之间的分值。
* 优点：

  + 零代码定义 Reward： 只要有成功的照片，你就拥有了奖励函数。
  + "事件" 驱动： 它的逻辑是 "我只要结果那个状态"，非常适合解决那些结果明确但过程复杂的任务（如：扣上纽扣）。

### 2.3 Agent体系

在 SERL 中，SACAgent 是核心基类。

#### 2.3.1 SACAgent 核心机制

| SACAgent 核心机制 | 功能 |
| --- | --- |
| `critic_loss_fn()` | Clipped Double Q + ensemble min + 可选 entropy backup |
| `policy_loss_fn()` | 标准 SAC actor 目标：Q−a⋅logπ |
| `temperature_loss_fn()` | Lagrange 乘子自动调温 |
| `update_high_utd()` | 关键优化：UTD ratio 多次 critic 更新 + 1 次 actor 更新，用 `jax.lax.scan` 实现 JIT 内循环 |
| `update()` | 支持选择性更新子网络（`networks_to_update` 参数） |

#### 2.3.2 继承体系

DrQAgent 和 VICEAgent 是 SACAgent 的子类。这意味着它复用了 SAC 的所有稳定机制（双 Q、熵自动调节），只是在输入端加了一层图像处理逻辑。

DrQAgent 增强 (drq.py)

* 对 observations 和 next\_observations 做 random crop 数据增强
* 支持 3 种编码器：small (4 层 CNN)、resnet（从零训练）、resnet-pretrained（ImageNet 预训练 ResNet-10 冻结 + 可训练 pooling head）

VICEAgent 奖励学习 (vice.py)

* 用 Binary Classifier 替代手工奖励函数
* 训练技巧：Mixup + Label Smoothing + Gradient Penalty（防 GAN 模式崩塌）
* 推理时：`sigmoid(logit) >= 0.5 → reward=1, 否则 reward=0`

```
SACAgent (sac.py:21)
├─ create_pixels()    → 像素输入，共享/独立编码器
└─ create_states()    → 状态输入，无编码器
    └─ DrQAgent (drq.py:23) 继承 SACAgent
        ├─ create_drq()          → 支持 small/resnet/resnet-pretrained 编码器
        ├─ data_augmentation_fn() → 随机裁剪增强
        ├─ update_high_utd()     → 高 UTD 比率训练（关键优化）
        └─ update_critics()     → 仅更新 Critic
            └─ VICEAgent (vice.py:26) 继承 DrQAgent
                ├─ create_vice()      → 额外 VICE 分类器网络
                ├─ update_vice()      → BCE + Mixup + Label Smoothing + 梯度惩罚
                ├─ vice_reward()      → 用分类器替代手工奖励
                └─ update_high_utd() → 用 VICE 奖励替换环境奖励
```

### 2.4 深度对比表

| 特性 | DrQ Agent | VICE Agent |
| --- | --- | --- |
| 主要任务 | 视觉表征学习（把图看透）。 | 奖励函数生成（把分打准）。 |
| 核心技术 | 随机裁剪、平移（Augmentation）。 | 基于正负样本的对抗学习 / 判别。 |
| 解决的难题 | 图像噪声、过拟合、泛化差。 | 奖励函数难写、信号稀疏。 |
| 在 SERL 里的位置 | 作为视觉主干（Backbone）的增强。 | 作为自动裁判（Reward Classifier）的原型。 |

## 0x03 DrQ：让视觉 RL 更稳健

真实机器人通常使用图像观测。如果直接用像素训练，视觉过拟合会非常严重。DrQ 的核心是对图像做数据增强，例如 random crop，让策略不要记住某个固定像素位置，而是学到更鲁棒的视觉特征。

SERL 的 `DrQAgent` 在 SAC 基础上增加了图像增强和视觉编码器支持。

### 3.1 数据增强的作用

DrQ 最重要的特性是 random crop 数据增强。这在真实机器人环境中尤为关键：

* **视觉正则化**：阻止模型记住绝对像素位置，强迫其学习鲁棒的空间特征
* **对真实世界的适应性**：真实环境中光照、背景、物体位置都会有微小变化，数据增强让模型对此具备天然的泛化能力

特色解释：为什么 DrQ 在 SERL 中不可或缺？

* 像素级鲁棒性：如果没有 DrQ 的数据增强，机械臂只要位置偏了一点，或者光照变了一下，RL 就会失败。DrQ 通过随机裁剪，让网络学会关注"物体本身"，而不是"物体在图像中的绝对坐标"。
* 极致的性能：在 update\_high\_utd中，可以看到 DrQ 是在进行高 UTD 更新之前，先做一次数据增强。
* 细节：由于 batched\_random\_crop 是在 JAX 内部实现的，它运行在 GPU 上。相比于在 CPU 上做增强再传给 GPU，这节省了巨大的数据传输开销。

论文实验中使用了 ImageNet 预训练 ResNet-10 作为视觉 backbone，并连接到 MLP。观测包含相机图像和机器人本体信息，例如末端位姿、速度、力、力矩等。

### 3.2 视觉编码器体系

#### 3.2.1 EncodingWrapper（特征融合） 设计

`EncodingWrapper` 是 SERL 处理视觉和本体感知输入的核心封装：

* **多图像键支持**（dict of encoders）
* **帧堆叠处理**：`rearrange(image, "T H W C -> H W (T C)")` 和 `rearrange(image, "B T H W C -> B H W (T C)")`
* **本体感知融合**：Dense → LayerNorm → tanh → concat
* **`stop_gradient` 支持**：允许冻结视觉编码器
* **将视觉像素与本体感知（关节角度）拼接**。视觉告诉你物体在哪，本体感知告诉你手在哪。"眼手合一"是精细操作的前提。

**编码器选项**：

* **SmallEncoder**：4 层 Conv (32→64→128→256) + Spatial Softmax + bottleneck
* **ResNet-10 (from scratch)**：10 层 ResNet + Spatial Learned Embeddings
* **PreTrained ResNet-10**：ImageNet 预训练冻结 + 可训练 pooling head

**Spatial Softmax / Spatial Learned Embeddings** 是一个关键设计：它们将卷积特征图转为空间坐标的加权期望，保留空间信息同时大幅降维。这比全局池化更能保留空间位置信息，对机器人视觉任务至关重要。

#### 3.2.2 视觉 - 本体感知 Fusion 详解

在 SERL 中，观测数据结构如下：

```
observations = {
    "image1": (T, H, W, C),   # 相机1图像帧栈
    "image2": (T, H, W, C),   # 相机2图像帧栈
    "state":  (T, D),         # 本体感受 (tcp_pose, tcp_vel, gripper, force, torque)
}
```

`EncodingWrapper` 的处理流程如下：

![SERL-EncodingWrapper](images/img_003.png)

**Late Fusion 设计**：视觉和本体感受各自独立编码，最后拼接。这不是 Early Fusion，而是有深刻的设计考量：

1. **模态异质性**：图像是 2D 空间信号，适合卷积；state 是 1D 向量，适合全连接
2. **梯度隔离**：Late Fusion 允许对视觉编码器做 `stop_gradient`，冻结视觉特征只训练本体感受分支
3. **本体感受的 LayerNorm + tanh**：state 的各维度量纲差异大（位置 vs 速度 vs 力），归一化到统一尺度，限制范围避免梯度爆炸

#### 3.2.3 多任务接口条件

代码中预留了但未使用的多任务接口：

|  | 功能 | 状态 |
| --- | --- | --- |
| GCEncodingWrapper | Goal-Conditioned: obs + goal 图像 fusion | 代码存在但未被任何 Agent 使用 |
| LCEncodingWrapper | Language-Conditioned: obs + 语言指令 fusion | 代码存在但未被任何 Agent 使用 |

这些接口为未来的多任务学习或条件策略扩展预留了空间。

### 3.3 算法实现

#### 3.3.1 DrQAgent 网络架构

由于 DrQAgent 继承 SACAgent，其网络架构基本相同，但有以下关键差异：

| 特性 | SACAgent | DrQAgent |
| --- | --- | --- |
| **Actor架构** | 编码器+MLP[256,256] | 相同 |
| **Critic架构** | 编码器+Ensemble MLP[256,256]×2 | 相同 |
| **数据增强** | 无 | 随机裁剪(4像素填充) |
| **训练稳定性** | 标准SAC | 增强提高泛化 |

#### 3.3.2 编码器选择

DrQAgent 支持多种编码器类型：

1. **SmallEncoder**：轻量级4层卷积网络
2. **ResNet-10**：中等深度ResNet
3. **ResNet-10-Pretrained**：ImageNet预训练的ResNet-10

#### 3.3.3 `async_drq_sim.py`代码细节

以 `async_drq_sim.py` 为例，SERL 的训练流程是典型的异步 actor-learner 架构：

**1. Learner 初始化**：创建 TrainerServer，注册 replay buffer，等待数据填充

**2. Actor 初始化**：创建 TrainerClient，连接 Learner

**3. Actor 循环**：

* 采样动作（前 N 步随机，之后用策略）
* `env.step()` → 收集 transition → `data_store.insert()`
* 定期 `client.update()` → 推送数据
* 定期 `evaluate()` → 发送 stats

**4. Learner 循环**：

* 等待 buffer 填充到 `training_starts`
* 发送初始网络参数给 Actor
* **RLPD 50/50 采样**：batch 一半来自 demo\_buffer，一半来自 online replay\_buffer
* `update_critics()` × (UTD-1) 次 → `update_high_utd(utd_ratio=1)` × 1 次
* 定期 `server.publish_network()` 同步参数
* WandB 日志记录

这种异步架构让机器人控制的高实时性要求和训练的高吞吐量要求解耦，互不干扰。

#### 3.3.4 drq.py 逻辑流程图

![SERL-drq.py 逻辑流程图](images/img_004.png)

#### 3.3.5 drq.py 代码拆解

* 数据增强（data\_augmentation\_fn）：这是 DrQ 的"灵魂"。它对 observations 中的每一个 pixel\_key（图像键）执行 batched\_random\_crop。
  + 原理：即使图像只位移了 1-2 个像素，对于神经网络来说也是全新的输入。这相当于有限的数据池注水，极大增加了泛化性。

```
def data_augmentation_fn(self, rng, observations):
    for pixel_key in self.config["image_keys"]:
        observations = observations.copy(
            add_or_replace={
                pixel_key: batched_random_crop(
                    observations[pixel_key], 
                    rng, 
                    padding=4, 
                    num_batch_dims=2
                )
            }
        )
    return observations
```

训练时，会对观测和下一观测进行数据增强。

```
def update_high_utd(self, batch, utd_ratio, pmap_axis=None):
    # 训练时对观测和下一观测进行数据增强
    rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
    obs = self.data_augmentation_fn(obs_rng, batch["observations"])
    next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
    
    batch = batch.copy(
        add_or_replace={
            "observations": obs,
            "next_observations": next_obs,
        }
    )
    
    # 调用父类SAC的更新
    return SACAgent.update_high_utd(new_agent, batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis)
```

update\_critics 方法也有调用如下：

```
def update_critics(self, batch, pmap_axis=None):
    # 数据增强
    obs = self.data_augmentation_fn(obs_rng, batch["observations"])
    next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
    
    # 仅更新Critic网络
    new_agent, critic_infos = new_agent.update(
        batch,
        pmap_axis=pmap_axis,
        networks_to_update=frozenset({"critic"})  # 只更新Critic
    )
```

* 编码器选择（create\_drq）：它提供了灵活的配置：

  + small：简单的卷积层，适合简单任务。
  + resnet：标准的 ResNet-10，SERL 的主力方案。
  + resnet-pretrained：加载预训练权重（通常是 ImageNet 预训练），适合非常复杂的视觉场景。
  + 优秀选择较小模型：在高 UTD 场景下，计算机通常在带宽和前向传播受限。如果模型太大，GPU 每一秒处理不完那 20 次更新，整个训练过程就会从"20 分钟学会"变成"2 小时学会"。在机器人领域，实时性（Speed）往往比深度（Depth）更重要。
* 随机种子（随机裁剪偏移量）：在计算数据增强时，它对 observations 和 next\_observations 使用了两个不同的随机种子（obs\_rng 和 next\_obs\_rng）。

  + 用不同种子的好处：如果 s 往左移，s' 往右移，这会产生一种类似"抖动"的效果。这强迫网络明白：无论目标在图像的哪个角落，无论视角怎么晃动，动作与状态之间的物理逻辑是不变的。这极大地增强了策略的"空间不变性"。
  + 用同样种子的坏处：如果你对 s 和 s' 使用相同的偏移（比如都往左上角移 4 像素），同样种子会导致虽然位置变了，但 s 到 s' 的相对位置没变，机器人学到的是一种"死板的平移"。
* 共用视觉头（Shared Encoder）在 create\_drq 中，shared\_encoder 默认是 True。如果设为 False（Actor 和 Critic 各用各的视觉网络），其优势如下：

  + 特征一致性：Actor 想知道："杯子在哪里？"。Critic 想知道："手抓到杯子的价值是多少？"。它们关心的都是同样的物理特征。让它们共用一个 ResNet，就像是让两个学生共用一本最好的教科书。一个学生（Critic）学得快，他理解的知识点会通过梯度回传，帮助另一个学生（Actor）也快速进步。
  + 极速收敛：视觉特征的学习是 RL 中最慢的部分。由于 Critic 接收的是奖励信号（比较明确），它训练视觉头的速度通常比 Actor 快。搭便车效应：Actor 直接使用 Critic 已经练好的视觉特征，只需要学习如何映射到动作即可。这能节省 50% 以上的训练时间。
  + 节省显存：ResNet 的显存占用很大。在 UTD=20 时，显存消耗非常惊人。共用一个头可以腾出更多的空间给更大的 Batch Size。

## 0x04 VICE

vice.py 是 SERL 中一个非常高级的组件，它实现了 VICE（Variational Inverse Control with Events）算法。它的核心任务是：当机器人没有明确的奖励信号时，通过观察"成功图片"来给自己打分。这在真实的机器人部署中极其重要，因为你无法在真实世界里写出一个完美的数学公式来判断"杯子是否抓稳了"。

VICE 通过将策略访问过的所有状态以负标签的形式加入分类器的训练集，并在每次迭代后更新分类器，从而解决该问题。这样，强化学习过程类似于生成对抗网络（GAN），其中策略充当生成器，奖励分类器则作为判别器。

### 4.1 背景 & 原理

VICE 是 RLPD 的"心跳"。在 train\_rlpd.py 这种算法中，Demo 数据只是素材，而 VICE 的奖励才是灵魂。没了奖励，机器人就失去了进化的动力。

#### 4.1.1 动机：为什么要搞 VICE？

在仿真环境（如 MuJoCo）中，奖励函数很好写：奖励 = - 距离。

但在真实世界中：

* 你怎么知道机械臂末端离杯子还有几厘米？（除非你装了极其昂贵的外部捕捉系统）。
* 你怎么判断线缆是否进了槽位？

VICE 的思路：人类只需要给机器人看几张"任务成功"的照片（Goal Images），剩下的让机器自己去悟。

#### 4.1.2 变分推断（Variational Inference）

变分推断是一种"用简单模型逼近复杂真相"的方法。

它的核心逻辑是：

* 真相太复杂，无法直接计算：就像你（没有经过美术训练）无法直接画出某个美女的真容
* 构建一个简化模型：用简单的几何形状（椭圆、柳叶眉、杏仁眼）构建一个"标准模型"
* 调整模型参数：根据观察到的线索，不断调整模型的参数（脸型的长宽比、眉毛的弧度等）
* 优化目标：让简化模型"最好地解释观察到的线索"
* 结果：简化模型虽然不是真容，但已经非常接近真容

一句话总结：变分推断就是"用一个可调的简化模型，去逼近一个复杂但无法直接计算的真相"。

##### 贝叶斯推断的困境

在贝叶斯统计中，我们想计算后验分布：p(z|x) = p(x|z) \* p(z) / p(x)

其中：

* z 是隐变量（如女子的真容）
* x 是观测数据（如你看到的线索）
* p(x|z) 是似然（给定真容，看到线索的概率）
* p(z) 是先验（对真容的初始假设）
* p(x) 是证据（看到线索的总概率）

问题：p(x) 需要对所有可能的 z 积分，这在复杂模型中几乎不可能计算。

##### 变分推断的解决方案

既然无法直接计算 p(z|x)，那就用一个简单的 q(z) 来近似它。

* 目标：找到最好的 q(z)，使其尽可能接近 p(z|x)。
* ELBO = E\_q[log p(x,z)] - E\_q[log q(z)]
* 最大化 ELBO 等价于最小化 q(z) 和 p(z|x) 之间的 KL 散度。

#### 4.1.3 VICE 原理

VICE 是一种"从专家示范中学习评判标准"的方法。VICE 不是教你"怎么走路"，而是教你"什么算走到了目的地"。一旦你知道目的地在哪，你自然就知道该往哪走了。

它的核心逻辑是：

* 不直接学"怎么做"，而是学"什么算做好"，即，你不需要告诉机器人"怎么做"，也不需要给它一条完整的示范路径。
* 通过观察大量专家示范（成功案例）和非专家示范（失败案例），即，你只需要给它看一堆"成功长什么样子"的图片（goal state examples）。
* "事件"的定义来自你提供的成功状态样本，不需要完整示范轨迹。
* 用变分推断（variational inference）训练一个分类器，输出"当前状态是成功状态的概率"，作为reward。
* 然后用这个分类器来学"这个状态像不像成功"，把这个"像成功的概率"当作奖励信号，来训练强化学习。
* 目标不是"最大化累积分数"，而是"让某个事件（event）发生的概率最大"，发生关键时刻越多，reward 越高。

#### 4.1.4 对抗性进化：猫鼠游戏

VICE 把强化学习变成了一个"找游戏"。

1. 准备正样本：你手动把机器人摆到成功位置，拍 20 张照片。这些照片标签为 1。
2. 准备负样本：机器人自己在环境里瞎跑拍到的照片。这些照片标签为 0。
3. 训练判别器：训练一个二分类神经网络（Binary Classifier），输入一张图，输出"它有多像成功照片"。
4. 生成奖励：Reward = Classifier(当前图片)，机器人发现："只要我做出的动作让画面越来越像那 20 张照片，我的得分就越高。"

VICE 实际上是一个类似 GAN（生成对抗网络）的结构：

* 分类器（猫）：努力区分"真正的成功"和"机器人自以为是成功的动作"。
* 机器人（鼠）：努力做出让分类器认为是"成功"的动作。

随着训练的进行，分类器变得越来越挑剔，机器人也随之变得越来越精准。

如果 VICE 坏了：

* 在线试错 被 VICE 判为 0。
* 人类演示 也被 VICE 判为 0。
* 结果：Critic 只能接受着这个事实 —— "这个世界没有奖赏，做什么都是徒劳"。Q.网络随之萎缩到 0。
* 此时，Actor 唯一的指路明灯就只剩下那个额外的 BC Loss 了。所以，在 VICE 坏掉的情况下：
  + 如果没有 BC Loss，机器人会彻底变疯。
  + 如果有 BC Loss，机器人会退化成一个笨拙的模仿者，不再有进化的动力。

### 4.2 网络架构

VICE Agent 在标准 Actor-Critic 基础上增加了基于视觉的奖励分类器，实现了从人类演示到稀疏奖励的迁移学习。

| 组件 | 编码器 | 主干网络 | 输出 | 特殊设计 |
| --- | --- | --- | --- | --- |
| **Actor** | SmallEncoder/ResNet | MLP [256,256] | 动作分布 | tanh\_squash |
| **Critic** | 共享Actor编码器 | Ensemble MLP [256,256]×2 | Q值 | 双Critic |
| **Temperature** | 无 | Lagrange | 标量温度 | 自动调节 |
| **VICE Classifier** | 独立编码器 | MLP [256] | 二分类logit | 预训练+微调 |

核心组件架构如下：

**Actor 网络** ：

```
policy_def = Policy(
    encoder=encoders["actor"],              # 视觉编码器
    network=MLP(**policy_network_kwargs),   # [256, 256]
    action_dim=actions.shape[-1],
    tanh_squash_distribution=True,
    std_parameterization="uniform",
)
```

**Critic 网络**

```
critic_backbone = partial(MLP, **critic_network_kwargs)  # [256, 256]
critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
    name="critic_ensemble"  # 默认2个Critic
)
critic_def = partial(Critic, encoder=encoders["critic"], network=critic_backbone)
```

**Temperature 网络**

```
temperature_def = GeqLagrangeMultiplier(
    init_value=1.0,
    constraint_shape=(),
    constraint_type="geq",  # 大于等于约束
)
```

**VICE Reward Classifier**

```
vice_def = BinaryClassifier(
    pretrained_encoder=pretrained_encoder,  # 预训练ResNet
    encoder=vice_encoder_def,              # 独立的编码器
    network=MLP(**vice_network_kwargs),    # [256]
    enable_stacking=True,
)
```

另外，VICE使用独立的编码器。

```
# VICE使用独立的编码器
vice_encoders = {
    image_key: SmallEncoder(
        features=(32, 64, 128, 256),
        kernel_sizes=(3, 3, 3, 3),
        strides=(2, 2, 2, 2),
        padding="VALID",
        pool_method="avg",
        bottleneck_dim=256,
        spatial_block_size=8,
        name=f"vice_encoder_{image_key}",
    )
    for image_key in image_keys
}
```

### 4.3 奖励（reward）的计算

在 VICEAgent 中，奖励（reward）的计算依赖于一个学习到的二分类器（即 vice 网络），该网络用于区分专家/目标状态与智能体的观测。具体计算过程如下：

* 前向传播与 Sigmoid 激活： 将 next\_observations 输入 vice 网络（一个 BinaryClassifier），并通过 sigmoid 函数将输出映射为 0 到 1 之间的概率分数。此过程由 vice\_reward 方法实现：

  ```
  rews = nn.sigmoid(
      self.state.apply_fn(
          {"params": self.state.params},
          observation,
          name="vice",
          train=False,
      )
  )
  ```
* 二值化（阈值截断）： 在 Critic 更新阶段（如 update\_critics 和 update\_high\_utd 方法中），连续的概率分数会被转换为离散的二值奖励。若概率分数大于或等于 0.5，则奖励设为 1.0，否则为 0.0:

  ```
  rewards = (self.vice_reward(next_obs) >= 0.5) * 1.0
  ```
* 数据增强： 在计算奖励前，会先通过 data\_augmentation\_fn 对 next\_observations 进行数据增强，以提升奖励信号的鲁棒性。
* 总结：奖励本质上是一个二值指标（1.0 或 0.0），其取值取决于 vice 分类器是否将 next\_observation 判定为属于目标/专家状态分布（即 sigmoid 输出的概率 ≥0.5）。

### 4.4 训练逻辑

在 VICEAgent 中，BinaryClassifier（在代码中实例化为 vice 网络）的训练逻辑主要集中在 update\_vice 方法中。其训练过程使用了二元交叉熵（BCE）损失，并引入了 Mixup、标签平滑（Label Smoothing）和梯度惩罚（Gradient Penalty）等正则化技术来防止 GAN 模式崩溃。

具体训练步骤如下：

**数据准备与标签平滑（Label Smoothing）：**

* 假设 batch 的后半部分为目标图像（正样本，标签为 1），前半部分为普通观测（负样本，标签为 0）。
* 对图像进行数据增强（通过 data\_augmentation\_fn），并将原始图像与增强后的图像拼接。
* 应用标签平滑技术处理标签： y\_batch = y\_batch \* (1 - 0.2) + 0.5 \* 0.2，以缓解大 logits 带来的数值问题。

**特征编码：**

* 使用 encode\_images 方法将拼接后的所有图像通过 encoder 提取为特征向量（embeddings）。

**Mixup 正则化：**

* 在特征空间中对 embeddings 和标签进行 Mixup 操作（通过 mixup\_data\_rng），生成混合特征 mix\_encoded 和对应的混合标签。
* 定义 mixup\_loss\_fn，计算混合特征通过 create\_classifier 初始化网络权重，输入特征模态由 config.classifier\_keys 指定后的 BCE 损失：`bce_loss = lam_θ * bce_loss_a + (1 - lam_θ) * bce_loss_b`。

**梯度惩罚（Gradient Penalty）：**

* 在 Mixup 生成的特征之间进行随机插值，生成用于计算梯度惩罚的数据 gp\_encoded。
* 定义 gp\_loss\_fn，计算 BinaryClassifier 对 gp\_encoded 输出的梯度。
* 计算梯度惩罚项： grad\_penalty = mean((grad\_norms - 1) \*\* 2)。
* 最终的总损失为：bce\_loss + 10 \* grad\_penalty。

**参数更新：**

* 构建 loss\_fns 字典，将 gp\_loss\_fn 指定给 vice 网络，其他网络（actor, critic, temperature）的损失设为 0。
* 调用 new\_agent.state.apply\_loss\_fns 执行梯度计算，并使用对应的优化器更新 BinaryClassifier 的参数。

**总结：**

BinaryClassifier 是在特征编码空间中进行二分类训练的，并通过 Mixup 和梯度惩罚来平滑决策边界，从而提高 VICE 奖励信号的稳定性和鲁棒性。

### 4.5 流程图

在 SERL 的流程中：

* 你教机器人做动作时，你只是在提供动作序列。
* 当你把这组数据存入 demo\_buffer 时，它的 reward 字段通常不是由你手动输入的。

它是怎么被贴上标签的？在 train\_rlpd.py 中，当你采样出一个 Batch（包含 Demo）时，系统会运行以下逻辑：

1. 取图：取出 Demo 数据里的那一帧图片 s。
2. 过分类器：把图片 s 塞进 VICE 分类器。所有数据（无论谁做的）都必须经过 VICE 的安检。VICE 说它是成功，它才是成功。
3. 重新打分：分类器返回一个分数（例如 0.95）。
4. 填入 r：这个 0.95 就成了这一条 Demo 数据在当前时刻的奖励 r。

VICE Agent（自奖励分类器）流程图如下：

1. Mixup & Label Smoothing: 应对极小样本量（可能只有几十张成功图片）的过拟合问题。
2. Gradient Penalty (GP): 模仿 GAN 的技巧，确保奖励函数不仅准，而且梯度连续，利于 RL 优化。
3. Reset-Free Friendly: 允许机器人在没有外部传感器（如压力传感器、红外对管）的情况下，仅靠视觉确认任务是否完成。

![SERL-VICE Agent](images/img_005.png)

### 4.6 工程细节

#### 4.6.1 分类即奖励（vice\_reward）

在 sac.py 或 rlpd.py 运行中，VICE 的核心合作伙伴是 Critic。合作方式如下：

* 环境给出一张图 s'。
* VICE 跳出来说："根据我的审美，这张图值 0.8 分"。
* Critic 拿着这个 0.8 分，代入贝尔曼公式：Q = r + γ Q\_next。
* Actor 看到 Q 值很高，就知道刚才那个动作是对的，以后多做。

VICE 本质上就是一个"奖励函数生成网络"，它判断的是现在这一瞬间好不好，将"判断任务是否成功"转换成了一个二分类问题。

* Logic：如果当前画面 s' 在分类器眼里很像成功画面，Sigmoid 输出接近 1.0，机器人就拿到了高分。

```
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        rewards = (self.vice_reward(next_obs) >= 0.5) * 1.0
        batch = batch.copy(add_or_replace={"rewards": rewards})
```

* 解读：VICE 会在每次更新之前，实时改写 Batch 里的奖励字段。
* 它把连续的概率值转成了 0/1 的二值奖励。这意味着只要机器人把物体放到了"看起来成功"的位置，它就立刻拿到了 1 分。

类比：

* VICE 像是你体内的"多巴胺"：当你看到巧克力（成功图片），大脑立刻分泌多巴胺，告诉你"这个瞬间真棒"。
* Critic 像是你的"理财顾问"：他根据你现在的多巴胺水平，预测你这辈子最后能不能幸福。

#### 4.6.2 策略 & 效果

VICE 的优越性：作为深度网络，它学到的是"特征空间的相似性（比如插头和插座的咬合关系），这种特征具有极强的泛化性和抗噪声能力。

为什么像素差异不行？

* 精准！物理上的"像素差异"和语义上的"任务成功"完全是两码事。
* 例子：插头离插座 1 厘米，和插头插进偏了 1 厘米，像素差异可能很小，但物理状态（成功 vs 失败）截然不同。

SERL 的策略：通常会先用之前提到的 BC（行为克隆）让机器人做出一些看起来像样的动作，从而给 VICE 提供高质量的"正样本"和"对比样本"。只有评价体系稳了，RL 的高速更新（UTD=20）才有意义。

#### 4.6.3 工程上的"防崩盘"设计

直接训练分类器在 RL 中很容易崩。比如：机器人发现只要把手在摄像头前晃一下，挡住背景，画面特征就很像成功照片了（这叫 Reward Hacking）。

update\_vice 函数用了几个大招来对付它：

* Mixup（数据混合）：
  + 将图片按比例混合。这防止了分类器只记住几张特定的死板图片。比如，它不只分两类看"成功"和"失败"，还把两张图按比例重叠。
  + 用意：让分类器的判断标准不再是非黑即白，而是产生一个平滑的过渡区域，引导机器人慢慢靠近目标。
* Label Smoothing（标签平滑）：不给 1.0 分，只给 0.9 分。
  + y\_batch = y\_batch \* (1 - 0.2) + 0.5 \* 0.2。把 1 变成 0.9，把 0 变成 0.1。这让奖励信号不会太极端。
  + 用意：防止分类器产生极大的梯度，把机器人"吓跑"。
* Gradient Penalty（梯度惩罚）：强制要求分类器的输出对输入的变化是平滑的。
  + 惩罚权重的剧烈波动。这让奖励函数的等高线变得平滑，Actor 顺着梯度往上爬时不会"脚滑"。
  + 用意：如果奖励函数像悬崖一样陡峭，Actor 根本找不到往上爬的方向。它需要的是一个温和的"斜坡"。

## 0xFF 参考
