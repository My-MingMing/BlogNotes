# 探秘Transformer系列之（16）--- 资源占用

目录

* [探秘Transformer系列之（16）--- 资源占用](#探秘transformer系列之16----资源占用)
  + [文章总表](#文章总表)
  + [0x00 概述](#0x00-概述)
  + [0x01 背景知识](#0x01-背景知识)
    - [1.1 数据类型](#11-数据类型)
    - [1.2 进制&换算](#12-进制换算)
      * [数字进制](#数字进制)
      * [存储度量](#存储度量)
      * [换算](#换算)
    - [1.3 参数显存占用](#13-参数显存占用)
      * [有参数的层](#有参数的层)
      * [无参数的层](#无参数的层)
      * [所需资源](#所需资源)
    - [1.4 计算量](#14-计算量)
  + [0x02 Transformer参数量](#0x02-transformer参数量)
    - [2.1 术语](#21-术语)
    - [2.2 embedding层](#22-embedding层)
    - [2.3 Transformer层](#23-transformer层)
      * [MHA](#mha)
      * [FFN](#ffn)
      * [LayerNorm](#layernorm)
      * [小结](#小结)
    - [2.4 lm\_head](#24-lm_head)
    - [2.5 最终参数量](#25-最终参数量)
    - [2.6 LLaMA3](#26-llama3)
      * [SwiGLU](#swiglu)
      * [GQA](#gqa)
  + [0x03 Transformer显存占用](#0x03-transformer显存占用)
    - [3.1 训练](#31-训练)
    - [3.2 推理](#32-推理)
    - [3.3 激活](#33-激活)
      * [架构](#架构)
      * [术语说明](#术语说明)
      * [数据量](#数据量)
        + [注意力块](#注意力块)
        + [MLP](#mlp)
        + [LayerNorm](#layernorm-1)
        + [总结](#总结)
        + [并行](#并行)
  + [0x04 Transformer计算量](#0x04-transformer计算量)
    - [4.1 矩阵乘法](#41-矩阵乘法)
    - [4.2 前向传播计算量](#42-前向传播计算量)
      * [Embedding](#embedding)
      * [MHA](#mha-1)
        + [计算Q、K、V](#计算qkv)
        + [QK^T](#qkt)
        + [乘以V](#乘以v)
        + [线性映射](#线性映射)
      * [MLP](#mlp-1)
      * [LayerNorm](#layernorm-2)
      * [单层layer](#单层layer)
    - [4.3 综合思考](#43-综合思考)
      * [反向传播](#反向传播)
        + [单层](#单层)
        + [logits](#logits)
      * [总体计算量](#总体计算量)
    - [4.4 计算特点](#44-计算特点)
      * [与参数量的关系](#与参数量的关系)
        + [单次推理](#单次推理)
        + [单次训练](#单次训练)
      * [带宽受限](#带宽受限)
        + [注意力计算](#注意力计算)
        + [FFN计算](#ffn计算)
      * [KV Cache的影响](#kv-cache的影响)
        + [prefill](#prefill)
        + [decode](#decode)
        + [总体](#总体)
        + [kv cache 节省了多少计算量](#kv-cache-节省了多少计算量)
  + [0x05 优化方向](#0x05-优化方向)
    - [5.1 基于注意力机制来修改外推技术](#51-基于注意力机制来修改外推技术)
    - [5.2 基于Memory机制外推技术](#52-基于memory机制外推技术)
  + [0xFF 参考](#0xff-参考)

## 文章总表

全部文章列表在这里 [探秘Transformer系列之文章列表](https://www.cnblogs.com/rossiXYZ/p/18785601)，后续每发一篇文章，会修改这里。

## 0x00 概述

对于标准 Transformer 模型，不管是 Encoder Only 的 Bert 系列模型，还是 Decoder Only 的 GPT 系列模型，同配置下参数量和计算量都是类似的。其中的一个关键点是：标准 Transformer block（层）输入、输出以及中间 Hidden Dim 保持不变，始终是 Token Embedding 的 Hidden Dim，所有的 Transformer Block 都非常规整。

如下图所示，Encoder主要参数都来自几个矩阵乘的 Weight 矩阵，其中 d 表示 Token Embedding 的 Hidden Dim，l 表示 Token 数，h 表示 MHA 中的 Head 个数，\(d\_{FFN}\) 表示 FFN 层中间升维后的 Dim。其主要几个模块的参数量如下。

* MHA：\(W\_Q，W\_K，W\_V\) 的大小都是 d x d。当然这里也可以从 h 个 Head 的角度去看，则每个 Head 的 \(W\_Q，W\_K，W\_V\) 为 d x d/h。在 MHA 的最后还有一个矩阵乘操作，对应的 \(W\_{out}\) 维度依然为 d x d。所以MHA处权重矩阵的参数量是 \(3d \times d + d \times d\)。
* FFN：标准 Transformer 的 FFN 中有两个 Linear 层（先升维再降维），对应权重矩阵 \(W\_1\) 和$ W\_2$ 的大小都是 \(d\_{FFN}\) x d，并且标准的 \(d\_{FFN}\) 为 4d，也就是说 FFN 处两个权重矩阵的参数量为 8d x d。

![](images/img_001.jpg)

综上，在标准的 Transformer 模型或者 LLaMA 系列（MHA）中，如果忽略词表、Embedding、LayerNorm 等参数后，总参数量为（所有 Transformer Block）： \(N = n\_{layer} \times (n\_{mha}+ n\_{ffn}) = n\_{layer} \times (3d \times d + d \times d + 8d \times d) = 12 \times n\_{layer} \times d \times d\)

注意：本章参考了多篇论文，其中对术语的定义各不相同，因为模型结构也不同，所以计算结果与其它资料可能也有差异。

## 0x01 背景知识

### 1.1 数据类型

深度学习中用的数值类型命名规范一般为TypeNum，比如Int64、Float32、Double64。

* Type：有Int，Float，Double等。
* Num: 一般是 8，16，32，64，128，表示该类型所占据的比特数目。

常用的数值类型如下图所示。

| 类型 | 大小（字节数） |
| --- | --- |
| int4 | 0.5 |
| int8 | 1 |
| int16 | 2 |
| int32 | 4 |
| int64 | 8 |
| float32 | 4 |
| float16 | 2 |
|  |  |

### 1.2 进制&换算

我们先抛出一个问题:1B参数对应多少G显存？B和G都代表十亿（1000M或1024M），但这是两个不同的度量维度。

#### 数字进制

B是英美常用的进制单位，比如：

* 1K = 1000，一千；
* 1M = 1000 K，百万；
* 1B = 1000 M，十亿；

可以看出来，这个进制单位以 1000 为进制。以 Qwen-7B 为例，7B 的意思就是 这个 LLM 的 模型参数有 70亿 个 参数。

#### 存储度量

G是计算机内存/磁盘存储的度量，基本单位是字节，进制是 1024。单位依次是：KB / MB / GB / TB。平时说显存有多少G/M是说有多少G/M个字节（byte），1个字节=8比特（bit）。举例来说：有一个1000x1000的 矩阵，float32，那么占用的显存差不多就是1000x1000x4 Byte = 4MB。

#### 换算

可以看出来，\(1B=10^9 byte \approx 1GB\)，1B和1G的大小基本一致，所以我们记作B和G相等。但是，1B模型参数对应多少G内存和参数的精度有关。如果是全精度训练（fp32），一个参数对应32比特，也就是4个字节，参数换算到显存的时候要乘4，也就是1B模型参数对应4G显存。如果是fp16或者bf16就是乘2，1B模型参数对应2G显存。具体如下表所示。

| 数据类型 | 每1B参数需要占用内存 |
| --- | --- |
| fp32 | 4G |
| fp16/bf16 | 2G |
| int8 | 1G |
| int4 | 0.5G |

### 1.3 参数显存占用

有参数的模块才会占用显存。这部份的显存占用和输入无关，模型加载完成之后就会占用。一般的卷积层都会占用显存，而我们经常使用的激活层Relu没有参数，所以不会占用缓存。

#### 有参数的层

常见的有参数的模块主要包括：

* 卷积层，通常的conv2d。
* 全连接层，也就是Linear层。
* BatchNorm层。
* Embedding层。

#### 无参数的层

常见的无参数的模块主要包括：

* 多数的激活层，比如Sigmoid/ReLU。
* 池化层。
* Dropout。

#### 所需资源

我们可以用如下公式来计算神经网络的显存占用：显存占用 = 模型显存占用 + 输入输出相关的显存

模型显存占用是模型中与输入无关的显存占用，主要包括：

* 模型权重参数。
* 梯度（一般是参数量的1倍）。
* 优化器的动量（和具体优化器密切相关，比如普通SGD没有动量，momentum-SGD动量与梯度一样，Adam优化器动量数量是梯度的两倍）。

输入输出相关的显存占用主要如下：

* batch\_size × 每个样本的显存占用。
* 每一层的feature map，需要保存激活来进行反向传播。

因为 反向传播 / Adam-优化 / Transformer架构 等因素，一般来说，训练需要的显存，是 同样规模推理 的 3-4倍。

### 1.4 计算量

上文提到Transformer的计算复杂度是 $O(dN^2) $。大 O 表示法关注的是计算量级与输入规模之间的关系，并不是具体的计算量。具体计算量通常用FLOPs体现。这里简单列举一些比较常见的单位：

* FLOPs ：floating point of operations的缩写，是浮点运算次数，一般特指乘加运算次数，理解为计算量，可以用来衡量算法/模型复杂度。
* 一个GFLOPS（gigaFLOPS）= 每秒十亿（=10^9）次的浮点运算
* 一个TFLOPS（teraFLOPS） = 每秒一万亿（=10^12）次的浮点运算

## 0x02 Transformer参数量

以Decoder only模型为例，其主要包括 3 个部分：embedding，decoder，head。最主要部分是decoder，其由若干个decoder-layer组成，每个decoder-layer又分为两部分：MHA和FFN。我们接下来逐一看看这些模块的参数量。

### 2.1 术语

我们先给出本节使用的术语。

| Symbol | Meaning |
| --- | --- |
| \(d\) | 模型的词嵌入大小（The model size / hidden state dimension / positional encoding size） |
| \(h\) | 注意力头个数 |
| \(s\) | 文本总长度（prompt+解码器输出） |
| \(b\) | 数据batch size（批大小） |
| \(l\) | Transformer层数 |
| \(v\) | 词表大小 |

### 2.2 embedding层

embedding层的输入形状是[b,s,v]，输出形状是[b,s,d]，参数量为\(v \times d\)。如果采用可训练式的位置编码，会有一些可训练模型参数，但是其数量比较少。如果采用相对位置编码，例如RoPE和ALiBi，则不包含可训练的模型参数。因此我们忽略位置编码的参数。

### 2.3 Transformer层

Transformer模型由 l 个相同的层组成，每个层主要分为两部分：MHA和FFN。因为多头只是逻辑上切分，物理上没有增加模块，因此后续讨论中省略多头（某些论文中如果讨论多头相关，我们会以论文为准），而又因为Decoder only模型使用的是自注意力，因此接下来我们认为 Q、K、V、O的维度相等。

#### MHA

MHA中包含四个权重矩阵\(W^Q,W^K,W^V,W^O\)以及偏置（某些模型可能没有偏置）。4个权重矩阵的形状为 [\(d\),\(d\)]，4个偏置的形状为 [\(d\)]，其中 \(d = h \times d\_{head}\)。因此，多头注意力层参数量为：\(4\times (d \times d + d) = 4d^2 + 4d\)。

#### FFN

FFN包括两个线性层。

* 第一层将原有的维度映射到4倍原维度大小，即从\(d\)映射到4\(d\)。权重矩阵形状是[d, 4d]，偏置形状是[4d]。参数量为：\(d\times 4d + 4d\)
* 第二层从4倍维度降维回原始维度。即从4\(d\)映射到\(d\)。权重矩阵形状是[4d, d]，偏置形状是[d]。参数量为： \(4d\times d + d\)

最终FFN的参数是：\(8d^2 + 5d\)。

#### LayerNorm

对于Layer Norm来说，其缩放参数 \(\gamma\)与平移参数 \(beta\) 维度都为 \(d\)，因此参数量是 \(2 \times d\)。因为MHA和FFN都有LayerNorm，因此总参数量是\(4 \times d\)。

#### 小结

综上，单个Transformer层的参数量是：\(12d^2 + 13d\)。

### 2.4 lm\_head

lm\_head是自然语言处理模型中的一个组件，主要作用是将模型的输出（通常是经过Transformer编码器处理后的隐藏状态）转换成预测下一个词的概率分布。

Head与embedding的参数量相同。如果是tied embedding（即，head权重矩阵与词嵌入矩阵是参数共享的），则两者公用一个参数。

### 2.5 最终参数量

最终，l 层transformer模型的可训练模型参数量为\(l(12d^2 + 13d) + 2vd\) 。当d较大时，可以忽略一次项，模型参数量近似为\(12ld^2\) 。

![](images/img_002.jpg)

### 2.6 LLaMA3

我们再用LLaMA3来看看在工业界落地中的一些特殊之处。

#### SwiGLU

LLaMA 等模型在 FFN 中会使用 SwiGLU 激活，这也就导致其会额外多了一个权重矩阵。LLaMA论文中提到，使用 SwiGLU 后将 dFFN 从 4d 降低到了 8d/3。这样 3 个权重矩阵的参数量还是 8d，总的参数量依然可以使用 \(12 \times n\_{layer}\times d\times d\)来 预估。

#### GQA

前面公式对应的是 MHA（Multi Head Attention），这也是 LLaMA-1 系列模型的标准实现。不过，LLaMA-2 的 30B 和 70B 模型以及 LLaMA-3 的全部模型都开始使用 GQA（Grouped Query Attention）。使用 GQA 时，多个 注意力头会共享一个 Key 和 Value，此时\(W^K,W^V\)的大小会变为 d x d/g，其中 g 表示每 g 个 Head 共享相同的 Key 和 Value。LLaMA 2论文提到，为了保持使用 GQA 和使用 MHA 的总参数量保持不变，对于 GQA 模型，LLaMA 2会将 FFN Dim 维度乘以 1.3。

经过上述调整之后，LLaMA 3 不再是标准的 Transformer Block，此时使用 \(N=12d^2\) 来预估参数量已经不太准确。但依旧可以将其按照（\(W^Q,W^O\)）（\(W^K,W^V\)），$W\_{FFN} $和 \(W\_{emb}\) 4 个部分来统计。比如，对于 LLaMA 3 模型，我们可以按照下述方式估计其参数量：\(N = n\_{layer} \times (2d^2 + 2d \times d \times kv/h + 3d \times d\_{FFN})+2 \times Vocab \times d\)。

## 0x03 Transformer显存占用

### 3.1 训练

在训练神经网络的过程中，占用显存的大头主要分为四部分：模型参数、前向计算过程中产生的中间激活、后向传播计算得到的梯度、优化器状态。后面几个的数量可能比模型参数更大，因此对模型内存的需求量也更大。

训练大模型时经常采用AdamW优化器，并用混合精度训练来加速训练，我们基于这个前提分析显存占用。在一次训练迭代中，每个可训练模型参数需要保存这个参数本身、参数对应的梯度以及优化器对这个参数的两个状态（Adam中的一阶动量和二阶动量）。设模型参数量为 Φ ，那么梯度的元素数量为 Φ ，AdamW优化器的元素数量为 2Φ 。在混合精度训练中，会使用半精度来进行前向与反向传播计算，优化器更新模型参数时会使用单精度进行状态、梯度以及参数的更新。所以一个参数在训练时占用的空间为正向传播时使用半精度和反向传播时使用单精度所占用的空间之和。因此，使用AdamW优化器和混合精度训练来训练时候，针对每个可训练模型参数，训练阶段会占用 (2+4)+(2+4)+(4+4)=20bytes 。参数量为 Φ 的大模型，模型参数、梯度和优化器状态占用的显存大小为 20Φ bytes 。

![](images/img_003.jpg)

模型参数、梯度与优化器状态的空间占用已经计算完了，接下来就是在前向传播时的中间激活部分的空间占用。我们将在后续小节进行分析。

模型的训练包含 Forward 和 Backward 过程。Backward 过程实际上包含两部分，一部分是对输入的梯度（链式法则），一部分是对权重的梯度。其实这两部分主要的计算量都是矩阵乘法，并且大小与 Forward中的大小一致，因此往往会直接近似 Backward 的计算量为 Forward 的 2 倍。

### 3.2 推理

推理阶段通常比训练阶段要求更低的显存，因为不涉及梯度计算和参数更新等大量计算。少了梯度、优化器状态和中间激活，模型推理阶段占用的显存要远小于训练阶段。

如果使用KV cache来加速推理过程，KV cache也需要占用显存，KV cache占用的显存下文会详细介绍，此处忽略。此外，输入数据也需要放到GPU上，还有一些中间结果（推理过程中的中间结果用完会尽快释放掉），不过这部分占用的显存是很小的，也可以忽略。

最终，推理阶段的主要显存占用为模型的参数，模型参数内存 = n × p。n是模型参数总量，p是每个参数占用的字节数。如果使用半精度进行推理的话，一个参数占用2bytes空间，那么模型在推理时的显存占用约为：

\[mem\_{inference} = 2 \times n\_{params}
\]

以下是计算模型推理时所需显存的一些关键因素：

* 模型结构： 模型的结构包括层数、每层的神经元数量、卷积核大小等。较深的模型通常需要更多的显存，因为每一层都会产生中间计算结果。
* 输入数据： 推理时所需的显存与输入数据的尺寸有关。更大尺寸的输入数据会占用更多的显存。
* 批处理大小 BatchSize： 批处理大小是指一次推理中处理的样本数量。较大的批处理大小可能会增加显存使用，因为需要同时存储多个样本的计算结果。
* 数据类型： 使用的数据类型（如单精度浮点数、半精度浮点数）也会影响显存需求。较低精度的数据类型通常会减少显存需求。
* 中间计算： 在模型的推理过程中，可能会产生一些中间计算结果，这些中间结果也会占用一定的显存。

### 3.3 激活

训练中的激活（activations）指的是：前向传播过程中计算得到的，并在反向传播过程中需要用到的所有张量。这里的激活不包含模型参数和优化器状态，但包含了dropout操作需要用到的mask矩阵。

在一次训练迭代中，模型参数（或梯度）占用的显存大小只与模型参数量和参数数据类型有关，与输入数据的大小是没有关系的。优化器状态占用的显存大小也是一样，与优化器类型有关，与模型参数量有关，但与输入数据的大小无关。而中间激活值与输入数据的大小（批次大小 b 和序列长度 s ）是成正相关的，随着批次大小 b 和序列长度 s 的增大，中间激活占用的显存会同步增大。当我们训练神经网络遇到显存不足OOM（Out Of Memory）问题时，通常会尝试减小批次大小来避免显存不足的问题，这种方式减少的其实是中间激活占用的显存，而不是模型参数、梯度和优化器的显存。

我们接下来以论文“Reducing Activation Recomputation in Large Transformer Models”中的Megatron为例，分步来计算一下中间激活的显存占用。

#### 架构

下图就是Megatron的架构。

![](images/img_004.jpg)

其代码如下所示。其中指定了core\_attention就是submodules.core\_attention，linear\_proj就是submodules.linear\_proj。

```
class Attention(MegatronModule, ABC):
    """Attention layer abstract class.
    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules, CrossAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )
        
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
            )
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out) # 这里是线性层

        return output, bias
```

最终注意力代码是：

```
class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use
        # simple strides to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout ----------------- 在这里有softmax的dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value.size(1), value.size(2), query.size(0), value.size(3))

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context
```

#### 术语说明

我们首先看看论文中的术语。

* a是 transformer 模型中注意力头 (attention heads) 的个数。
* b为每个GPU的batch size；
* h是每个 transformer 层的隐含维度
* L为Transformer的层数；
* p为流水线并行的并行机器数；
* s为句子的长度，即序列中词元的个数
* t为张量并行的并行机器数；
* v为词典的大小；

我们假设激活数据类型为 fp16。

#### 数据量

每个Transformer层由一个注意力和一个MLP构成，中间还有两个LayerNorm。下面，我们来推导存储每个元素的激活所需的内存。在下面的分析中需要注意几点：

* 单位是bytes，而不是元素个数。
* 大模型在训练过程中通常采用混合精度训练，因此，在分析中间激活的显存占用时，我们假设中间激活值是以float16或bfloat16数据格式来保存的，每个元素占了2个bytes。唯一例外的是，dropout操作的mask矩阵，每个元素只占1个bytes。
* 在分析中间激活的显存占用时，只考虑激活占用显存的大头，忽略掉一些小的buffers。比如，对于layer normalization，计算梯度时需要用到层的输入、输入的均值 和方差 。输入包含了 bsℎ 个元素，而输入的均值和方差分别包含了 bs 个元素。由于 ℎ 通常是比较大的（千数量级），有 bsℎ≫bs 。因此，对于layer normalization，中间激活近似估计为 bsℎ ，而不是 bsℎ+2bs 。

##### 注意力块

注意力块的激活如下。

| 保存内容 | 操作 | 激活大小 | 所属模块 | 保存原因 |
| --- | --- | --- | --- | --- |
| X | Query (Q), Key (K), Value (V) 相关的矩阵乘法 | 2bsh | self attention | 保存Q/K/V共同的输入X |
| Q、K | \(QK^T\) 矩阵乘法 | 4bsh | self attention | 保存 \(QK^T\) 矩阵乘法的输入 |
| \(QK^T\) | Softmax | \(2 bas^2\) | self attention | 保存Softmax 的输入，形状是 [b, a, s, s] |
| Mask | Softmax dropout | \(bas^2\) | self attention | 保存Softmax dropout 的mask，形状和\(QK^T\)相同，一个byte即可 |
| V | 注意力计算 | 2bsh | self attention | 保存\(softmax(\frac{QK^T}{\sqrt d})V\)的输入V |
| Score | 注意力计算 | \(2 bas^2\) | self attention | 保存\(softmax(\frac{QK^T}{\sqrt d})V\)的输入\(softmax(\frac{QK^T}{\sqrt d})\) |
| Linear | 计算输出映射 | 2bsh | linear projection | 输入映射需要保存其输入 |
| Mask | attention dropout | bsh | attention dropout | 24内dropout需要保存mask矩阵，一个byte即可 |
| 总计 |  | \(11bsh + 5bas^2\) |  |  |

我们回顾一下MHA的计算逻辑如下：

\[MultiHead(Q,K,V)=Concat(head\_1,head\_2,...,head\_{n\_{heads}})W\_O \\where\ head\_i = Attention(QW^Q\_i, KW^K\_i, VW^V\_i) \\=softmax(\frac{QW^Q\_i(KW\_i^K)^T}{\sqrt d\_{head}}) VW^V\_i
\]

上述表格中的各个计算解释如下。

* 输入X。X被用来计算Q、K、V。X的形状是[b,s,h]，元素个数是bsh，FP16占据两个byte，所以显存为2bsh。
* 中间激活 Q、K。这两者被用来计算\(QK^T\)。Q、K的形状都是[b,s,h]，元素类型是FP16，两者占据显存大小是4bsh。
* 中间激活\(QK^T\)。\(QK^T\)是softmax的输入，元素类型是FP16，占据显存大小是\(2bs^2a\)。a是注意力头数目。

  Q的形状是[b,a,s,h/a]，\(K^T\)形状是[b,a,h/a,s]。\(QK^T\)形状是[b,a,s,s]。计算公式如下：\(score=softmax(QK^T/\sqrt d\_k)\)
* dropout用到的mask矩阵。softmax操作完成之后，会进行dropout操作。需要保存一个mask矩阵，mask矩阵的形状与\(QK^T\)相同，类型是int，占据显存是\(bs^2a\)。
* score权重矩阵和V。这两者被用来计算Z。

  + softmax和dropout结束之后，得到了score权重矩阵，大小是2\(bs^2a\)。
  + V的形状都是[b,s,h]，元素类型是FP16，占据显存大小是2bsh。
* 计算输出映射以及一个dropout操作。输入映射需要保存其输入，大小为 2bsh ；dropout需要保存mask矩阵，大小为 bsh 。二者占用显存大小合计为 3bsh。

因此，将上述中间激活相加得到self-attention块的中间激活占用显存大小为 \(11bsh + 5bas^2\)

##### MLP

FFN的两个线性层以2*sbh*和8*sbh*的大小存储它们的输入。GeLU非线性还需要其大小为8*sbh*的输入用于反向传播。最后，dropout将其掩码存储为*sbh*大小。总的来说，MLP块需要19*sbh*字节的存储空间。

| 模块 | 动作 | 激活大小 |
| --- | --- | --- |
| linear 1 | 第一个线性层需要保存其输入 | 2 bsh |
| GeLU | 激活函数需要保存其输入 | 8 bsh |
| linear 2 | 第二个线性层需要保存其输入 | 8 bsh |
| dropout | 最后有一个dropout操作，需要保存mask矩阵 | bsh |
| 总计 |  | 19*sbh* |

我们回顾一下MHA的计算逻辑如下：

\[FFN(x) = f\_{gelu}(xW\_1+b\_1)W\_2 + b\_2
\]

上述的各个计算如下。

* 第一个线性层需要保存其输入，占用显存大小为 2bsh 。
* 激活函数需要保存其输入，占用显存大小为 8bsh 。
* 第二个线性层需要保存其输入，占用显存大小为 8bsh。
* 最后有一个dropout操作，需要保存mask矩阵，占用显存大小为bsh 。

因此，对于MLP块，需要保存的中间激活值为 19bsh 。

##### LayerNorm

另外，self-attention块和MLP块分别对应了一个layer normalization。每个layer norm需要保存其输入，大小为 2*sbh*。2个layer norm需要保存的中间激活为 4*sbh*。

##### 总结

综上，每个transformer层需要保存的中间激活占用显存大小为\(34bsh + 5bas^2\)。对于 l 层transformer模型，还有embedding层、最后的LayerNorm和输出层。当隐藏维度 ℎ 比较大，层数l 较深时，这部分的中间激活是很少的，可以忽略。因此，对于 l 层transformer模型，中间激活占用的显存大小可以近似为 \((34bsh + 5bas^2)\times l\)。

作为对比，下图是哈佛代码中解码器对应的激活情况，里面有各个张量的形状。

![](images/img_005.jpg)

有研究指出，`13B` 的 `LLM` 推理时，每个 `token` 大约消耗 `1MB` 的显存。

另外，对于计算量和显存量，我们也很容易见到不同的计算结果，这基本是因为计算原则不同，比如：梯度可能是FP16存储，参数可能是FP32存储，是否采用重计算等等。

##### 并行

实际工作中，LLM总是以各种并行策略进行训练或者推理，激活又各不相同。下图是各种并行策略下，每个Transfromer层的激活大小（bytes）。

![](images/img_006.jpg)

我们再来看看并行策略下，对于 l 层transformer模型，embedding层、最后的LayerNorm和输出层所输出的激活。

* 位置和单词嵌入不需要为反向传播存储任何大量的激活。但是dropout需要存储。嵌入层中的dropout也会沿着序列维度进行并行（sequence parallelism）。因此，它的存储将占据sbhp/t大小。请注意，系数p是因为流水线并行中，我们需要存储p个microbatches（微批次）。
* 输出层之前的Layer Norm也使用序列并行（sequence parallelism），因此需要2sbh/t存储。输出层会投影到词汇表维度，这需要存储大小为2sbh/t的输入。最后，交叉熵损失（cross entropy loss）需要存储以32位浮点进行计算的logit，因此需要4sbv/t的存储空间。请注意，由于我们只考虑流水线第一阶段的激活，因此上述激活，即总共4sbh/t(1+v/h)，仅在没有流水线并行（p=1）的情况下才会考虑在内。
* 输入嵌入、最后一个LayerNorm和输出层而产生的总共额外内存为：![](images/img_007.jpg)

## 0x04 Transformer计算量

广义上，当处理一个 token 时，模型执行两种类型的操作：注意力计算和矩阵-向量乘法。

* MHA（红框）：\(W\_Q\)，\(W\_K\)，\(W\_V\) 对应的计算量都为 2 x (d x d x l)，其中 2 表示一个乘法和一个加法。
* MHA（蓝框）：\(W\_{out}\) 对应的计算量为 2 x (d x d x l)。
* MHA Attention（绿色圆角方块）：计算量是2 x (l x d/h x l + l x d/h x l) x h = 4 x d x l x l。如果是 Decoder（LLM），由于 Causal Mask 的存在，此处的计算量应该减半，也就是 2 x d x l x l。
* FFN（绿框）：W1 和 W2 对应的计算量为 $2 \times (d\_{FFN} \times d \times l) $和 \(2\times (d \times \_{FFN} \times l)\)。LLaMA 的 SwiGLU 类似。

![](images/img_008.jpg)

我们后续也按照megatron论文的术语进行分析，忽略多头，即头数为1。

### 4.1 矩阵乘法

在decode阶段，则主要是矩阵-向量乘法。一个大矩阵乘以一个向量，得到另一个向量。

因此我们首先看看矩阵乘法的计算特点。人们定义算术强度（Arithmetic Intensity）为FLOP : I/O。当将一个\(N\times M\)矩阵与一个\(M\times P\)矩阵相乘以产生一个\(N\times P\)矩阵时，矩阵-向量乘法对每个矩阵元素执行一次乘加运算。FLOP（浮点操作，即计算量）为\(2M\times P \times N\)，I/O（从GPU内存传输到GPU寄存器的数据传输）计数为\(M\times N + M \times P + N \times P\) 。

### 4.2 前向传播计算量

#### Embedding

Embedding操作的输入是[b,s]。在实际计算的矩阵-向量乘法中，embedding操作并不会使用这整个embedding大矩阵，每个 token 只读取这个矩阵中的一行，就是查表操作。最终输出张量变成[b,s,h]。因此计算量相对很小，后面我们将忽略这部分。

#### MHA

在标准的Transformer计算中，假设\(Q,K,V \in R^{s\times h}\)，则计算如下（省略了\(\sqrt h\)）。N是序列长度，h是维度。

* 获取注意力分数 ：$ S = QK^T \in R^{s \times s}$。对每个 query 向量，都计算它与所有位置的 key 向量之间的点积。
* 获取注意力权重：$ P = softmax(S) \in R^{s \times s}$。即归一化得到的一组标量。
* 计算最终输出：\(O = PV \in R^{s \times h}\)。使用注意力权重，对所有之前的 value 向量进行加权求和来计算一个向量o。

因此我们可以知道，计算S和O是主要的部分。

##### 计算Q、K、V

单个矩阵乘法是：[b, s, h] x [h, h] 得到 [b, s, h]，因此其计算量是\(2bsh^2\)。三个矩阵的计算量是 \(3 \times 2 bsh^2 = 6 bsh^2\)

##### QK^T

在这个阶段，针对每个query元素，注意力计算会对每个键元素执行一次乘加操作以计算点积。总体操作为：[b, s, h] x [b, h, s] = [b, s, s] ，其计算量是：\(2bs^2h\)

softmax 函数不会改变输入矩阵的维度，即 [𝑠,𝑠]→[*s*,*s*]，native softmax 的 `FLOPs` 为 (4/5)*sh*。因为比较小，所以可以忽略。缩放 \(\sqrt d\) 是逐元素操作，也可以忽略。

##### 乘以V

乘以V（attention over values）阶段会对每个值元素执行一次乘加操作以计算加权和。总体操作为： [b, s, s] x [b, s, h] = [b, s, h]，计算量是：\(2bs^2h\)。

##### 线性映射

线性映射（post-attention linear projection）这一步是与\(W^O\)的多头融合，矩阵乘法的输入和输出形状为 [b,s,ℎ]×[ℎ,ℎ]→[b,s,ℎ] 。计算量为 \(2bsℎ^2\) 。

#### MLP

这一步涉及两个操作。

* 第一个线性层，矩阵乘法的输入和输出形状为 [b,s,ℎ]×[ℎ,4ℎ]→[b,s,4ℎ] 。计算量为 \(8bsℎ^2\) 。
* 第二个线性层，矩阵乘法的输入和输出形状为 [b,s,4ℎ]×[4ℎ,ℎ]→[b,s,ℎ] 。计算量为 \(8bsℎ^2\) 。

#### LayerNorm

`LayerNorm` 操作是逐元素进行的，因此不存在通用的公式来。`LayerNorm` 层的两个权重都是一个长度为 ℎ 的向量，`FLOPs` 可以预估为: 2ℎ，但通常忽略不计。

#### 单层layer

将上述计算量相加，得到前向传播阶段中每个transformer层的计算量大约为 $24bsℎ2+4bs2ℎ $，可以发现：

* 参数量和计算量跟head数量无关，head划分更多是通过特征子空间划分提高精度，而不是为了节省参数量或者计算量。
* 回忆参数量是\(12lh^2\)，所以在给定固定序列长度的情况下，计算量也随着参数的数量增加而线性增加。
* 计算复杂度随着序列长度的增加呈二次方增加的趋势。

| Attention | 计算量 | FFN | 计算量 |
| --- | --- | --- | --- |
| 计算Q、K、V | \(6 bsh^2\) | 第一个线性层 | \(8 bsℎ^2\) |
| QK^T | \(2 bs^2h\) | 第二个线性层 | \(8 bsℎ^2\) |
| 乘以V | \(2 bs^2h\) |  |  |
| 线性映射 | \(2 bsℎ^2\) |  |  |

### 4.3 综合思考

模型的训练包含前向传播和反向传播过程。上述只是主要考虑到前向传播阶段中，Transformer的计算量。我们接下来结合反向传播来综合考虑。反向传播过程实际上包含两部分，一部分是对输入的梯度的计算，一部分是对权重的梯度。其实这两部分主要的计算量都是矩阵乘法，并且大小与 前向传播中的计算量大小一致，因此往往会直接把反向传播的计算量近似为前向传播的 2 倍。

#### 反向传播

我们把反向传播加进来继续分析。

##### 单层

单个Transformer层的计算量现在如下：

* 前向传播所需要的浮点数运算：\(24 bsℎ^2 + 4 bs^2ℎ\)。

* 对于backward，对于神经网络中的权重和输入需要计算梯度，因此反向传播需要2倍FLOPs。
* 如果使用activation checkpointing：在backward的时候，每一层需要额外的计算forward。

所以每层需要的总浮点数计算为\(4×(24 bsℎ^2 + 4 bs^2ℎ)=96bsℎ^2(1+s/6ℎ)\)。

##### logits

另一个耗费计算量的部分是logits的计算：将隐藏向量映射为词表大小，得到每个 token 对应的 logits 向量。矩阵乘法的输入和输出形状为 [b,s,ℎ]×[ℎ,V]→[b,s,V] 。矩阵乘法的输入和输出形状为: [𝑠,ℎ]×[ℎ,𝑉]−>[*s*,*V*]。

因此前向传播需要 2bsℎV ，反向传播需要 4bsℎV ，总体需要 6bsℎV 的计算量。

#### 总体计算量

Megatron-LM的经典论文 "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"给出了一个计算标准Transformer-decoder结构浮点数运算的公式。对于一个 l 层的transformer模型，输入形状为 [b,s] 时，其计算量如下。

* 单次推理，前向传播所需要的浮点数运算：\(l\times(24bsℎ^2+4bs^2ℎ)+2bsℎV\)
* 单次训练，前向后向传播需要浮点运算为：

\[96blsh^2(1+\frac{s}{6ℎ}+\frac{V}{16lℎ}) + 2bshV
\]

如果没有如果使用activation checkpointing，则是

\[72blsh^2(1+\frac{s}{6ℎ}+\frac{V}{16lℎ}) + 2bshV
\]

在Megatron-Deepspeed的代码里，我们也能看到用这个公式来计算TFLOPS（每秒所执行的浮点运算次数，floating-point operations per second）：

```
# General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
# https://arxiv.org/pdf/2104.04473.pdf).
# The factor of 4 is when used with activation check-pointing,
# otherwise it will be 3, but for 200B model, activation check-pointing will always be on.
checkpoint_activations_factor = 4 if args.checkpoint_activations else 3
# GLU activations double the hidden states in the upscaling feed-forward in each transformer layer
# This leads to 16bsh^2 instead of 8bsh^2 per first feed-forward layer in MLP, thus we increase the coefficient by 8.
# Refer to https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/283#issue-1260805063 for more details.
coefficient = 32 if args.glu_activation else 24
flops_per_iteration = (coefficient * checkpoint_activations_factor * batch_size * seq_len * num_layers * (hidden_size**2)) * (1. + (seq_len / (6. * hidden_size)) + (vocab_size / (16. * num_layers * hidden_size)))
tflops = flops_per_iteration / (elapsed_time_per_iteration * args.world_size * (10**12))
```

### 4.4 计算特点

#### 与参数量的关系

我们先给出结论：计算量主要和模型参数和token数相关。假设数据集中总共包含 D 个 Token，模型参数量为N，则对于序列不是特别长的场景，所有 Token Forward的计算量可以近似为2ND。

##### 单次推理

单次推理时候，计算量和参数量的关系如下：

\[\frac{计算量}{参数量} = \frac{l∗(24bsℎ^2+4bs^2ℎ)+2bsℎV}{l(12h^2 + 13h) + 2vh} \approx \frac{24lbsh^2}{12lh^2} = 2bs
\]

因为单次推理时输入的token数为bs，因此可以近似认为，在一次前向传播中，对于每个token，每个模型参数需要进行2次浮点运算（一次乘法，一次加法）。即从单个 Token 单个矩阵乘的视角，可以近似认为，单次推理时（只包含正向传播）的计算量就是参数量的 2 倍，就是每个 token 过一遍所有参数的计算量。

一次迭代训练包含了前向传播和后向传播，后向传播的计算量是前向传播的 `2` 倍。因此，即一次迭代训练中，对于每个 token 和 每个模型参数，需要进行 6 次浮点数运算。

在论文"Scaling Laws for Neural Language Model"中也有类似的计算公式，具体如下图所示。

![](images/img_009.jpg)

##### 单次训练

一次训练迭代包含了前向传播和后向传播，因为反向传播计算量是前向传播的2倍，所以单次训练，对于每个token，每个模型参数需要进行6次浮点运算。训练总算力（Flops）= 6 \* 模型的参数量 \* 训练数据的 token 数。这就是所有训练数据过一遍训练所需的算力。如果需要训练需要多少时长，则可以近似使用下面公式：

\[\frac{6 \times 模型参数量 \times 数据量 }{(GPU数量 \times GPU\ FLOPS \times GPU利用率)}
\]

#### 带宽受限

算力并不能说明一切，模型还需要访问 GPU 内存，内存带宽也可能成为瓶颈。因为需要把参数从内存里面读出来吧？内存访问量 = 参数数量 \* 2 bytes。针对内存带宽部分，大语言模型中的计算具备一些鲜明的特点。我们一一进行分析。

##### 注意力计算

在大语言模型推理中，注意力计算是访存密集型的，其耗时受限于硬件的访存带宽，而非运算速度。

对于矩阵乘法算子，其特点如下。

* 参数量太大会导致矩阵乘法算子成为访存密集型。当输入的数据不够多，运算量不够大的时候，这些算子会因为参数访存过多而受限于访存带宽。
* 计算量将随着Batchsize增长而快速增加。当Batchsize小于16时，我们可以认为矩阵乘法算子为访存密集型的。只有当Batchsize充分大时，矩阵乘法算子才会变成计算密集型的，它们的性质会随着Batchsize变化而变化。

##### FFN计算

对于FFN算子，在大多数端侧应用中，我们都是以Batchsize=1的方式去调用大语言模型，此时网络中大部分的计算量和访存量都集中在FFN中。大语言模型整体的运算-访存比极低，整个网络都将是访存密集的，其运行耗时完全受限于访存带宽而非硬件算力。

#### KV Cache的影响

KV Cache是对注意力优化的重要途经。它本质上是文本中每个之前位置的 key 向量和 value 向量的集合。这项技术的出现大大缩减了Self Attention的计算量。这使得在KV Cache技术出现后，可以把推理流程分为prefill和decode阶段（我们会在后文详细分析）。下图就是decode阶段对应的图例。概括地说，其中包含了两种算子：

* 自注意力（*Self-Attention*，黄色标出）涉及矩阵-矩阵乘法。
* 密集投影（*Dense Projection*，绿色标出）涉及向量-矩阵乘法。

Self Attention算子的计算特点非常显著：这是一个运算访存比接近1:1的访存密集型算子。对其访存量和计算量进行理论估计，可得发现，其内存访问量和计算量的复杂度都是\(O(batch\ size \times sequence\ length \times hidden \ dimension)\)。作为对比，对MatMul和FeedForward（都是矩阵乘法算子）做类似的估计，可得结论：其内存访问量和计算量的复杂度都是\(O(batch\ size \times hidden \ dimension)\)。

![](images/img_010.jpg)

##### prefill

MHA 块的 `FLOPs`: \(8sh^2 + 4s^2h\)。FFN的是\(16sh^2\)。

##### decode

`MHA` 层每一轮解码的 `FLOPs`: \(8h^2 + 4(s+1)h\)。FFN的是\(16h^2\)。

##### 总体

而在输入数据形状为 [𝑏,𝑠]的情况下，一次训练/推理：

`prefill` 阶段每轮总计算量：\(𝑏×(24lℎ^2𝑠+4lℎ𝑠^2)+2𝑏𝑠ℎ𝑉)=24lℎ^2𝑏𝑠+4lℎ𝑏𝑠^2+2𝑏𝑠ℎ𝑉\)

`decode` 阶段每轮总计算量：\(𝑏×(8lℎ^2+4lℎ(𝑠+1)+16lℎ^2)+2𝑏ℎ𝑉=24lℎ^2𝑏+4lℎ𝑏(𝑠+1)+2𝑏𝑠ℎ𝑉\)

##### kv cache 节省了多少计算量

对于上下文长度 *s*，不使用 kv cache d的 self-attention 的总计算量复杂度为：总计算量：\(𝑂(𝑠^3ℎ)\)，使用后的总计算量近似为 \(𝑂(𝑠^2ℎ)\)。计算量节省比率：

\[节省比率=𝑂(𝑠^3ℎ)−𝑂(𝑠^2ℎ)=1−\frac{1}{s}
\]

计算复杂度从 \(𝑂(𝑠^3ℎ)\) 降低到 \(𝑂(𝑠^2ℎ)\)，即使用 kv cache 可节省约 𝑠 倍的计算量。当 𝑠较大时，1/s接近于 0。输出 tokens 数越多，计算量节省越可观。

## 0x05 优化方向

自回归大语言模型在运行效率上面最大的缺陷是解码过程是串行和变长的，并行计算和内存带宽资源无法得到高效利用，进而也导致了内存的管理和回收问题。针对此情形，工业界已经出现了不少的系统优化方案，这些上面每种技术手段都可以大幅度地提升模型推理的速度、性能。

### 5.1 基于注意力机制来修改外推技术

博文“How Do Language Models put Attention Weights over Long Context“中提到，不同层的注意力分布有显著差异：

* 起始层主要是词嵌入和词嵌入的一层层混合，注意力分布大致均匀。
* 中间层的注意力模式变得更加复杂，大部分概率质量集中在初始标记（注意力汇聚）和最近的/最后标记（近期偏见）上。
* 最后层则可以看到所有的注意力模式。

从上面可以看出，也就是说，中间层大部分都是“V形”注意力分布，意味着中间层很多的token其实作用不大。因此可以考虑针对不同的层来通过减少token的方式来加速推理，增加外推能力。

我们接下来就看看如何基于注意力机制来增加外推能力。

| 名称 | 主要思想 |
| --- | --- |
| StreamLLM | 在组装KV-Cache的时，包括所有头部的token（Sink模式），同时引入Window Attention机制来提高计算效率。 |
| LM-Infinite | 采用V-shaped注意力机制。因为中间token注意力分布较少，因此引入Λ形注意力掩码，也设置一个距离上限来限制“有效距离”。同时可以选择性地关注中间的具有最大的注意力logits的k个tokens。 |
| SirLLM | 通过度量Token的熵和一个记忆衰减机制来筛选关键短语。熵值高的token被认为包含更多的信息。记忆衰减机制是：将token熵缓存中的每个熵值乘以一个小于1的衰减比率。随着时间的推移，较早的信息会逐渐被遗忘，而最近的关键信息则被保留。 |
| Sparase-Q | 令牌通常只关注序列的一小部分。如果能有效地预测哪些令牌将获得高注意力分数，就可以仅存储高分令牌的键值，从而提高内存带宽效率。因此提出一种压缩思想，通过估计最大注意力分数来选择r个分量，然后确定top-k的key向量和value向量。 |
| Dynamic Memory Compression | DMC在预训练的LLMs上进行微调来学习压缩策略，然后在推理时对关键值缓存进行在线压缩。DMC引入了决策变量α和重要性变量ω，这些变量在每个时间步骤决定是将当前的key和value表示追加到缓存中，还是与缓存中的顶部元素进行加权平均。 |
| Infini-attention | 将压缩记忆（compressive memory）整合到标准的注意力机制中，并在单个 Transformer 块中构建了掩蔽局部注意力（masked local attention）和长期线性注意力（long-term linear attention）机制。 |
| LongLoRA | 引入Shifted Sparse Attention对模型进行微调以此对上下文长度进行拓展。经过Shifted Sparse Attention微调的模型在推理时保留了原始的标准自注意力架构。这意味着在推理阶段，模型可以使用未修改的注意力机制，从而使得大部分现有的优化和基础设施可以重用。 |
| self-extend Attention | 使用简单的floor division操作将未见过的大的相对位置映射到预训练期间遇到的相对位置。为了解决长距离依赖和邻近依赖的问题，Self Extend引入了双层注意力机制：分组注意力（Grouped Attention）和邻近注意力（Neighbor Attention）。 |
| Dual Chunk Attention | 通过将长序列的注意力计算分解为基于块的模块，使得模型能够有效地捕获同一块内（Intra-Chunk）和不同块间（Inter-Chunk）的相对位置信息。然后将内部块、跨块和连续块的注意力输出合并，得到最终的输出表示。这一表示考虑了序列中的局部和全局信息，从而使得模型能够有效地处理长序列。 |

### 5.2 基于Memory机制外推技术

基于Memory机制的外推技术其实沿用的还是压缩思想，借助外部存储将历史信息存储，然后使用最近的token进行查询获取一些历史上重要的token。

| 名称 | 主要思想 |
| --- | --- |
| InfLLM | 通过构建一个额外的上下文记忆模块来让存储远离当前处理位置的上下文信息，并设计了一个高效的机制来查找与当前处理的标记相关的单元，以便在注意力计算中使用。 |
| Recurrent Memory Transformer (RMT) | 通过结合循环神经网络（RNN）的循环机制和Transformer模型的记忆增强能力来实现上下文拓展。RMT在Transformer模型的基础上引入了一个记忆机制，该机制由一组可训练的实值向量（称为记忆标记）组成。这些记忆向量可以存储和处理局部和全局信息，并通过循环机制在长序列的不同段之间传递信息。 |

## 0xFF 参考

[多个大语言微调模型并行推断的潜力](https://abcdabcd987.com/2023/09/11/multi-lora-potentials/)

[Contiguous Batching/Inflight Batching](https://www.usenix.org/system/files/osdi22-yu.pdf)

[Full Stack Transformer Inference Optimization Season 2: Deploying Long-Context Models](https://yaofu.notion.site/Full-Stack-Transformer-Inference-Optimization-Season-2-Deploying-Long-Context-Models-ee25d3a77ba14f73b8ae19147f77d5e2) Yao Fu [Paper version](https://arxiv.org/abs/2405.08944)

[GPTQ](https://arxiv.org/abs/2210.17323)/[AWQ](https://arxiv.org/abs/2306.00978)

[How Do Language Models put Attention Weights over Long Context](https://yaofu.notion.site/How-Do-Language-Models-put-Attention-Weights-over-Long-Context-10250219d5ce42e8b465087c383a034e) Yao Fu

[HunYuan MoE：聊一聊 LLM 参数量、计算量和 MFU 等](https://mp.weixin.qq.com/s?__biz=Mzk0ODU3MjcxNA==&mid=2247488396&idx=1&sn=d58aaec004c2a4c0a0db597579e27eb4&chksm=c29a8e1154dd5de28a904f8af75b683c2c9ec93a32b9d6eabe8febe75527172e42c293e17f09&mpshare=1&scene=1&srcid=111493vB4moJWO5thWubiqww&sharer_shareinfo=232541b218d616dd703ebe350f295cc3&sharer_shareinfo_first=232541b218d616dd703ebe350f295cc3#rd) AI闲谈

[llm 参数量-计算量-显存占用分析](https://www.armcvai.cn/2024-09-20/llm-params-flops.html) Zhang

[LLM 大模型训练-推理显存占用分析](https://bruceyuan.com/post/llm-train-infer-memoery-usage-calculation.html) [chaofa用代码打点酱油](https://bruceyuan.com/)

[LLM（廿三）：LLM 中的长文本问题](https://zhuanlan.zhihu.com/p/640641794) [紫气东来](https://www.zhihu.com/people/zi-qi-dong-lai-1)

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://yaofu.notion.site/Full-Stack-Transformer-Inference-Optimization-Season-2-Deploying-Long-Context-Models-ee25d3a77ba14f73b8ae19147f77d5e2)

[OpenPPL-LLM | OpenPPL之大语言模型推理引擎来啦](https://zhuanlan.zhihu.com/p/653808774) [OpenPPL](https://www.zhihu.com/people/openppl)

[PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)

[Towards 100x Speedup: Full Stack Transformer Inference Optimization](https://yaofu.notion.site/Towards-100x-Speedup-Full-Stack-Transformer-Inference-Optimization-43124c3688e14cffaf2f1d6cbdf26c6c) Yao Fu

[Transformer 估算 101](https://mp.weixin.qq.com/s/MFgTUDAOODgMDb59eZC9Cw)

[Transformer 数据估计- 显存占用](https://zhuanlan.zhihu.com/p/678503627) [Bruce 仗剑走天涯](https://www.zhihu.com/people/void-73-73)

[分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065) [回旋托马斯x](https://www.zhihu.com/people/springxchen)

[剖析GPT推断中的批处理效应](https://abcdabcd987.com/2023/05/13/transformer-batching/) [Lequn Chen || abcdabcd987](https://abcdabcd987.com/blog)

[多个大语言微调模型并行推断的潜力](https://abcdabcd987.com/2023/09/11/multi-lora-potentials/) [Lequn Chen || abcdabcd987](https://abcdabcd987.com/blog)

[大模型 - 部署 - 容量估算](https://zhuanlan.zhihu.com/p/694980607) [思想柳叶刀](https://www.zhihu.com/people/calvin-97-63)

[大模型推理瓶颈及极限理论值分析](http://mp.weixin.qq.com/s?__biz=Mzg4MTkwMTQ4NA==&mid=2247485270&idx=1&sn=fbbcae4d787bd67e0d4dd5e18aa98e8c&chksm=cf5fad95f828248302e0cb134d2f75987e71b5b6b2344d3dd6288d9a7ae67250b26157f90090&scene=21#wechat_redirect) 喜欢卷卷的瓦力

[激活内存：模型推理需要多少内存](https://mp.weixin.qq.com/s?__biz=MzAwMDc2NjQ4Nw==&mid=2663562897&idx=1&sn=436b6c073f892d4f48c2f9d3ea81b6f6&chksm=8035e6969544f4896747afb93b23f28cecf068ac9970724162b6b00f35796f536fc448ef16fd&mpshare=1&scene=1&srcid=0130YyBBgZp8luoN0yD9jHOE&sharer_shareinfo=8a3a771a4a01dfd8c77aa118f9bc1a6e&sharer_shareinfo_first=8a3a771a4a01dfd8c77aa118f9bc1a6e#rd) 魏新宇 [大魏分享]
