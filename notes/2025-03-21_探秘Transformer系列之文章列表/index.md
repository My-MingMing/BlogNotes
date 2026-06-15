# 探秘Transformer系列之文章列表

## 0x01 系列说明

因为各种事情，好久没有写博客了，之前写得一些草稿也没有时间整理（都没有时间登录博客和微信，导致最近才发现好多未读消息和私信，在这里和各位朋友说下万分抱歉）。现在恢复更新，是因为最近有些从非AI领域转过来的新同学来找我询问是否有比较好的学习资料，他们希望在短期内迅速上手 Transformer。我在网上找了下，但是没有找到非常合适的系统的学习资料，于是就萌发了自己写一个系列的想法，遂有此系列。在整理过程中，我也发现了自己很多似是而非的错误理解，因此这个系列也是自己一个整理、学习和提高的过程。

本系列试图从零开始解析Transformer，目标是：

* 解析Transformer如何运作，以及为何如此运作，让新同学可以入门Transformer。
* 力争融入一些比较新的或者有特色的论文或者理念，让老鸟也可以通过阅读本系列来了解一些新观点，有所收获。

几点说明：

* 本系列是对论文、博客和代码的学习和解读，借鉴了很多网上朋友的文章，在此表示感谢，并且会在参考中列出。因为本系列参考文章太多，可能有漏给出处的现象。如果原作者发现，还请指出，我在参考文献中进行增补。
* 本系列有些内容是个人梳理和思考的结果（反推或者猜测），可能和原始论文作者的思路或者与实际历史发展轨迹不尽相同。这么写是因为这样推导让我觉得可以给出直观且合理的解释。如果理解有误，还请各位读者指出。
* 对于某些领域，这里会融入目前一些较新的或者有特色的解释，因为笔者的时间和精力有限，难以阅读大量文献。如果有遗漏的精品文献，也请各位读者指出。

## 0x02 目录

[探秘Transformer系列之（1）：注意力机制](https://www.cnblogs.com/rossiXYZ/p/18705809)

[探秘Transformer系列之（2）---总体架构](https://www.cnblogs.com/rossiXYZ/p/18706134)

[探秘Transformer系列之（3）---数据处理](https://www.cnblogs.com/rossiXYZ/p/18722830)

[探秘Transformer系列之（4）--- 编码器 & 解码器](https://www.cnblogs.com/rossiXYZ/p/18727704)

[探秘Transformer系列之（5）--- 训练&推理](https://www.cnblogs.com/rossiXYZ/p/18730583)

[探秘Transformer系列之（6）--- token](https://www.cnblogs.com/rossiXYZ/p/18732939)

[探秘Transformer系列之（7）--- embedding](https://www.cnblogs.com/rossiXYZ/p/18741857)

[探秘Transformer系列之（8）--- 位置编码](https://www.cnblogs.com/rossiXYZ/p/18744797)

[探秘Transformer系列之（9）--- 位置编码分类](https://www.cnblogs.com/rossiXYZ/p/18746838)

[探秘Transformer系列之（10）--- 自注意力](https://www.cnblogs.com/rossiXYZ/p/18751758)

[探秘Transformer系列之（11）--- 掩码](https://www.cnblogs.com/rossiXYZ/p/18758992)

[探秘Transformer系列之（12）--- 多头自注意力](https://www.cnblogs.com/rossiXYZ/p/18759167)

[探秘Transformer系列之（13）--- FFN](https://www.cnblogs.com/rossiXYZ/p/18765884)

[探秘Transformer系列之（14）--- 残差网络和归一化](https://www.cnblogs.com/rossiXYZ/p/18774865)

[探秘Transformer系列之（15）--- 采样和输出](https://www.cnblogs.com/rossiXYZ/p/18777574)

[探秘Transformer系列之（16）--- 资源占用](https://www.cnblogs.com/rossiXYZ/p/18785615)

[探秘Transformer系列之（17）--- RoPE](https://www.cnblogs.com/rossiXYZ/p/18787343)

[探秘Transformer系列之（18）--- FlashAttention](https://www.cnblogs.com/rossiXYZ/p/18791822)

[探秘Transformer系列之（19）----FlashAttention V2 及升级版本](https://www.cnblogs.com/rossiXYZ/p/18798185)

[探秘Transformer系列之（20）--- KV Cache](https://www.cnblogs.com/rossiXYZ/p/18799503)

[探秘Transformer系列之（21）--- MoE](https://www.cnblogs.com/rossiXYZ/p/18800825)

[探秘Transformer系列之（22）--- LoRA](https://www.cnblogs.com/rossiXYZ/p/18805103)

[探秘Transformer系列之（23）--- 长度外推](https://www.cnblogs.com/rossiXYZ/p/18808744)

[探秘Transformer系列之（24）--- KV Cache优化](https://www.cnblogs.com/rossiXYZ/p/18811723)

[探秘Transformer系列之（25）--- KV Cache优化之处理长文本序列](https://www.cnblogs.com/rossiXYZ/p/18811785)

[探秘Transformer系列之（26）--- KV Cache优化---分离or合并](https://www.cnblogs.com/rossiXYZ/p/18815541)

[探秘Transformer系列之（27）--- MQA & GQA](https://www.cnblogs.com/rossiXYZ/p/18823734)

[探秘Transformer系列之（28）--- DeepSeek MLA](https://www.cnblogs.com/rossiXYZ/p/18827618)

[探秘Transformer系列之（29）--- DeepSeek MoE](https://www.cnblogs.com/rossiXYZ/p/18835426)

[探秘Transformer系列之（30）--- 投机解码](https://www.cnblogs.com/rossiXYZ/p/18837229)

[探秘Transformer系列之（31）--- Medusa](https://www.cnblogs.com/rossiXYZ/p/18845475)
