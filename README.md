# CraftBuddy / 智心匠，智能组装引导助手

## 功能：

1. 视频语义分割：根据教学视频，自动拆解步骤，
2. 视频语义描述：根据提示，结合动作，生成动作指引；
3. 视频语义比对：输入实际操作视频，和教学视频进行比对，生成改进建议；
4. 视频语义解构：生成结构化时序场景图，用于下游应用分析；
5. 视频语义生成：根据标准操作流程，生成视频指引。

## 数据集

* [宜家家居组装](https://ikeaasm.github.io/)

## 实施路线

### 参考项目：

* TimeChat([code](https://github.com/RenShuhuai-Andy/TimeChat), [paper](https://arxiv.org/abs/2312.02051))：从BLIP2架构演化而来，数据集主要是YouCook。考虑可以使用宜家组装数据集，重新训练一个针对组装类视频的VLM模型。

<p align="center" width="100%">
<a target="_blank"><img src="https://github.com/RenShuhuai-Andy/TimeChat/raw/master/figs/arch.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

* VisualNarrationProceL([code](https://github.com/Yuhan-Shen/VisualNarrationProceL-CVPR21), [paper](https://www.khoury.northeastern.edu/home/eelhami/publications/VisualNarrationProceL-CVPR21.pdf))：专门针对指导类视频和文本之间进行步骤弱对齐的一篇文章。

![image](https://github.com/tongda/CraftBuddy/assets/653425/98586b18-f117-42aa-bab6-b806a5132371)

### 实施计划：

1. 先做一个练手的项目：基于InternLM2复现一个BLIP2模型，熟悉一下Vision Encoder, QFormer等VLM模型常用的组件，以及VLM模型的评估方法，数据集格式等；
2. 构建数据集：参考TimeChat中使用的TimeIT数据集，将宜家组装数据集构建成TimeIT格式；
3. 开发Pipeline：TBD
4. 模型训练：TBD
