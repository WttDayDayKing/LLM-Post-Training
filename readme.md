该仓库包括可参考的数据配方和后训练技术。分别对应两个文件夹。
调研：
* 数据构建
** 数据混合
DoReMi（Group DRO 训代理模型）、
DOGE（最小化反传梯度差异定权重）、
Data Mixing Law（拟合配比-验证损失关系）、
IDEAL（梯度引导迭代式小幅精修）：https://github.com/ming-bot/IDEAL
PIKE
** 数据选择
LESS（一阶梯度对齐目标分布）、
SelectIT（用 LLM 内部不确定性选数据），
ADAPT（计算样本与验证机样本在模型最后一层隐藏状态值差异作为质量分数进行重加权）等聚焦实例级。ACSESS（前向选择、后向选择和 Datamodels 三种机制自动识别互补的样本选择策略并加权组合）。

* 后训练
OPSD、SDFT、

思路：
* 数据配方
** 层次化数据构建；
1、按照数据域动态调整配比（IDEAL），进行上采样和下采样；
2、样本级动态重加权，更新完配比后，根据质量分数对样本进行重加权；
LESS（一阶梯度对齐目标分布）、SelectIT（用 LLM 内部不确定性选数据）等聚焦实例级，与 IDEAL 的领域级配比正交、可互补。
* 后训练
** domain-level training
1、on-policy self-