# Methodology

## 1. 方法概述

给定历史轨迹序列 $\mathcal{X}=\{\mathbf{x}_t\}_{t=1}^{L}$，其中 $\mathbf{x}_t \in \mathbb{R}^{C}$ 表示第 $t$ 个观测时刻的多维运动状态；同时给定未来目标时间间隔序列 $\boldsymbol{\delta}=\{\delta_h\}_{h=1}^{H}$，我们的目标是预测未来 $H$ 个时刻的归一化二维运动增量 $\hat{\mathcal{Y}}^{\text{norm}}=\{\hat{\mathbf{y}}_h^{\text{norm}}\}_{h=1}^{H}$，并进一步重建未来轨迹位置序列 $\hat{\mathcal{P}}=\{\hat{\mathbf{p}}_h\}_{h=1}^{H}$。与逐点递归预测不同，我们将整个框架建模为一个分层的 **patch encoder + time-conditioned query decoder**。核心思想是：先在局部时间窗口内编码点级依赖，再在 patch 级别聚合长程历史上下文，最后依据未来时间间隔构造逐步的条件 query，通过交叉注意力从历史记忆中检索与每个未来时刻最相关的信息。

## 2. Patch 级分层历史编码

我们将长度为 $L$ 的历史序列划分为 $N$ 个 patch，每个 patch 含有 $P$ 个观测点，即 $L=N\times P$。若输入长度与该设定不完全一致，则在实现中进行截断或补零。对第 $n$ 个 patch 中的第 $p$ 个点，我们首先根据历史中的时间间隔通道累积得到时间戳 $\tau_{n,p}$，再通过可学习时间映射 $\phi_t(\cdot)$ 得到时间嵌入，并与原始点特征拼接形成点级表示：

$$
\mathbf{z}_{n,p}=[\mathbf{x}_{n,p};\phi_t(\tau_{n,p})].
$$

在每个 patch 内部，我们使用点级多头自注意力和前馈网络建模局部动态依赖，然后利用有效掩码做 masked mean pooling 得到局部 patch token $\mathbf{h}_n^{\text{loc}} \in \mathbb{R}^{d}$。若该 patch 至少包含一个有效点，我们进一步加入一个可学习的 existence bias，以显式区分“有效 patch”和“空 patch”。

仅依赖局部 patch token 仍不足以捕获长程时序依赖，因此我们进一步在 patch 序列上进行上下文建模。与标准 Transformer 不同，我们在 patch 级注意力中引入一个显式的时间距离偏置：

$$
b_{ij}=\log\left(\exp\left(-\frac{|i-j|\Delta_p}{\tau}\right)+\epsilon\right),
$$

其中 $\Delta_p$ 表示 patch 级时间步长，$\tau$ 为衰减温度。该偏置鼓励模型优先关注时间上更接近的 patch，同时保留全局交互能力。于是，在第 $\ell$ 层中，patch 表示先经过带时间偏置的 patch attention，再经过标准 Transformer 编码块：

$$
\bar{\mathbf{h}}_i^{(\ell)}=\mathbf{h}_i^{(\ell-1)}+\sum_{j=1}^{N}\text{Attn}_{ij}^{(\ell)}\mathbf{v}_j^{(\ell)},
$$

$$
\mathbf{h}_i^{(\ell)}=\bar{\mathbf{h}}_i^{(\ell)}+\text{Transformer}^{(\ell)}(\text{PE}(\bar{\mathbf{h}}_i^{(\ell)})).
$$

经过 $M$ 层堆叠后，我们得到最终的历史记忆 $\mathcal{H}=\{\mathbf{h}_1^{(M)},\dots,\mathbf{h}_N^{(M)}\}$。这一分层设计将 patch 内局部动态建模与 patch 间全局依赖建模解耦，在保证表达能力的同时降低了对完整长序列直接做注意力计算的成本。

## 3. 基于时间条件查询的未来解码

为了让未来预测不仅依赖目标时间间隔，还能显式感知当前历史状态，我们首先构造一个全局历史锚点。设最后一个有效观测为 $\mathbf{x}_{\text{last}}$，最后一个有效 patch 的上下文化 token 为 $\mathbf{h}_{\text{last}}^{(M)}$，则全局锚点定义为

$$
\mathbf{g}=\psi_g([\mathbf{x}_{\text{last}};\mathbf{h}_{\text{last}}^{(M)}]),
$$

其中 $\psi_g(\cdot)$ 为一个 MLP。随后，对于第 $h$ 个未来预测步，我们分别编码三类信息：一是目标时间间隔 $\delta_h$，我们先做对数压缩，再通过可学习映射得到时间嵌入 $\mathbf{d}_h=\phi_f(\log(1+\delta_h))$；二是该预测步的步号嵌入 $\mathbf{s}_h=\text{Emb}(h)$；三是全局历史锚点 $\mathbf{g}$。三者拼接后送入 query 投影网络，得到该未来步对应的条件 query：

$$
\mathbf{q}_h=\psi_q([\mathbf{g};\mathbf{d}_h;\mathbf{s}_h]).
$$

这种设计使每个未来步都拥有独立的时间语义和状态语义：即使两个未来步具有相同或相近的时间间隔，模型仍然可以依靠步号嵌入将其区分；同时，全局锚点将当前轨迹状态显式注入了解码过程。

给定未来 query 序列 $\mathcal{Q}=\{\mathbf{q}_1,\dots,\mathbf{q}_H\}$ 以及历史记忆 $\mathcal{H}$，我们通过交叉注意力为每一个未来步检索其最相关的历史上下文：

$$
\tilde{\mathbf{c}}_h=\text{LN}\left(\text{MHA}(\mathbf{q}_h,\mathcal{H},\mathcal{H})+\mathbf{q}_h\right).
$$

最后，将检索得到的上下文表示、原始 query、最后观测以及时间间隔嵌入共同输入到步级解码器中，预测该时刻的归一化运动增量：

$$
\hat{\mathbf{y}}_h^{\text{norm}}=\psi_d([\tilde{\mathbf{c}}_h;\mathbf{q}_h;\mathbf{x}_{\text{last}};\mathbf{d}_h]).
$$

相较于单一的全局回归头，这种 query-based 解码方式更适合不规则时间间隔下的轨迹预测，因为每个未来步都可以依据自身的目标间隔从历史中检索不同的信息模式。

## 4. 轨迹重建与训练目标

网络直接输出的是归一化运动增量。设运动尺度向量为 $\mathbf{s}_m$，位置尺度向量为 $\mathbf{s}_p$，则物理空间中的运动增量首先由反归一化得到。在速度建模模式下，第 $h$ 步位移为 $\Delta \hat{\mathbf{p}}_h=\hat{\mathbf{y}}_h\cdot\delta_h$；在位移建模模式下，则直接有 $\Delta \hat{\mathbf{p}}_h=\hat{\mathbf{y}}_h$。未来轨迹位置由累积和重建：

$$
\hat{\mathbf{p}}_h=\sum_{r=1}^{h}\Delta\hat{\mathbf{p}}_r.
$$

训练时，我们从三个层面共同约束模型：逐步运动增量、完整未来轨迹形状以及最终落点误差。总体预测损失写为

$$
\mathcal{L}=
\lambda_m\mathcal{L}_{\text{motion}}+
\lambda_t\mathcal{L}_{\text{traj}}+
\lambda_f\mathcal{L}_{\text{final}},
$$

其中

$$
\mathcal{L}_{\text{motion}}=\frac{1}{H}\sum_{h=1}^{H}\ell(\hat{\mathbf{y}}_h^{\text{norm}},\mathbf{y}_h^{\text{norm}}),
$$

$$
\mathcal{L}_{\text{traj}}=\frac{1}{H}\sum_{h=1}^{H}\ell(\hat{\mathbf{p}}_h^{\text{norm}},\mathbf{p}_h^{\text{norm}}),
$$

$$
\mathcal{L}_{\text{final}}=\ell(\hat{\mathbf{p}}_H^{\text{norm}},\mathbf{p}_H^{\text{norm}}).
$$

这里 $\ell(\cdot,\cdot)$ 可选为 Smooth L1 或 MSE。三者联合能够同时保证局部运动精度、全局轨迹一致性以及终点预测能力。

## 5. 基于时间条件查询的局部 Patch 补全

除未来预测外，我们还考虑历史序列中的 patch 补全问题。设第 $m$ 个 patch 为缺失 patch。与直接利用全局历史做补全不同，我们将补全严格限制在一个固定的三 patch 局部 chunk 内。记该局部可见 patch 集合为 $\mathcal{N}(m)=\{j_1,j_2,j_3\}$，则相应的局部记忆为 $\mathcal{H}_m^{\text{loc}}=\{\mathbf{h}_{j_1}^{(M)},\mathbf{h}_{j_2}^{(M)},\mathbf{h}_{j_3}^{(M)}\}$。这种设计引入了明确的局部连续性先验：对于短时轨迹缺失，近邻运动模式往往比远距离上下文更可靠。

为了恢复缺失 patch，我们构造一个带有时间语义、位置语义和局部状态语义的补全 query。设缺失 patch 的中心时间戳为 $\tau_m^c$，patch 索引嵌入为 $\mathbf{u}_m=\text{Emb}(m)$，左右最近的可见 patch 表示分别为 $\mathbf{h}_m^{-}$ 和 $\mathbf{h}_m^{+}$，局部三 patch chunk 的平均摘要为 $\bar{\mathbf{h}}_m$。则补全 query 定义为

$$
\mathbf{q}_m^{\text{comp}}=\psi_c([\phi_c(\tau_m^c);\mathbf{u}_m;\mathbf{h}_m^{-};\mathbf{h}_m^{+};\bar{\mathbf{h}}_m]).
$$

在此基础上，我们通过局部交叉注意力恢复缺失 patch 的潜在表示：

$$
\hat{\mathbf{h}}_m^{\text{comp}}=
\text{LN}\left(\text{MHA}(\mathbf{q}_m^{\text{comp}},\mathcal{H}_m^{\text{loc}},\mathcal{H}_m^{\text{loc}})+\mathbf{q}_m^{\text{comp}}\right).
$$

这使得补全过程不再是盲目的直接回归，而是一个局部条件检索过程：模型依据缺失 patch 的时间位置与邻域状态，从局部 chunk 中自适应抽取最相关的动态模式。

训练时，我们在 patch 级和 point 级同时监督补全结果。设 $\mathbf{h}_m^{\star}$ 为完整观测下得到的目标 patch 表示，则 patch 级补全损失为

$$
\mathcal{L}_{\text{comp\_patch}}=\ell_c(\hat{\mathbf{h}}_m^{\text{comp}},\mathbf{h}_m^{\star}).
$$

随后，我们使用一个点级解码器，结合补全后的 patch token 与 patch 内相对时间编码，逐点重建缺失 patch 内的轨迹特征：

$$
\hat{\mathbf{x}}_{m,p}=\psi_p([\hat{\mathbf{h}}_m^{\text{comp}};\phi_p(\tau_{m,p})]), \qquad p=1,\dots,P.
$$

对应的 point 级补全损失为

$$
\mathcal{L}_{\text{comp\_point}}=\frac{1}{P}\sum_{p=1}^{P}\ell_p(\hat{\mathbf{x}}_{m,p},\mathbf{x}_{m,p}^{\star}).
$$

因此，补全分支的总损失为

$$
\mathcal{L}_{\text{completion}}=
\lambda_{\text{cp}}\mathcal{L}_{\text{comp\_patch}}+
\lambda_{\text{pt}}\mathcal{L}_{\text{comp\_point}}.
$$

若与未来预测任务联合训练，则整体目标写为

$$
\mathcal{L}_{\text{all}}=\mathcal{L}+\lambda_{\text{comp}}\mathcal{L}_{\text{completion}}.
$$

可以看到，未来预测与局部补全共享同一套“时间条件 query + cross-attention”建模范式，但作用范围不同：未来预测依赖全局历史记忆以支持长程外推，而缺失 patch 补全则显式限制在三 patch 的局部 chunk 内，以强调短时平滑性和结构一致性。这种统一但分工明确的设计，使模型能够在不规则采样场景下同时处理未来轨迹预测和历史缺失恢复。
