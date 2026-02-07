# Reservoir Computing与Echo State Networks深度研究报告

## 摘要

本报告深入分析Reservoir Computing（储层计算）的理论基础，特别是Echo State Networks（ESN）的核心原理，以及近期ResFormer等将Reservoir与Transformer结合的突破性工作。我们重点探讨固定随机权重+可训练输出层这一范式的理论基础，分析其Universal Approximation能力，并探讨将Reservoir思想应用于Transformer Attention的理论可行性。本研究为正交随机注意力Transformer新范式提供理论支撑。

---

## 1. Reservoir Computing理论基础

### 1.1 Echo State Network(ESN)的核心原理

Echo State Network（ESN）是由Jaeger于2001年提出的一种递归神经网络（RNN）架构，属于Reservoir Computing（储层计算）框架的核心实现。ESN的设计理念颠覆了传统RNN的训练范式：

**核心创新**：
- **储层（Reservoir）**：一个大规模、稀疏、随机连接的循环神经网络
- **固定权重**：储层内部权重$W_{\text{res}}$和输入权重$W_{\text{in}}$随机初始化后**固定不变**
- **可训练输出层**：仅输出权重$W_{\text{out}}$通过线性回归进行训练

**ESN架构示意图**：
```
输入 u(t) → [输入层] → 储层 x(t) → [输出层] → 输出 y(t)
                      ↑___________|
                      (循环连接)
```

### 1.2 固定随机内部权重+可训练输出层的范式

#### 1.2.1 数学形式化

**状态更新方程**：

对于标准ESN，储层状态$x(t) \in \mathbb{R}^{N_r}$的更新遵循：

$$x(t) = f(W_{\text{res}} x(t-1) + W_{\text{in}} u(t) + b)$$

其中：
- $N_r$：储层神经元数量（通常$N_r \gg \text{input\_dim}$）
- $W_{\text{res}} \in \mathbb{R}^{N_r \times N_r}$：储层内部权重矩阵（**固定随机**）
- $W_{\text{in}} \in \mathbb{R}^{N_r \times N_{\text{in}}}$：输入权重矩阵（**固定随机**）
- $u(t) \in \mathbb{R}^{N_{\text{in}}}$：时刻$t$的输入
- $b \in \mathbb{R}^{N_r}$：偏置向量
- $f(\cdot)$：激活函数（通常为$\tanh$）

**泄漏积分器变体（Leaky Integrator ESN, LI-ESN）**：

$$x(t) = (1-\alpha)x(t-1) + \alpha \cdot f(W_{\text{res}} x(t-1) + W_{\text{in}} u(t) + b)$$

其中$\alpha \in (0,1]$为泄漏率（leak rate），控制状态更新的平滑程度。

**输出计算**：

$$y(t) = W_{\text{out}} \cdot [x(t); u(t)]$$

其中$[x(t); u(t)]$表示储层状态与输入的拼接（可选），$W_{\text{out}}$为**唯一可训练**的参数。

#### 1.2.2 输出层训练的解析解

ESN的训练简化为线性回归问题。给定训练序列$\{u(t), y_{\text{target}}(t)\}_{t=1}^{T}$：

**步骤1：收集储层状态**

运行储层收集状态矩阵$X \in \mathbb{R}^{T \times N_r}$：

$$X = \begin{bmatrix} x(1)^T \\ x(2)^T \\ \vdots \\ x(T)^T \end{bmatrix}$$

**步骤2：计算输出权重**

使用岭回归（Ridge Regression）求解：

$$W_{\text{out}} = Y_{\text{target}} X^T (XX^T + \lambda I)^{-1}$$

或使用伪逆（Moore-Penrose pseudoinverse）：

$$W_{\text{out}} = Y_{\text{target}} X^{\dagger}$$

其中$\lambda$为正则化参数，防止过拟合。

**关键优势**：
- 训练速度极快（线性代数运算，无需迭代优化）
- 避免梯度消失/爆炸问题
- 不存在局部最优问题

### 1.3 Echo State Property (ESP) — 回声状态性质

#### 1.3.1 形式化定义

**回声状态性质（ESP）**是ESN理论的核心概念，确保储层动力学的稳定性和一致性。

**定义**：对于紧致输入集$\mathcal{U}$上的任意有界输入序列$\{u(t)\}$，储层具有ESP当且仅当：对于任意两个初始状态$x(0)$和$x'(0)$，对应的轨迹满足：

$$\lim_{t \to \infty} \|x(t) - x'(t)\| = 0$$

即储层状态最终收敛到仅依赖于输入历史的唯一轨迹，与初始条件无关。

#### 1.3.2 ESP的充分条件

**定理**（ESP充分条件）：对于使用$\tanh$激活函数的ESN，若储层权重矩阵$W_{\text{res}}$满足：

$$\sigma_{\max}(W_{\text{res}}) < 1$$

其中$\sigma_{\max}$表示最大奇异值，则ESN具有回声状态性质。

**实用设计准则**：

1. **谱半径条件**：$\rho(W_{\text{res}}) < 1$（谱半径小于1）
   - 实践中通常设置$\rho(W_{\text{res}}) \approx 0.9$
   
2. **稀疏连接**：储层通常设置为稀疏（如10%连接密度）

3. **输入缩放**：输入权重$W_{\text{in}}$需要适当缩放

**数学证明思路**：

对于两个初始状态$x_1(t)$和$x_2(t)$，状态差$\Delta x(t) = x_1(t) - x_2(t)$满足：

$$\begin{aligned}
\|\Delta x(t+1)\| &= \|f(W_{\text{res}}x_1(t) + W_{\text{in}}u(t)) - f(W_{\text{res}}x_2(t) + W_{\text{in}}u(t))\| \\
&\leq L_f \|W_{\text{res}}(x_1(t) - x_2(t))\| \\
&\leq L_f \sigma_{\max}(W_{\text{res}}) \|\Delta x(t)\|
\end{aligned}$$

其中$L_f$为激活函数的Lipschitz常数（对于$\tanh$，$L_f = 1$）。

当$L_f \cdot \sigma_{\max}(W_{\text{res}}) < 1$时，状态差指数收敛到0。

### 1.4 Reservoir的Universal Approximation能力

#### 1.4.1 通用逼近定理

**定理**（ESN通用逼近）：设$\mathcal{U}: (\mathbb{R}^d)^{\mathbb{Z}} \to \mathbb{R}^{\mathbb{Z}}$为任意因果、时不变、具有**消逝记忆性质**（Fading Memory Property, FMP）的滤波器。对于任意$\epsilon > 0$，存在足够大的储层规模$N_r$，使得存在一个ESN实现的滤波器$\mathcal{U}_{\text{ESN}}$满足：

$$\sup_{\mathbf{u}} \|\mathcal{U}(\mathbf{u}) - \mathcal{U}_{\text{ESN}}(\mathbf{u})\|_{\infty} < \epsilon$$

**消逝记忆性质（FMP）**：滤波器$\mathcal{U}$具有FMP，如果对于任意$\epsilon > 0$，存在$\tau > 0$，使得对于所有输入序列$\mathbf{u}, \mathbf{v}$：

$$u(t) = v(t), \forall |t| \leq \tau \Rightarrow |\mathcal{U}(\mathbf{u})(0) - \mathcal{U}(\mathbf{v})(0)| < \epsilon$$

即滤波器对遥远过去的输入"遗忘"。

#### 1.4.2 随机通用逼近

**更强的结果**（随机通用逼近）：对于给定的激活函数$\phi$和分布$\mu$，若内部权重按照特定分布采样，则以高概率，ESN是通用逼近器。

这意味着：**随机生成的储层权重已经具备通用逼近能力，无需精心设计的权重初始化！**

---

## 2. 近期相关工作

### 2.1 ResFormer: Reservoir + Transformer的集成

#### 2.1.1 研究动机

2025年提出的ResFormer代表了Reservoir Computing与Transformer结合的最新突破。其核心动机：

- **Transformer的局限**：Self-Attention的$O(K^2)$复杂度限制了处理长序列的能力
- **Reservoir的优势**：线性时间复杂度$O(K)$和常数空间复杂度，天然适合长序列建模

#### 2.1.2 ResFormer架构

ResFormer采用**级联（cascaded）架构**，将两种记忆机制结合：

$$\hat{y}_i = \mathbb{T}(\mathbb{R}(\epsilon(\mathbf{u}_1^{i-1})) \uplus \epsilon(\mathbf{u}_i))$$

其中：
- $\mathbb{R}(\cdot)$：Reservoir模块（长期记忆，LTM）
- $\mathbb{T}(\cdot)$：Transformer模块（短期记忆，STM）
- $\uplus$：Cross-Attention融合操作
- $\epsilon(\cdot)$：嵌入函数

**架构细节**：

1. **长期记忆模块（LTM）**：
   - 使用泄漏积分器Reservoir处理整个语料库
   - 固定随机权重，非线性读出（ReLU激活）
   - 线性时间复杂度$O(K \cdot q \cdot d)$

2. **短期记忆模块（STM）**：
   - 标准Transformer处理单个句子
   - 捕捉token级别的局部依赖

3. **Cross-Attention融合**：
   - 将Reservoir状态与当前句子嵌入融合
   - 使Transformer能够访问全局上下文

#### 2.1.3 复杂度分析

| 模型 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Transformer | $O(K^2 d)$ | $O(Kd + K^2)$ |
| RNN | $O(K d^2)$ | $O(Kd)$ |
| Longformer | $O(Kd^2 + gKd)$ | $O(Kqd)$ |
| Mamba | $O(Krd)$ | $O(rd^2)$ |
| **ResFormer** | $O(Kqd)$ | $O((qd + q^2 + n^2)/B)$ |

其中$K$为序列长度，$q$为句子长度，$d$为隐藏维度，$n$为储层神经元数，$B$为batch size。

#### 2.1.4 实验结果

ResFormer在多个NLP分类任务上取得显著突破：

- **EmoryNLP**：相比DeepSeek-Qwen提升**+22.3%**，相比ModernBERT提升**+19.9%**
- **MELD**：相比ModernBERT提升+8.58%，相比DeepSeek提升+8.00%，相比Longformer提升+14.6%
- **内存效率**：在EmoryNLP上仅消耗基线模型**1/3**的RAM

**关键发现**：
- 非线性读出（ReLU）优于线性读出
- 泄漏参数$\alpha \approx 0.4 \sim 0.5$时性能最佳
- Cross-Attention融合方法至关重要

### 2.2 Reservoir在处理长序列方面的优势

#### 2.2.1 理论优势

1. **线性复杂度**：Reservoir状态更新为$O(N_r^2)$每步，与序列长度无关
2. **常数空间**：仅需维护当前储层状态，无需存储完整历史
3. **自然记忆**：循环结构天然保留历史信息

#### 2.2.2 记忆容量分析

ESN的记忆容量与储层规模$N_r$成正比。对于NARMA（Nonlinear AutoRegressive Moving Average）系统辨识任务，ESN展现出卓越的长程依赖建模能力。

**记忆容量定义**：

$$MC = \sum_{\tau=0}^{\infty} MC(\tau)$$

其中$MC(\tau)$为延迟$\tau$的记忆容量：

$$MC(\tau) = \frac{\text{Cov}^2(y(t), u(t-\tau))}{\text{Var}(y(t))\text{Var}(u(t-\tau))}$$

---

## 3. 数学分析

### 3.1 Reservoir状态更新方程的完整推导

#### 3.1.1 离散时间动力学

考虑标准ESN的状态更新：

$$x(t) = \tanh(W_{\text{res}} x(t-1) + W_{\text{in}} u(t) + b)$$

**展开形式**（逐元素）：

$$x_i(t) = \tanh\left(\sum_{j=1}^{N_r} W_{\text{res},ij} x_j(t-1) + \sum_{k=1}^{N_{\text{in}}} W_{\text{in},ik} u_k(t) + b_i\right)$$

#### 3.1.2 连续时间近似

当时间步长$\Delta t \to 0$时，LI-ESN可近似为连续时间动力学：

$$\tau \frac{dx(t)}{dt} = -x(t) + \tanh(W_{\text{res}} x(t) + W_{\text{in}} u(t) + b)$$

其中$\tau = \frac{1-\alpha}{\alpha} \Delta t$为时间常数。

#### 3.1.3 状态轨迹的线性化分析

在固定点$x^*$附近线性化：

$$x(t) \approx x^* + \delta x(t)$$

$$\delta x(t) \approx J \cdot \delta x(t-1)$$

其中Jacobian矩阵：

$$J = \text{diag}(1 - \tanh^2(W_{\text{res}}x^* + W_{\text{in}}u + b)) \cdot W_{\text{res}}$$

稳定性要求$J$的特征值在单位圆内。

### 3.2 输出层权重计算的解析解

#### 3.2.1 最小二乘解

给定训练数据$\{(u(t), y_{\text{target}}(t))\}_{t=1}^{T}$，收集储层状态$X \in \mathbb{R}^{T \times N_r}$和目标输出$Y \in \mathbb{R}^{T \times N_{\text{out}}}$。

**优化问题**：

$$\min_{W_{\text{out}}} \|X W_{\text{out}}^T - Y\|_F^2 + \lambda \|W_{\text{out}}\|_F^2$$

**解析解**：

$$W_{\text{out}}^T = (X^T X + \lambda I)^{-1} X^T Y$$

或等价地：

$$W_{\text{out}} = Y^T X (X^T X + \lambda I)^{-1}$$

#### 3.2.2 伪逆形式

当$\lambda = 0$时：

$$W_{\text{out}}^T = X^{\dagger} Y$$

其中伪逆$X^{\dagger}$的计算：

$$X^{\dagger} = \begin{cases}
(X^T X)^{-1} X^T & \text{if } T \geq N_r \text{ (过完备)} \\
X^T (X X^T)^{-1} & \text{if } T < N_r \text{ (欠完备)}
\end{cases}$$

#### 3.2.3 在线学习更新

对于流式数据，可使用递归最小二乘（RLS）：

$$P(t) = P(t-1) - \frac{P(t-1)x(t)x(t)^T P(t-1)}{1 + x(t)^T P(t-1)x(t)}$$

$$W_{\text{out}}(t) = W_{\text{out}}(t-1) + (y_{\text{target}}(t) - W_{\text{out}}(t-1)x(t)) x(t)^T P(t)$$

### 3.3 固定随机投影的保距性质

#### 3.3.1 Johnson-Lindenstrauss引理

**定理**（Johnson-Lindenstrauss引理）：设$x_1, \ldots, x_n \in \mathbb{R}^d$，对于任意$\epsilon \in (0,1)$，若投影维度$k$满足：

$$k \geq \frac{8\delta \log n}{\epsilon^2 - 2\epsilon^3/3}$$

则随机投影$\Pi_k: \mathbb{R}^d \to \mathbb{R}^k$以至少$1 - n(n-1)/n^{2\delta}$的概率保持所有成对距离：

$$(1-\epsilon)\|x_i - x_j\|^2 \leq \frac{d}{k}\|\Pi_k(x_i) - \Pi_k(x_j)\|^2 \leq (1+\epsilon)\|x_i - x_j\|^2$$

**关键洞察**：
- 投影维度$k$仅依赖于样本数$n$（对数级）和精度$\epsilon$
- **与原始维度$d$无关**！

#### 3.3.2 随机投影矩阵的构造

**高斯随机投影**：

矩阵$R \in \mathbb{R}^{k \times d}$的元素独立采样：

$$R_{ij} \sim \mathcal{N}(0, 1/d)$$

投影：$y = Rx \in \mathbb{R}^k$

**稀疏随机投影**：

$$R_{ij} = \begin{cases}
+\sqrt{s} & \text{概率 } 1/(2s) \\
0 & \text{概率 } 1 - 1/s \\
-\sqrt{s} & \text{概率 } 1/(2s)
\end{cases}$$

当$s = \sqrt{d}$时，计算复杂度从$O(kd)$降至$O(k\sqrt{d})$。

#### 3.3.3 保距性质的数学证明

**引理**：对于固定向量$x \in \mathbb{R}^d$，设$R \in \mathbb{R}^{k \times d}$为随机矩阵，元素$R_{ij} \sim \mathcal{N}(0, 1/d)$。则：

$$\mathbb{E}[\|Rx\|^2] = \|x\|^2$$

且：

$$\Pr\left(\left|\|Rx\|^2 - \|x\|^2\right| \geq \epsilon \|x\|^2\right) \leq 2\exp(-k\epsilon^2/8)$$

**证明**：

设$z = Rx$。由于$R_{ij}$独立高斯分布，$z_i \sim \mathcal{N}(0, \|x\|^2/d)$。

$$\|Rx\|^2 = \sum_{i=1}^k z_i^2 = \frac{\|x\|^2}{d} \sum_{i=1}^k \left(\frac{\sqrt{d} \cdot z_i}{\|x\|}\right)^2$$

其中$\frac{\sqrt{d} \cdot z_i}{\|x\|} \sim \mathcal{N}(0, 1)$，因此：

$$\frac{d \cdot \|Rx\|^2}{\|x\|^2} \sim \chi^2(k)$$

利用$\chi^2$分布的集中不等式可得结论。

#### 3.3.4 内积保持

**定理**：对于$x, y \in \mathbb{R}^d$且$\|x\|_2, \|y\|_2 \leq 1$，设$\Phi$为$k \times d$随机矩阵，元素$\sim \mathcal{N}(0, 1/d)$。则：

$$\Pr\left(\left|\frac{d}{k}(\Phi x)^T(\Phi y) - x^T y\right| \geq \epsilon\right) \leq 2\exp\left(\frac{-k\epsilon^2}{C_1 + C_2\epsilon}\right)$$

其中$C_1 \approx 2.5$，$C_2 \approx 7.7$。

这表明随机投影不仅保持距离，还**保持内积结构**，对Attention机制至关重要。

---

## 4. 与本研究的联系

### 4.1 Self-Attention作为Reservoir的理论可能性

#### 4.1.1 Attention机制的递归解释

标准Self-Attention可视为一种特殊的循环操作：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**关键观察**：

1. **状态传播**：每个token的表示通过与其他所有token交互而更新
2. **非线性变换**：Softmax提供非线性
3. **全局依赖**：所有位置两两交互

#### 4.1.2 Attention作为Reservoir的理论框架

我们可以将Self-Attention重新解释为Reservoir-like操作：

**类比映射**：

| ESN组件 | Attention对应 |
|--------|--------------|
| 输入$u(t)$ | Query $Q$ |
| 储层状态$x(t-1)$ | Key $K$, Value $V$ |
| 状态更新 | Attention输出 |
| 输出层 | 后续FFN层 |

**差异**：

1. **权重性质**：ESN使用**固定随机**权重，Attention使用**可学习**的Q、K、V投影
2. **动态性**：ESN状态随时间演化，Attention是前向的一次性操作
3. **记忆机制**：ESN有显式循环记忆，Attention依赖位置编码隐式编码顺序

### 4.2 正交随机QK投影 vs 标准Reservoir随机权重

#### 4.2.1 正交随机投影的优势

本研究提出的"正交随机注意力"与标准Reservoir的关键区别：

**标准Reservoir权重**：
- 元素独立随机采样（如$\mathcal{N}(0, \sigma^2)$）
- 需要调整谱半径$\rho(W_{\text{res}}) < 1$
- 行向量近似正交但不精确

**正交随机投影**：
- 行向量**严格正交**
- 自动满足$WW^T = I$
- 谱半径恰好为1
- 更好的数值稳定性

#### 4.2.2 数学对比

**标准随机矩阵** $W_{\text{std}} \in \mathbb{R}^{m \times n}$（$m < n$）：

$$W_{\text{std},ij} \sim \mathcal{N}(0, 1/n)$$

性质：
- $\mathbb{E}[W_{\text{std}} W_{\text{std}}^T] = I_m$
- 但实际$W_{\text{std}} W_{\text{std}}^T \neq I_m$（有随机波动）

**正交随机矩阵** $W_{\text{orth}} \in \mathbb{R}^{m \times n}$：

构造方法（如Haar测度采样）：

1. 生成随机高斯矩阵$A \in \mathbb{R}^{n \times n}$
2. QR分解：$A = QR$
3. 取$W_{\text{orth}} = Q_{1:m,:}$（前$m$行）

性质：
- $W_{\text{orth}} W_{\text{orth}}^T = I_m$（精确成立）
- 保持范数：$\|W_{\text{orth}}x\| = \|x\|$对所有$x$成立
- 最优保距性

#### 4.2.3 在Attention中的应用

**正交随机QK投影**：

$$Q = X W_Q, \quad K = X W_K$$

其中$W_Q, W_K \in \mathbb{R}^{d \times d_k}$为正交随机矩阵。

**优势**：

1. **稳定的注意力分布**：正交投影保持输入的相对几何结构
2. **避免梯度问题**：正交权重具有良好条件数
3. **理论可分析性**：正交性简化数学分析

### 4.3 将Reservoir思想扩展到Transformer

#### 4.3.1 设计原则

基于Reservoir Computing的洞见，我们提出以下设计原则：

**原则1：固定随机投影作为特征提取器**

将部分Attention层的Q、K投影设为固定随机（正交）：

$$\text{RandomAttention}(X) = \text{softmax}\left(\frac{X W_Q (X W_K)^T}{\sqrt{d_k}}\right) X W_V$$

其中$W_Q, W_K$固定随机，$W_V$可训练。

**原则2：混合架构**

借鉴ResFormer的级联思想：

- **全局层**：固定随机Attention捕捉长程依赖（类似Reservoir）
- **局部层**：可学习Attention捕捉细粒度模式

**原则3：可训练输出层**

保持Transformer的FFN层可训练，作为"读出层"：

$$\text{Output} = \text{FFN}(\text{RandomAttention}(X))$$

#### 4.3.2 理论可行性分析

**命题**：固定随机QK投影的Attention具有通用逼近能力。

**论证**：

1. 随机投影将输入映射到高维特征空间
2. Softmax核可逼近任意连续核函数（Mercer定理）
3. 可训练的V投影和FFN提供足够的表达能力
4. 结合ELM的通用逼近理论，整体架构具备通用性

#### 4.3.3 复杂度优势

| 组件 | 标准Transformer | 随机投影Transformer |
|-----|----------------|-------------------|
| Q/K投影 | $O(d^2)$参数，可训练 | $O(1)$训练参数（固定） |
| Attention计算 | $O(n^2 d)$ | $O(n^2 d_k)$（$d_k \leq d$） |
| 训练时间 | 慢（需优化所有参数） | 快（仅优化V和FFN） |
| 内存占用 | 高 | 低（无需存储Q/K梯度） |

#### 4.3.4 与现有工作的联系

**与Performer的联系**：

Performer使用随机特征近似Softmax Attention：

$$\exp(q_i^T k_j / \sqrt{d}) \approx \phi(q_i)^T \phi(k_j)$$

其中$\phi$为随机特征映射。

我们的方法与之互补：
- Performer：随机化核函数
- 我们的方法：随机化投影矩阵

**与Linformer的联系**：

Linformer将Attention矩阵低秩近似：

$$\text{Attention} \approx \text{softmax}(Q E^T F K^T)V$$

其中$E, F \in \mathbb{R}^{n \times k}$为可学习投影（$k \ll n$）。

我们的方法使用固定随机投影，无需学习，进一步降低复杂度。

---

## 5. 总结与展望

### 5.1 核心发现

1. **Reservoir Computing的理论基础坚实**：ESN的ESP性质和Universal Approximation定理为固定随机权重+可训练输出层范式提供了严格数学保证。

2. **随机投影的保距性质关键**：Johnson-Lindenstrauss引理解释了为什么随机投影能保持数据结构，这对Attention机制同样适用。

3. **ResFormer验证了混合架构的可行性**：Reservoir与Transformer的级联结合在长序列任务上取得显著突破。

4. **正交随机投影优于标准随机权重**：严格的正交性带来更好的数值稳定性和理论可分析性。

### 5.2 本研究的理论支撑

本报告为正交随机注意力Transformer提供了以下理论支撑：

1. **数学基础**：随机投影的保距性质保证了Attention分布的稳定性
2. **表达能力**：ELM和ESN的通用逼近理论支持固定随机投影的表达能力
3. **效率优势**：固定权重显著减少可训练参数，加速训练
4. **架构借鉴**：ResFormer的成功经验可直接应用

### 5.3 未来研究方向

1. **理论深化**：证明正交随机Attention的Universal Approximation性质
2. **架构优化**：探索最优的固定随机与可学习层混合比例
3. **任务扩展**：验证在视觉、语音等其他模态的有效性
4. **硬件协同**：利用固定权重的特性设计专用加速器

---

## 参考文献

1. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
2. Jaeger, H., & Haas, H. (2004). Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication.
3. Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training.
4. Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006). Extreme learning machine: Theory and applications.
5. Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space.
6. Liu, J., et al. (2025). ResFormer: All-Time Reservoir Memory for Long Sequence Classification. arXiv:2509.24074.
7. Gallicchio, C., Micheli, A., & Pedrelli, L. (2017). Deep reservoir computing: A critical experimental analysis.
8. Grigoryeva, L., & Ortega, J. P. (2018). Echo state networks are universal.
9. Choromanski, K., et al. (2020). Rethinking attention with performers.
10. Wang, S., et al. (2020). Linformer: Self-attention with linear complexity.

---

*报告生成时间：2025年*
*本报告为"正交随机注意力Transformer"研究项目提供理论支撑*
