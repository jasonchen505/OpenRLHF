# LLM 后训练 & Agent RL 技术面试五类问题深度应对指南

> 基于 OpenRLHF / slime / verl / ROLL 四大框架的源码级分析
> 按面试五类能力维度组织：底层原理 → 实验验证 → 问题定位 → 工程落地 → 业务场景

---

## 目录

1. [第一类：底层原理深入理解](#第一类底层原理深入理解)
2. [第二类：实验与方案验证能力](#第二类实验与方案验证能力)
3. [第三类：问题定位能力](#第三类问题定位能力)
4. [第四类：工程落地能力](#第四类工程落地能力)
5. [第五类：业务与实际场景理解](#第五类业务与实际场景理解)
6. [综合模拟面试](#综合模拟面试)

---

# 第一类：底层原理深入理解

> **考察重点**: 不是回答清楚概念，而是讲清楚"这个方法解决什么问题、存在哪些局限性、有哪些改进方法"

---

## 1.1 PPO 在 LLM 场景下为什么这么设计？解决了什么问题？有什么局限？

### 解决的核心问题

PPO (Proximal Policy Optimization) 解决的是 **策略梯度更新步长难以控制** 的问题。原始 REINFORCE 的梯度更新没有约束，一步走偏就可能导致策略崩溃(collapse)。PPO 的核心贡献是通过 **clipped surrogate objective** 限制每步策略更新幅度，实现"信任域"优化。

**具体到 LLM 场景**，PPO 解决了三个问题：
1. **训练不稳定**: LLM 的动作空间是整个词表(~100K)，单步梯度更新可能改变大量 token 的概率分布。Clipping 约束更新幅度。
2. **Sample efficiency**: 通过 importance sampling，同一批 rollout 数据可以复用多个 epoch(PPO 的 "P" 就是 Proximal)，不需要像 vanilla REINFORCE 那样用完就丢。
3. **Credit assignment**: 通过 GAE (Generalized Advantage Estimation) 将稀疏的 sequence-level reward 分配到每个 token，解决"哪个 token 贡献了最终 reward"的问题。

### 设计细节与原因

**代码级理解** (`openrlhf/models/loss.py:166-187`):
```python
raw_policy_log_ratio = log_probs - old_log_probs
policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
ratio = policy_log_ratio.exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - eps_low, 1 + eps_high) * advantages
loss = -torch.min(surr1, surr2)
```

- **为什么用 log_ratio 而不是直接算 ratio?** — 数值稳定性。`exp(a-b)` 在 `a-b` 很大时溢出，但在 log 空间可以 clamp 到 [-20, 20]，`exp(20)≈4.8e8` 足够大但不会 inf。
- **为什么是 min 而不是 max?** — PPO 是最大化目标但用 loss 最小化框架。`min(surr1, surr2)` 取两个 surrogate 的下界，这是 **悲观估计**：当 ratio 偏离 1 时，取更保守的那个方向。
- **Dual-clip 的存在原因** (`loss.py:188-194`): 当 advantage<0 时，标准 PPO 允许 ratio 下降到 `1-ε`，但 dual-clip 加了下界 `dual_clip*advantage`(dual_clip>1)，防止策略"过度否定"某些 token。

### 局限性

1. **Critic 瓶颈**: PPO 需要 Critic 网络做 GAE，但 Critic 通常和 Actor 一样大(甚至更大因为要输出 per-token value)。对于 7B 模型，需要额外 7B 的 Critic 显存。
2. **On-policy 限制**: 每次策略更新后，旧的 rollout 数据就"过期"了(因为 ratio ≠ 1)。这导致 **sample efficiency 很低** — 推理成本高但数据只能用 1-2 个 epoch。
3. **GAE 的 bias-variance 困境**: λ=0 时低方差但高偏差(TD(0))，λ=1 时低偏差但高方差(MC)。LLM 的 response 长度变化大(10-2000 tokens)，固定 λ 无法适配所有长度。
4. **Clipping 是"一刀切"**: 全局统一的 ε_clip 不区分 token 重要性。重要的 reasoning token 和不重要的 filler token 受到同样的 clipping 约束。

### 改进方向

| 问题 | 改进方案 | 框架支持 |
|------|---------|---------|
| Critic 显存开销 | GRPO/REINFORCE++ (去掉 Critic) | OpenRLHF/verl/ROLL 全部支持 |
| Off-policy 数据利用 | IS correction (TIS/ICEPOP) | OpenRLHF loss.py |
| Token 重要性不均 | GSPO (sequence-level ratio) | verl core_algos.py |
| 策略熵崩溃 | Dual-clip, Entropy bonus | OpenRLHF PolicyLoss |
| 生成成本高 | Async training (overlap gen+train) | OpenRLHF ppo_trainer_async.py |

---

## 1.2 GRPO 为什么不需要 Critic？它的方差问题怎么理解？

### 设计动机

GRPO (Group Relative Policy Optimization) 的核心洞察：**同一个 prompt 生成的多个 response 之间的 reward 差异本身就是 advantage 的估计**。

```
prompt → [response_1, response_2, ..., response_N]
reward = [r_1, r_2, ..., r_N]
advantage_i = (r_i - mean(r)) / std(r)
```

这本质上是用 **Monte Carlo 采样估计 advantage**，代替了 PPO 中 Critic 网络的 **bootstrap 估计**。

### 解决了什么问题

1. **省显存**: 不需要 Critic 网络。对于 7B 模型 PPO 需要 4 个模型(Actor+Critic+Ref+Reward)，GRPO 只需要 3 个(Actor+Ref+Reward)，显存节省约 25%。
2. **训练简单**: 不需要训练 Critic(本身就不稳定)，不需要 GAE 的超参数(γ, λ)。
3. **适合 reasoning 任务**: 数学推理的 reward 通常是二值的(对/错)，组内统计量天然提供了有意义的 baseline。

### 方差问题深度分析

**核心问题**: 当组内 reward 方差很小时(例如全部答对或全部答错)，std→0，advantage 趋向无穷大或全为零。

**代码中的处理** (`experience_maker.py`):
```python
rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
```

`1e-9` 的 epsilon 防止除零，但当 std 接近 0 时，advantage 被放大到极大值，梯度爆炸。

**三种缓解方案**:

1. **DAPO Dynamic Filtering** (`samples_generator.py:169-195`): 过滤掉全对/全错的 prompt 组。如果 `avg_score > max_threshold` 或 `avg_score < min_threshold`，整组丢弃并替换新 prompt。代价是推理浪费。

2. **Dr. GRPO**: 去掉 std normalization，只减 mean。`advantage_i = r_i - mean(r)`。好处是不存在除零问题；坏处是 advantage 的 scale 随 reward 绝对值变化。

3. **n_samples_per_prompt 增大**: 更多 sample → 更稳定的统计量。但推理成本线性增长。实践中 8-16 是常用范围。

### 与其他无 Critic 方法的对比

| 方法 | Baseline 来源 | 方差控制 | 代码位置 |
|------|-------------|---------|---------|
| GRPO | 组内 mean+std normalization | std→0 时爆炸 | `experience_maker.py:284-291` |
| REINFORCE++-baseline | 组内 mean subtraction | 更鲁棒 | `experience_maker.py:292-308` |
| RLOO | Leave-one-out mean | 消除自相关 | `experience_maker.py:265-271` |
| Dr. GRPO | 组内 mean, 无 std norm | 不会爆炸 | `experience_maker.py:303-308` |

---

## 1.3 KL 散度在 RLHF 中到底起什么作用？放在 reward 里和放在 loss 里有什么区别？

### 核心作用

KL 散度是 **策略正则化器**：防止优化 reward 的过程中，策略偏离 reference policy 太远，导致"reward hacking"——模型找到 reward function 的漏洞获得高分但实际质量下降。

**类比**: 就像 L2 正则化防止模型过拟合训练集，KL 正则化防止策略过拟合 reward model。

### 两种实现方式的区别

**方式一：放在 reward 里 (reward shaping)** — OpenRLHF 的默认方式
```python
# experience_maker.py
kl_reward = -kl_coef * kl                          # 每个 response token
last_reward = scatter(reward, eos_indices)          # 外部 reward 放在 EOS
token_reward = last_reward + kl_reward              # 合并
```

- KL 作为 per-token reward penalty 参与 GAE 计算
- 通过 advantage 间接影响梯度
- 可以有 token 级别的差异(某些 token 的 KL 大就多惩罚)

**方式二：放在 loss 里 (explicit regularization)**
```python
loss = policy_loss + kl_coef * kl_loss
```

- KL 作为独立的 loss 项直接加到总 loss
- 不经过 GAE，直接贡献梯度
- 通常是 sequence-level 或 batch-level 的 KL

**实际差异**:
- Reward shaping 更"柔和"，KL 通过 advantage 的 discounted return 影响整个 future trajectory
- Explicit loss 更"直接"，KL 只影响当前 token 的梯度
- OpenRLHF 默认用 reward shaping，但 `--algo.kl.use_loss` 可以切换到 explicit loss

### 三种 KL 估计器的选择

```python
# k1: 无偏但可能为负 (逐 token)
kl = log(π/π_ref)

# k2: 非负但有偏 (平方近似)
kl = (log(π/π_ref))² / 2

# k3: 非负且无偏 (reverse KL 的 Taylor 展开)
kl = exp(log π_ref - log π) - 1 - (log π_ref - log π)
```

**实际使用建议**: k3 最稳定。k1 可能为负导致"反向激励"(KL 大的 token 反而获得正 reward)，k2 的平方会放大异常值。

### Adaptive KL Controller

```python
# kl_controller.py
proportional_error = clip(KL_current / KL_target - 1, -0.2, 0.2)
kl_coef *= (1 + proportional_error * n_steps / horizon)
```

当实际 KL 超过 target 时增大惩罚系数，反之减小。`[-0.2, 0.2]` 的 clipping 防止调整过激。

---

## 1.4 DPO 和 PPO 的本质区别是什么？DPO 的隐式 reward 是什么？

### 理论本质

**PPO** 是在线强化学习：策略在环境中交互采样，用 reward signal 更新策略。
**DPO** 是离线偏好学习：直接从 chosen/rejected 对中学习，不需要 reward model。

**DPO 的推导关键**: 从 Bradley-Terry 偏好模型出发：
```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
```
其中 `r(x,y)` 是 reward function。DPO 的核心洞察是：**最优策略和 reward function 之间存在闭式解**：
```
r(x, y) = β * log(π*(y|x) / π_ref(y|x))
```

这就是 DPO 的 **隐式 reward** — reward 由策略和 reference 的 log-ratio 定义。

### DPO Loss 实现

```python
# loss.py:DPOLoss
logits = (policy_chosen_logps - policy_rejected_logps) - \
         (reference_chosen_logps - reference_rejected_logps)
losses = -F.logsigmoid(beta * logits)
```

`logits` 代表隐式 reward margin。当 `logits > 0` 时，策略给 chosen 的相对概率高于 reference，说明策略学到了偏好。

### 何时用 DPO vs PPO

| 场景 | 推荐 | 原因 |
|------|------|------|
| 有高质量 preference 数据 | DPO | 简单稳定，不需要在线采样 |
| 有 reward function(可编程) | PPO | 可以在线探索，不受数据分布限制 |
| 需要 tool use / multi-turn | PPO | 环境交互是必须的 |
| 计算资源有限 | DPO | 不需要推理引擎 |
| Reasoning 任务(数学/代码) | PPO+GRPO | Reward 是 verifiable 的，在线探索更有效 |
| 对话质量(主观) | DPO | Human preference 数据更直接 |

### DPO 的局限

1. **Distribution shift**: DPO 只在 chosen/rejected 数据上训练，策略可能从未探索过空间中的其他区域。PPO 在线采样可以探索。
2. **Overfitting to preference**: 当 β 太小时，策略可能通过"记忆"preference 对来获得高隐式 reward，但泛化差。IPO 用 MSE loss 缓解。
3. **No iterative refinement**: DPO 通常只训一轮；PPO 可以多轮迭代，每轮用新策略采样。

---

## 1.5 重要性采样修正 (IS Correction) 解决什么问题？三种方式的区别？

### 问题来源

在异步训练中(如 OpenRLHF 的 `PPOTrainerAsync`)，generation 和 training 并行运行。vLLM 生成样本时用的权重可能比当前训练权重旧几个 step。

```
vLLM rollout: π_{t-3}(a|s)  ← 旧权重
Actor training: π_t(a|s)    ← 当前权重
old_log_probs: π_{t-1}(a|s) ← 上一步的权重
```

如果不修正，PPO 的 ratio `π_t/π_{t-1}` 会低估 off-policy 程度，因为实际的 off-policy ratio 应该是 `π_t/π_{t-3}`。

### 三种修正方式

```python
# TIS (Truncated IS): 软截断
vllm_is = exp(old_log_probs - rollout_log_probs).clamp(low, high)
loss = vllm_is * loss

# ICEPOP: 硬过滤
mask = (vllm_is >= low) & (vllm_is <= high)
loss = (vllm_is * mask) * loss

# seq-mask-tis: 序列级判断 + token级修正
seq_is = exp(mean(log_ratio))  # 序列级
seq_mask = (seq_is >= low) & (seq_is <= high)  # 过滤
vllm_is = exp(log_ratio).detach()  # token级修正
loss = seq_mask.unsqueeze(-1) * vllm_is * loss
```

**TIS**: 每个 token 的 IS weight 被 clamp 到 [low, high]。保留所有 token 但限制极端值。
**ICEPOP**: 超出范围的 token 直接被 mask 掉(不贡献 loss)。更激进，丢弃 off-policy 程度大的 token。
**seq-mask-tis**: 先用序列级几何均值判断整条序列是否"可信"，不可信的整条序列被丢弃。然后对可信序列用 token 级 IS 修正。

### 如何选择

- **同步训练** (colocate): 不需要 IS correction，rollout 和 training 用同一权重。
- **轻度异步** (1-2 step delay): TIS 就够了。
- **重度异步** (多 step delay): seq-mask-tis 更安全，避免逐 token 判断的噪声。
- **Partial rollout 模式**: ICEPOP 最激进，因为 partial rollout 的 off-policy 程度最大。

---

## 1.6 GSPO (Group Sequence Policy Optimization) 为什么要做 sequence-level ratio？

### 问题：token-level clipping 的不公平

标准 PPO 对每个 token 独立做 ratio clipping。但在 LLM 中，一个 response 的质量是整体的——某些 token 是"关键推理 token"(如数学符号)，某些是"filler token"(如"让我们来看看")。对它们施加同样的 clipping 是不公平的。

### GSPO 的解决方案

```python
# loss.py:170-178
log_ratio = log_probs - rollout_log_probs
ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
ratio = ratio.exp().unsqueeze(-1) * action_mask
```

计算 **几何均值** 的 token-level ratio: `r_seq = exp(mean_t(log π_t/π_old_t))`，然后用这个序列级 ratio 做 clipping。

**效果**: 整条序列要么"被接受"(ratio 在 clip 范围内)，要么"被拒绝"(ratio 超出范围)。避免了某些 token 被 clip 而其他 token 不被 clip 的不一致。

### 局限

强制 `token_level_loss=False`，即每条序列的 loss 先平均再 batch 平均。这可能放大长短序列的权重差异。

---

## 1.7 Reward Model 为什么在 EOS 位置提取 reward？Critic Model 为什么要输出 per-token value？

### Reward Model: EOS 提取

```python
# model.py
values = hidden_states[:, -1, :]  # 最后一个 token 的 hidden state
reward = self.value_head(values)   # Linear(hidden_size, 1)
```

**原因**: Causal attention 下，最后一个 token 的 hidden state 包含了整个序列的信息(因为它 attend to 所有前面的 token)。用它作为整个 response 的 representation 是自然的。

**局限**: 如果 response 很长(如 2000 tokens)，最后 token 的 hidden state 可能"遗忘"了前面的关键信息(attention 分散)。一些改进方案用 pooling 或加权。

### Critic Model: Per-token Value

Critic 需要输出每个 token 的 value `V(s_t)`，因为 GAE 的公式需要 per-token value:
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t = δ_t + γλA_{t+1}
```

如果 Critic 只输出一个标量(像 Reward Model 那样)，GAE 就无法做 token 级的 credit assignment。

---

# 第二类：实验与方案验证能力

> **考察重点**: 不仅关注做了什么，更关注怎么证明它是有效的。追问实验细节能看出是否有真正深入理解。

---

## 2.1 如何验证 PPO/GRPO 训练是否有效？

### 核心监控指标体系

**第一层：训练信号指标** (每个 step 监控)

| 指标 | 健康范围 | 异常信号 | 含义 |
|------|---------|---------|------|
| `policy_loss` | 缓慢下降 | 突然飙升或趋近0 | 策略优化目标 |
| `ppo_clip_ratio` | 0.1-0.4 | >0.7 或 <0.05 | 被 clip 的 token 比例 |
| `ppo_kl` | 接近 target KL | 远偏离 target | 策略与旧策略的 KL |
| `kl` | 在 target 附近 | 持续增长 | 策略与 reference 的 KL |
| `entropy` | 缓慢下降 | 暴跌 | 策略的探索程度 |

**第二层：rollout 质量指标** (每个 rollout batch 监控)

| 指标 | 健康趋势 | 异常信号 | 含义 |
|------|---------|---------|------|
| `rollout/reward_mean` | 缓慢上升 | 突然跳变或饱和 | 模型能力提升 |
| `rollout/reward_std` | 稳定 | 趋近0 | reward 方差消失 |
| `rollout/response_length_mean` | 稳定或缓慢变化 | 持续增长 | 长度偏好/奖励破解 |
| `rollout/truncated_rate` | <20% | >50% | 模型不会停止 |

**第三层：eval 指标** (定期评估)

| 指标 | 健康趋势 | 关注点 |
|------|---------|-------|
| `eval_pass@k` | 上升 | 模型能力的真实提升 |
| `eval_response_length_mean` | 稳定 | 不应为提升 pass@k 而变长 |
| `dynamic_filtering_pass_rate` | 20-80% | 过滤率太高说明数据太简单/太难 |

### 验证实验设计

**Ablation 1: 算法对比**
```
同一数据集 + 同一模型 + 同一 prompt
├── PPO (GAE, critic)
├── GRPO (group_norm, n=8)
├── REINFORCE++-baseline (n=8)
└── RLOO (n=8)
比较: eval_pass@1, 训练稳定性, 显存占用
```

**Ablation 2: n_samples_per_prompt 的影响**
```
GRPO with n ∈ {2, 4, 8, 16, 32}
比较: reward 曲线方差, 收敛速度, 推理成本/收益
```

**Ablation 3: KL coefficient 的影响**
```
kl_coef ∈ {0.001, 0.01, 0.05, 0.1}
比较: KL 轨迹, reward vs KL 的 Pareto 曲线
```

### 实验细节——面试官可能追问的点

**Q: 你的 eval 指标是什么？为什么选 pass@k 而不是 reward mean?**

"A: Reward mean 可能被 reward hacking 操纵——模型可能学会利用 reward function 的漏洞。Pass@k 是外部 verifiable metric(如数学题的答案匹配)，不受 reward hacking 影响。我会同时监控两者：如果 reward 上升但 pass@k 不变或下降，说明发生了 reward hacking。"

**Q: 你怎么判断是模型能力提升还是 reward hacking?**

"A: 三个信号：(1) reward 上升但 eval pass@k 不变——reward hacking；(2) response_length 持续增长——模型在'凑字数'获取 reward；(3) KL 持续增长——策略偏离 reference 太远，可能在 reward boundary 外推。"

**Q: clip_ratio 太高怎么办？**

"A: 说明策略更新太激进。可以：(1) 减小 learning rate；(2) 减小 PPO epoch 数(从 2 降到 1)；(3) 增大 clip_eps(从 0.2 到 0.3 给更多空间)；(4) 检查 advantage normalization 是否正确——极端 advantage 值会导致 ratio 偏移。"

---

## 2.2 如何验证 DAPO Dynamic Filtering 的有效性？

### 实验设计

```
对照组: 不使用 dynamic filtering
实验组: filtering range ∈ {(0, 0.8), (0.1, 0.9), (0.2, 0.8)}
比较:
  1. 训练效率 (达到相同 pass@k 的 step 数)
  2. Filter 通过率 (太高=数据太简单，太低=数据太难)
  3. 每 step 的有效样本数
```

### 关键代码路径

```python
# samples_generator.py:169-195
for batch in experiences_per_prompt:
    avg_score = mean(scores)
    if avg_score < min_score or avg_score > max_score:
        discard(batch)
        dispatch_replacement_prompt()
```

### 面试深挖

**Q: Filtering 会不会导致训练数据分布偏移？**

"A: 会。Dynamic filtering 本质上是在做 curriculum learning——只保留'有挑战但可学'的样本。这会导致训练数据分布和原始数据集不同。缓解方法：(1) 定期调整 filtering range；(2) 监控 filter 通过率，如果持续下降说明模型在进步，需要更新 range；(3) 保留一定比例的'太简单'样本防止遗忘。"

**Q: Replacement prompt 从哪里来？**

"A: 从同一个 dataloader 中取下一个 prompt。如果 dataloader 被 exhausted，剩余的 pending Ray refs 会被 cancel。这意味着 effective batch size 可能小于配置值——需要监控 `num_samples` 指标。"

---

## 2.3 如何验证序列长度平衡对训练效率的影响？

### 实验设计

```
对照组: 不使用 dynamic batching (round-robin 分配)
实验组: 使用 seqlen_balancing (Karmarkar-Karp 算法)
比较:
  1. timing/step_total
  2. GPU 利用率 (nvidia-smi)
  3. 每 step 的有效 token 数
```

### 关键代码路径

```python
# seqlen_balancing.py
batch_indexes = get_seqlen_balanced_partitions(total_lengths, num_batch, False)
```

### 面试深挖

**Q: 为什么用 Karmarkar-Karp 而不是简单的 greedy?**

"A: Greedy(最长优先分配到最少负载的 bin)在对抗性输入下可能很差。KK 算法通过 pairing largest with smallest 的 merge 策略，产出更均衡的 partition。实验上 KK 的 partition 方差比 greedy 小 20-30%。但 KK 的实现更复杂——在 OpenRLHF 中两种都实现了，可以对比。"

**Q: 平衡后 DP rank 之间的 token 数差异有多大？**

"A: 未平衡时差异可以达到 3-5x(一个 rank 全是长序列，另一个全是短序列)。平衡后差异通常 <10%。对于 8 GPU 的 DP，这意味着最慢的 rank 的等待时间从 5x 降到 1.1x。"

---

## 2.4 如何验证 weight 同步的正确性？

### 实验设计

```
1. 训练 1 个 step
2. 把 Actor 的权重和 vLLM 的权重分别取出
3. 逐参数比较 diff
4. diff 应该在数值精度范围内 (<1e-6 for fp32, <1e-3 for bf16)
```

### slime 的 `check-weight-update-equal` 功能

slime 提供了内置的权重一致性验证 CI 测试。每次 weight sync 后自动检查 vLLM engine 的权重版本号是否匹配训练侧。

### 面试深挖

**Q: ZeRO-3 下 weight 同步有什么特殊问题？**

"A: ZeRO-3 下每个 rank 只持有参数的 1/N 分片。要 broadcast 完整参数，需要先 allgather 到 rank 0，再 broadcast 给 vLLM。通信量 = 模型大小 × 2(BF16)。这是 ZeRO-3 在 RLHF 场景的主要瓶颈，所以 OpenRLHF 默认用 ZeRO-2。"

**Q: 如果 weight 同步失败了怎么办？**

"A: OpenRLHF 没有重试机制——如果 NCCL broadcast 失败，整个训练崩溃。只能靠 checkpoint 恢复。所以 `--ckpt.save_steps` 要设得足够小(如每 10-20 步)来限制回滚成本。"

---

# 第三类：问题定位能力

> **考察重点**: 模型上线后能力突然下降，系统突然变慢，实验结果和预期不一致——怎么排查？讲清楚优化思路与解决方案。

---

## 3.1 训练 loss 突然飙升 (NaN/Inf)

### 排查思路

```
Loss NaN/Inf
├── 1. 检查 reward 分布 → reward 是否有极端值?
├── 2. 检查 advantage → 是否有极大/极小值?
├── 3. 检查 gradient norm → 是否爆炸?
├── 4. 检查 KL → 是否趋近0或无穷?
└── 5. 检查 ratio → 是否被 clamp 到 [-20, 20]?
```

### 代码级定位

**Step 1: 检查 reward**
```python
# 经验: reward 超过 ±10 通常是异常的
# OpenRLHF 有 reward clipping: --reward.clip_range -10 10
```

**Step 2: 检查 advantage normalization**
```python
# experience_maker.py:315-328
mean = (all_advantages * all_action_masks).sum() / num_actions
var = ((all_advantages - mean).pow(2) * all_action_masks).sum() / num_actions
rstd = var.clamp(min=1e-8).rsqrt()
```
如果 `num_actions` 为 0(所有 token 被 mask 掉)，`mean` 和 `var` 都是 NaN。检查 `action_mask` 是否全为 0。

**Step 3: 检查 gradient norm**
```python
# ActorPPOTrainer.ppo_train() 中 DeepSpeed 会 log grad_norm
# grad_norm > 10 通常是异常的
# OpenRLHF 默认 max_norm=1.0 做 gradient clipping
```

**Step 4: 检查 loss 函数的数值保护**
```python
# loss.py:168
policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
```
如果看到大量 token 的 log_ratio 被 clamp 到 ±20，说明策略偏移严重——可能是 learning rate 太大或 PPO epoch 太多。

### 常见原因与解决方案

| 原因 | 诊断信号 | 解决方案 |
|------|---------|---------|
| Reward 有极端值 | reward std 异常大 | 加 reward clipping |
| Advantage 未正确 normalize | advantage 值在 ±100 范围 | 检查 normalization 代码 |
| Learning rate 太大 | grad_norm 持续 > 5 | 降低 lr 或加大 gradient clipping |
| PPO epoch 太多 | clip_ratio 持续 > 0.8 | 减少 epoch 数 |
| 数据有 bug | 某些 sample 的 token 全是 padding | 检查数据预处理 |

---

## 3.2 Eval pass@k 突然下降

### 排查思路

```
Eval pass@k 下降
├── 1. 检查 checkpoint → 是否加载了错误的 checkpoint?
├── 2. 检查 KL → 策略是否偏离 reference 太远?
├── 3. 检查 response_length → 是否变短/变长了?
├── 4. 检查 reward vs pass@k → 是否 reward hacking?
└── 5. 检查 eval 数据 → eval 数据是否和训练数据分布一致?
```

### 关键诊断

**Reward Hacking 判断**:
```
如果 reward 持续上升但 pass@k 不变或下降:
→ 模型在利用 reward function 的漏洞
→ 需要修改 reward function 或增大 KL penalty
```

**KL 爆炸判断**:
```
如果 KL 持续增长超过 target:
→ 策略偏离 reference 太远
→ 增大 kl_coef 或检查 adaptive KL controller 是否正常工作
```

**长度偏移判断**:
```
如果 response_length 显著变化:
→ 模型可能在"凑字数"获取长度相关的 reward
→ 启用 DAPO overlong penalty 或 ProRL stop properly penalty
```

### 代码级定位

```python
# 长度惩罚: length_penalty.py
# DAPO: 超过 expected_len 的部分线性惩罚
penalty = -min(exceed_len, buffer_len) / buffer_len * penalty_factor

# ProRL: 截断的 response 直接设为负 reward
if truncated:
    reward = stop_properly_penalty_coef  # 负值
```

---

## 3.3 训练速度突然变慢

### 排查思路

```
Training 变慢
├── 1. 检查 timing breakdown → 哪个阶段变慢了?
│   ├── timing/generation → vLLM 推理变慢
│   ├── timing/make_experience → RM/Critic/Ref forward 变慢
│   ├── timing/ppo_train → Actor/Critic 训练变慢
│   └── timing/broadcast → Weight 同步变慢
├── 2. 检查 GPU 利用率 → 是否有 GPU 空闲?
├── 3. 检查网络带宽 → 是否有通信瓶颈?
└── 4. 检查显存 → 是否触发了 swap?
```

### 代码级定位

**vLLM 推理变慢** (`timing/generation`):
```
可能原因:
1. Response 变长了 → 检查 rollout/response_length_mean
2. vLLM KV cache 不足 → 检查 gpu_memory_utilization
3. vLLM 引擎 hang → 检查 get_num_unfinished_requests()
4. DAPO filtering 导致重复生成 → 检查 dynamic_filtering_pass_rate
```

**Make experience 变慢** (`timing/make_experience`):
```
可能原因:
1. RM 推理变慢 → 检查 RM 模型大小和 batch size
2. Actor forward 变慢 → 检查序列长度是否增长
3. 同步等待(colocate) → 检查 empty_cache 调用频率
```

**Broadcast 变慢** (`timing/broadcast`):
```
可能原因:
1. 网络带宽下降 → 检查 NCCL 带宽
2. ZeRO-3 allgather 开销 → 切换到 ZeRO-2
3. vLLM engine 数量增加 → 通信扇出增大
```

### 面试深挖

**Q: 你怎么定位是 vLLM 的问题还是训练侧的问题？**

"A: 看 timing breakdown。如果 `timing/generation` 占比突然增大(正常应该和 `timing/ppo_train` 差不多)，说明是 vLLM 的问题。进一步用 `nvidia-smi` 看 GPU 利用率——如果 vLLM 的 GPU 利用率低但 generation 时间长，可能是 vLLM 引擎 hang 了。"

**Q: vLLM 引擎 hang 了怎么办？**

"A: OpenRLHF 没有自动重启机制。只能手动 kill 重启训练进程，从最近的 checkpoint 恢复。生产环境建议：(1) 加外部 watchdog 监控 GPU 利用率；(2) 设小的 `save_steps`；(3) slime 和 ROLL 有健康监控和自动引擎恢复机制。"

---

## 3.4 异步训练中的 Off-Policy 问题排查

### 诊断信号

```
IS 修正指标:
├── vllm_kl 持续增大 → rollout 权重越来越旧
├── IS weight 被大量 clamp/mask → off-policy 程度大
└── training 不稳定(loss 震荡) → IS 修正不够
```

### 代码级定位

```python
# loss.py:219
vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)
```

如果 `vllm_kl > 0.5`，说明 rollout 权重和当前权重差异很大。增大 `async_queue_size`(减慢 trainer 消费速度)或切换到同步模式。

### 面试深挖

**Q: 异步训练的 throughput 提升多少？代价是什么？**

"A: OpenRLHF 的 async 模式可以提升 30-50% throughput(因为 generation 和 training 重叠)。代价是 off-policy 程度增大，需要 IS 修正。在 Partial Rollout 模式下(不加锁)，throughput 可以再提升 20%，但 off-policy 程度更大，训练更不稳定。"

---

## 3.5 多轮 Agent RL 中的常见问题

### 问题 1: 所有 response 的 reward 都是 0

```
排查:
1. 检查 reward function 是否正确加载 → import 是否成功?
2. 检查 reward function 的输入格式 → queries/prompts/labels 是否匹配?
3. 检查环境 step() 是否返回了正确的 reward
4. 检查 action_ranges 是否正确 → loss 计算是否覆盖了正确的 token?
```

### 问题 2: 多轮对话的 tokenization 不一致

```
排查:
1. slime 的 TrajectoryManager 有 CLEAN/REALIGN/FORK 三种策略
2. 检查 BPE 的上下文依赖 → 不同拼接方式可能产生不同 token
3. 用 tokenizer.decode(tokenizer.encode(text)) == text 验证一致性
```

### 问题 3: Token 预算管理失败

```
排查:
1. 检查 max_length 设置 → 是否太小导致多轮还没完成就截断?
2. 检查每轮的 observation 长度 → 环境反馈是否太长?
3. 检查 truncated_rate → 高 truncation 说明预算不够
```

---

# 第四类：工程落地能力

> **考察重点**: 理论结合实际，真正落地生产价值过程中，算法怎么部署，系统保持稳定，上线后怎么保证数据回滚与监控。

---

## 4.1 如何在有限 GPU 上部署完整的 RLHF 系统？

### 资源需求分析

**标准 RLHF (PPO) 需要 4 个模型**:
```
Actor (7B):     ~14 GB (BF16) + optimizer ~42 GB (AdamW)
Critic (7B):    ~14 GB + optimizer ~42 GB
Reference (7B): ~14 GB (frozen, no optimizer)
Reward (7B):    ~14 GB (frozen, no optimizer)
vLLM (7B):      ~14 GB + KV cache ~8 GB
总计: ~162 GB (单卡放不下)
```

### 三种部署方案

**方案 A: Hybrid Engine + Sleep/Wake (8×A100 80GB)**

```bash
--train.colocate_all \
--vllm.enable_sleep \
--ds.enable_sleep \
--ds.zero_stage 2 \
--vllm.gpu_memory_utilization 0.5
```

所有模型共享 8 张 GPU，通过时间片轮转：
```
时间线:
[生成阶段] vLLM wake → Actor/Ref/RM sleep → vLLM 生成
[训练阶段] vLLM sleep → Actor reload → Actor train → Actor offload
           → Critic reload → Critic train → Critic offload
[同步阶段] vLLM wake_weights → broadcast → vLLM wake_kv_cache
```

**优势**: 8 卡即可运行完整 PPO。
**劣势**: throughput 低(模型交替占用 GPU)，每个 step 时间是分离部署的 2-3x。

**方案 B: 分离部署 (16×A100 80GB)**

```
GPU 0-7:  Actor + Critic (DeepSpeed ZeRO-2)
GPU 8-15: vLLM (4 engines × TP=2) + Ref + Reward
```

**优势**: 高 throughput。
**劣势**: 需要更多 GPU，跨节点通信。

**方案 C: GRPO 去掉 Critic (8×A100 80GB)**

```bash
--algo.advantage.estimator group_norm \
--rollout.n_samples_per_prompt 8
```

省掉 Critic 后显存减少 ~40%，8 卡可以轻松放下 Actor + Ref + Reward + vLLM。

### 面试深挖

**Q: 如果只有 4 张 A100 80GB，怎么跑 7B 的 RLHF？**

"A: 用 GRPO(去 Critic) + QLoRA(4-bit 量化 Actor/Ref) + Hybrid Engine。具体：Actor 和 Ref 用 4-bit 量化，每个模型只需 ~4GB；vLLM 用 BF16 推理(~14GB)；Reward Model 也用 4-bit。4 张 80GB 的卡可以放得下。但训练精度会下降，因为 4-bit 的 log prob 精度有限。"

**Q: sleep/wake 的切换开销有多大？**

"A: 取决于模型大小。7B 模型 offload 到 CPU 大约 2-3 秒(14GB BF16)，reload 回 GPU 大约 2-3 秒。每个 step 需要 2 次 offload(Actor + Critic)和 2 次 reload，总共约 10 秒。如果 PPO 训练本身只需 5 秒，那 sleep/wake 的开销是训练时间的 200%。这就是为什么大规模训练更推荐分离部署。"

---

## 4.2 训练中断后如何恢复？什么状态可以恢复，什么不能？

### 可恢复的状态

```python
# DeepSpeed checkpoint 保存:
model.save_checkpoint(save_dir, tag=tag, client_state={
    "episode": episode,
    "global_step": global_step,
    "total_consumed_prompts": total_consumed_prompts,
    "data_loader_state_dict": dataloader.state_dict(),
    "best_eval_metric_key": best_metric_key,
    "best_eval_metric_value": best_metric_value,
})
```

- **模型权重**: DeepSpeed checkpoint 格式
- **优化器状态**: Adam 的 m/v buffers
- **学习率调度器**: 当前 lr 和 step
- **训练进度**: episode, global_step, consumed_prompts
- **数据加载器状态**: 精确恢复到中断时的数据位置
- **Best checkpoint 追踪**: 最佳 eval metric 的 key/value

### 不可恢复的状态

1. **vLLM KV cache**: 推理引擎的缓存状态，重启后从空开始
2. **异步队列状态**: `rollout_queue` 和 `rollout_slots` 是 ephemeral 的
3. **Ray Actor 状态**: 所有 actor 重新创建
4. **WandB run**: 每次重启创建新 run(除非手动 `reinit`)

### Checkpoint 轮转策略

```python
# deepspeed.py
# 保留最多 max_num 个 checkpoint，或总大小不超过 max_mem
# 淘汰优先级: 无 metric → 最差 metric → 最旧
```

### 面试深挖

**Q: 如果 checkpoint 保存到一半进程崩溃了怎么办？**

"A: OpenRLHF 的策略是先保存 checkpoint，再写 metric.json。如果崩溃发生在两者之间，checkpoint 存在但没有 metric 文件。下次运行时它被视为'无 metric checkpoint'，在淘汰时优先被删除——这是安全的降级策略。"

**Q: 如何实现 checkpoint 的远程备份？**

"A: OpenRLHF 没有内置远程备份。需要自己做：(1) 在 `save_steps` 回调中上传到 S3/OSS；(2) 或者用共享文件系统(NFS/Lustre)直接保存到远程存储。slime 有基于磁盘的 weight sync 机制，可以借鉴。"

---

## 4.3 如何监控 RLHF 训练的健康状态？

### 监控 Dashboard 设计

```
第一面板: 训练信号
├── policy_loss (应该缓慢下降)
├── value_loss (应该缓慢下降)
├── kl (应该在 target 附近)
├── entropy (应该缓慢下降但不为 0)
└── clip_ratio (应该在 0.1-0.4)

第二面板: Rollout 质量
├── reward_mean (应该缓慢上升)
├── reward_std (应该稳定)
├── response_length_mean (应该稳定)
├── truncated_rate (应该 <20%)
└── filter_pass_rate (应该在 20-80%)

第三面板: 系统性能
├── timing/generation
├── timing/make_experience
├── timing/ppo_train
├── timing/broadcast
└── timing/step_total

第四面板: 资源使用
├── GPU 利用率
├── GPU 显存
├── 网络带宽
└── CPU 内存
```

### 关键告警规则

| 告警条件 | 可能原因 | 建议动作 |
|---------|---------|---------|
| policy_loss 为 NaN | 梯度爆炸 | 降低 lr, 检查 reward |
| clip_ratio > 0.7 | 策略更新太激进 | 减少 PPO epoch |
| reward_std → 0 | GRPO 方差消失 | 增大 n_samples |
| response_length 持续增长 | 长度偏好 | 启用 overlong penalty |
| truncated_rate > 50% | 模型不会停止 | 增大 max_new_tokens 或 stop penalty |
| timing/generation 突增 | vLLM hang | 检查 GPU 利用率 |
| KL 远偏离 target | KL controller 失效 | 检查 adaptive KL |

### 面试深挖

**Q: 你怎么判断什么时候该停训？**

"A: 三个条件任一满足即停：(1) eval pass@k 连续 3 个 eval 周期不提升(early stopping)；(2) KL 持续超过 target 的 2 倍且 KL controller 无法拉回；(3) reward 和 pass@k 出现背离(reward 上升但 pass@k 下降)。OpenRLHF 的 `--ckpt.best_metric_key` 会自动保存最佳 checkpoint，所以即使训过头也可以回退。"

---

## 4.4 如何做 RLHF 模型的 A/B 测试和上线？

### 上线流程设计

```
Stage 1: 离线评估
├── pass@k 在多个 eval 数据集上
├── 响应长度分布
├── 安全性检测 (有害内容比例)
└── 人工评估 (随机抽样 100 条)

Stage 2: 影子测试 (Shadow Testing)
├── 新模型和旧模型并行推理
├── 记录两者的输出但只返回旧模型的结果
├── 比较输出质量
└── 收集 reward/评分

Stage 3: 灰度发布
├── 5% 流量切换到新模型
├── 监控业务指标(用户满意度、对话完成率)
├── 逐步扩大到 20% → 50% → 100%
└── 保留回滚能力

Stage 4: 全量上线
├── 旧模型下线
├── 持续监控
└── 建立定期重训 pipeline
```

### 回滚策略

```
1. Checkpoint 级回滚: 加载上一个 best checkpoint
2. 流量级回滚: 灰度比例退回 0%
3. Reward function 回滚: 如果发现 reward hacking，回滚 reward function
```

### 面试深挖

**Q: 灰度期间怎么判断新模型是否更好？**

"A: 不仅看自动指标(pass@k, reward)，更看业务指标：(1) 用户满意度评分(如果有的话)；(2) 对话完成率(用户是否中途退出)；(3) 平均对话轮数(太短可能答非所问，太长可能啰嗦)；(4) 用户反馈/投诉率。"

**Q: 资源有限，离线评估和影子测试只能选一个，你选哪个？**

"A: 选影子测试。离线评估的 eval 数据集可能和真实用户分布不一致。影子测试用真实流量，能直接看到模型在实际场景中的表现。而且影子测试不影响用户体验(用户只看到旧模型的结果)，风险为零。"

---

## 4.5 vLLM 和训练框架之间的权重同步怎么做？有什么坑？

### 同步协议详解

```
1. 创建 torch process group:
   DS rank 0 ←→ vLLM engine-0 worker-0
   DS rank 0 ←→ vLLM engine-0 worker-1 (TP)
   DS rank 0 ←→ vLLM engine-1 worker-0
   ...

2. 对每个参数:
   ZeRO-3: allgather 到 rank 0
   rank 0: engine.update_weight.remote(name, dtype, shape)
   rank 0: broadcast tensor (src=0)
   vLLM workers: 接收并 load_weights
```

### 常见坑

**坑 1: ZeRO-3 的 allgather 开销**
```
问题: allgather 通信量 = 模型大小，每个 step 都要传输一次
解决: 用 ZeRO-2 (参数在 rank 0 上就有完整副本)
```

**坑 2: vLLM 版本兼容性**
```
问题: 不同 vLLM 版本的 weight 加载接口可能不同
解决: OpenRLHF 检查 vLLM 版本号 (>0.8.5)，不同版本设置不同环境变量
```

**坑 3: TP 场景下的权重分片**
```
问题: DeepSpeed 的 TP 方式和 vLLM 的 TP 方式可能不同
解决: 用 GatheredParameters 或 GatherReplacedLayerParams 先 gather 再 broadcast
```

**坑 4: CUDA IPC 的设备映射**
```
问题: IPC handle 中的 device_id 需要映射到本地设备
解决: WorkerWrap.update_weight_cuda_ipc() 中重新映射 device_id
```

### 面试深挖

**Q: NCCL broadcast 和 CUDA IPC 的选择标准？**

"A: NCCL broadcast 适合跨节点(但更慢)；CUDA IPC 适合同节点共置(更快但要求共存)。在 Hybrid Engine 模式下(所有模型共享 GPU)，用 CUDA IPC 可以避免 NCCL 的通信开销。但 CUDA IPC 需要 `torch.multiprocessing` 的 IPC 支持，不是所有环境都支持。"

---

# 第五类：业务与实际场景理解

> **考察重点**: 方案适合什么场景，用户关心什么，上线成本有多高，资源有限首先优化什么。

---

## 5.1 RLHF vs DPO vs SFT：不同场景的选型

### 决策矩阵

| 维度 | SFT | DPO | RLHF (PPO/GRPO) |
|------|-----|-----|-----------------|
| **数据需求** | 指令-回答对 | chosen/rejected 对 | reward function 或偏好数据 |
| **训练成本** | 低(1x) | 中(2x, 需要 ref) | 高(3-4x, 需要多模型) |
| **推理成本** | 低 | 低 | 高(需要在线采样) |
| **效果上限** | 受限于 SFT 数据 | 受限于 preference 数据 | 可以超越数据(reward 引导探索) |
| **适用场景** | 格式对齐、任务微调 | 对话质量、安全对齐 | 推理、代码、复杂任务 |
| **稳定性** | 非常稳定 | 稳定 | 不稳定(需要调参) |

### 业务场景推荐

**场景 1: 客服对话机器人**
```
选型: SFT → DPO
原因: 对话质量难以用 reward function 表达，但可以用人工标注的 preference 数据
成本: SFT 几天, DPO 1-2 天
用户关心: 回答的准确性和语气
```

**场景 2: 数学推理助手**
```
选型: SFT → GRPO/REINFORCE++
原因: 答案可验证(数学对错)，reward function 简单(0/1)
成本: SFT 几天, GRPO 1-2 周(需要大量采样)
用户关心: 答案的正确率
```

**场景 3: 代码生成工具**
```
选型: SFT → PPO (code sandbox reward)
原因: 代码可以通过测试用例验证，reward = 通过率
成本: SFT 几天, PPO 2-3 周(需要沙箱环境)
用户关心: 代码能否运行通过
```

**场景 4: 多轮工具调用 Agent**
```
选型: SFT → PPO + Multi-turn Agent RL
原因: 需要在线环境交互，reward 来自任务完成度
成本: SFT 几天, Agent RL 数周(环境搭建+训练)
用户关心: 任务完成率和效率
```

### 面试深挖

**Q: 如果老板说"我们有 100 条高质量 preference 数据，能做 DPO 吗？"**

"A: 100 条太少了。DPO 需要足够的 preference 对来覆盖不同场景。经验上至少需要 1K-10K 条。100 条数据更适合作为 eval set 来评估模型，或者做 few-shot prompting。如果数据实在不够，可以考虑：(1) 用 LLM-as-judge 扩充 preference 数据；(2) 做 SFT 而不是 DPO，SFT 对数据量的要求更低。"

**Q: 资源有限(8×A100)，应该先优化哪个环节？**

"A: 取决于瓶颈在哪：(1) 如果模型基础能力差(如 SFT 都没训好)，先做 SFT；(2) 如果 SFT 效果好但对齐差(回答不安全/不友好)，做 DPO(计算成本低)；(3) 如果需要提升推理/代码能力，做 GRPO(不需要 Critic，8 卡够用)；(4) PPO 只在 GRPO 效果不够好且资源充足时才用。"

---

## 5.2 RLHF 的成本分析

### 训练成本

**以 7B 模型 PPO 为例 (8×A100 80GB)**:

| 组件 | GPU 时间 | 数据量 | 成本占比 |
|------|---------|-------|---------|
| SFT | ~24h | 10K 样本 | 基线 |
| Reward Model | ~12h | 50K preference pairs | SFT 的 50% |
| PPO 训练 | ~72h | 10K prompts × 100 steps | SFT 的 3x |
| vLLM 推理 | ~48h | 10K × 8 samples × 100 steps | SFT 的 2x |
| **总计** | **~156h** | | **SFT 的 6.5x** |

**GRPO 的成本** (去掉 Critic):
```
总计: ~100h (SFT 的 4x)
节省: ~35% (省掉 Critic 训练和推理)
```

### 推理成本

```
PPO 训练期间: 每个 prompt 需要生成 8 个 response → 推理成本 = 8x
上线后: 只生成 1 个 response → 推理成本 = 1x
```

### 面试深挖

**Q: GRPO 的 n_samples_per_prompt 设多少最"划算"？**

"A: 这是一个 cost-quality trade-off。n 越大方差越小但推理成本线性增长。经验上 n=8 是甜蜜点：n 从 2→8 提升显著，n 从 8→32 提升边际递减。对于数学推理(二值 reward)，可能需要 n=16 来获得足够的方差。"

**Q: 如果只有一周时间+8 张卡，怎么做最有价值的事？**

"A: 优先级：(1) Day 1-2: SFT 微调，确保基础能力；(2) Day 3-5: GRPO 训练(n=8, 不需要 Critic)，提升推理能力；(3) Day 6: Eval 和人工评估；(4) Day 7: 写报告。不要尝试 PPO(时间不够调参)或 DPO(没有足够 preference 数据)。"

---

## 5.3 Reward 设计的业务考量

### Reward 的成本和质量权衡

| Reward 类型 | 成本 | 质量 | 适用场景 |
|------------|------|------|---------|
| Rule-based (答案匹配) | 极低 | 高(可验证) | 数学、逻辑 |
| Code sandbox | 低 | 高(可验证) | 代码生成 |
| LLM-as-judge | 中(推理成本) | 中(有 bias) | 开放式对话 |
| Human evaluation | 高 | 最高 | 最终评估 |
| RM (训练的) | 中(训练+推理) | 中 | 通用 |

### 面试深挖

**Q: LLM-as-judge 有什么 bias？怎么缓解？**

"A: 常见 bias：(1) Position bias(倾向于选择第一个回答)；(2) Length bias(倾向于更长的回答)；(3) Style bias(倾向于格式好看的回答)；(4) Self-enhancement(倾向于自己的风格)。缓解：(1) 随机化回答顺序；(2) 控制回答长度；(3) 用多个 LLM judge 取平均；(4) 用 rubric-based 评估(明确评分标准)。"

**Q: 怎么发现 reward hacking？**

"A: 三个信号：(1) Reward 持续上升但人类评估质量不变——模型找到了 reward 的漏洞；(2) Response 变得异常长——reward 可能隐含了长度偏好；(3) 回答格式越来越花哨但内容空洞——reward 可能偏好格式。监控手段：定期随机抽样做人工评估，比较 reward 排名和人工排名的相关性。"

---

## 5.4 Agent RL 的业务价值与挑战

### 适合 Agent RL 的场景

| 场景 | 价值 | 挑战 | 成熟度 |
|------|------|------|-------|
| 代码修复 (SWE-bench) | 高(自动化修 bug) | 环境搭建复杂 | 中 |
| 数学证明 | 中(学术价值) | 环境反馈稀疏 | 低 |
| 工具调用 (API) | 高(实用价值) | API 设计和安全 | 中 |
| 游戏 (Sokoban等) | 低(主要是研究) | 环境简单 | 高 |
| 搜索/RAG | 中(知识增强) | 多轮交互质量 | 中 |

### 面试深挖

**Q: Agent RL 目前最大的瓶颈是什么？**

"A: 三个瓶颈：(1) 环境搭建成本——每个任务需要独立的环境(沙箱、API、数据库)，工程量大；(2) Reward 稀疏——Agent 可能执行 10 步才得到一个 reward，credit assignment 困难；(3) 安全性——Agent 可能执行危险操作(如删除文件)，需要严格的沙箱隔离。目前最成熟的场景是代码生成(有标准 benchmark 和 sandbox)，最不成熟的是通用 Agent(环境太复杂)。"

**Q: 如果让你设计一个搜索 Agent 的 RL 训练，你会怎么做？**

"A: (1) 环境：用搜索引擎 API 作为环境，返回搜索结果片段；(2) Action space: 生成搜索 query + 选择结果 + 综合回答；(3) Reward: 最终回答的正确性(可以用 LLM-as-judge 或答案匹配)；(4) 训练：用 trajectory-wise 的 GRPO(每条轨迹是一个完整的搜索-回答过程)；(5) 安全：限制搜索 query 的长度和频率，防止滥用。"

---

# 综合模拟面试

## 模拟面试 1：系统设计

**面试官**: "假设你要为一个 70B 参数的代码生成模型设计 RLHF 训练系统，目标是提升代码在 HumanEval 上的 pass@1。你有 64 张 A100 80GB。请设计方案。"

**参考回答**:

"首先分析需求：代码生成的 reward 是可验证的(测试用例通过)，所以适合用 GRPO 而不是 PPO(不需要 Critic 省显存)。

**Step 1: 数据准备**
- SFT 数据：Code-Alpaca 或 StarCoder 数据，训 1-2 天
- Eval: HumanEval 的 pass@1 和 pass@10

**Step 2: 训练方案**
- 算法: GRPO, n_samples_per_prompt=8
- Reward: 代码在测试用例上的通过率(0-1 连续值)
- 训练后端: DeepSpeed ZeRO-2 (70B 需要 TP=4)
- 推理: vLLM, TP=4, 4 engines

**Step 3: 资源分配**
- 64 GPU = 8 nodes × 8 GPUs
- Actor: 32 GPUs (ZeRO-2, 4 nodes)
- Ref: 16 GPUs (冻结, 2 nodes)
- Reward: 8 GPUs (代码沙箱 reward, 1 node)
- vLLM: 8 GPUs (4 engines × TP=2, 1 node)
- 总计: 64 GPUs ✓

**Step 4: 关键超参数**
- lr: 1e-6 (cosine schedule)
- batch_size: 128
- rollout_batch_size: 1024
- max_new_tokens: 2048
- kl_coef: 0.01
- clip_eps: 0.2

**Step 5: 监控和评估**
- 每 50 步 eval HumanEval pass@1
- 监控 reward_mean, response_length, clip_ratio
- Best checkpoint 保存基于 eval pass@1

**预计成本**: ~1 周训练时间，~$5K-10K 云服务费用。"

---

## 模拟面试 2：问题排查

**面试官**: "你正在用 OpenRLHF 训练一个 7B 的数学推理模型，用 GRPO。训练到第 500 步时，你发现 eval pass@1 从 30% 突然掉到了 10%。请排查。"

**参考回答**:

"我会按以下顺序排查：

**1. 先看 rollout 指标**
- 如果 reward_mean 也下降 → 模型能力真的下降了
- 如果 reward_mean 没变但 pass@1 下降 → reward hacking

**2. 检查 KL 轨迹**
- 如果 KL 持续增长超过 target → 策略偏离太远，增大 kl_coef
- 如果 KL 正常 → 不是 KL 问题

**3. 检查 response_length**
- 如果变长了 → 可能长度偏好，启用 overlong penalty
- 如果变短了 → 可能截断太多，增大 max_new_tokens

**4. 检查 checkpoint**
- 是否加载了错误的 checkpoint？
- checkpoint 是否损坏？

**5. 检查训练信号**
- clip_ratio 是否异常(>0.7)？
- policy_loss 是否 NaN？

**最可能的原因**: 训练到 500 步时，GRPO 的 std normalization 出问题——如果某个 prompt 的 8 个 response 全对或全错，std→0，advantage 被放大。用 DAPO dynamic filtering 过滤掉这些 prompt。

**验证方法**: 在第 490 步的 checkpoint 上做 eval，如果 pass@1 还是 30%，说明问题出在 490-500 步之间。检查这 10 步的训练指标是否有异常。"

---

## 模拟面试 3：方案论证

**面试官**: "有人说'DPO 已经够用了，不需要 PPO'，你怎么看？"

**参考回答**:

"这个观点有道理但不完全对。DPO 确实简单稳定，但有几个场景 PPO 不可替代：

**DPO 的优势**:
- 简单：不需要推理引擎、不需要在线采样
- 稳定：loss 是 supervised 式的，不容易崩溃
- 低成本：只需要 preference 数据，不需要 reward function

**DPO 的局限(这些场景 PPO 更好)**:
1. **Reasoning 任务**: 数学/代码的 reward 是可验证的，PPO 可以通过在线探索发现 SFT/preference 数据中没有的解法。DPO 只能模仿数据中的偏好。
2. **Distribution shift**: DPO 只在 chosen/rejected 数据上训练，策略可能从未探索过空间中的其他区域。PPO 在线采样可以探索。
3. **多轮/工具使用**: Agent RL 需要和环境交互，DPO 做不到。
4. **迭代改进**: PPO 可以多轮迭代(每轮用新策略采样)，DPO 通常只训一轮。

**实际建议**:
- 如果有高质量 preference 数据 + 简单对齐任务 → DPO
- 如果有 verifiable reward + 需要探索 → GRPO/PPO
- 如果两者都有 → 先 DPO 再 GRPO(DPO 做粗调，GRPO 做精调)

**数据说话**: DeepSeek-R1、Qwen3 的 reasoning 模型都是用 GRPO/REINFORCE++ 训练的，而不是 DPO。这说明在 reasoning 场景，PPO/GRPO 确实更好。"

---

> **使用建议**: 面试前重点复习第一类(原理)和第三类(问题排查)——这两类最能体现"真正做过"而不是"背过"。回答时尽量结合代码细节(如"在 OpenRLHF 的 loss.py 中...")，展示源码级理解。
