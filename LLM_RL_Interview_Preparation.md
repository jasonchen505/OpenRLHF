# LLM 后训练 (Post-Training) & Agent RL 面试准备指南

> 基于 OpenRLHF、slime、verl、ROLL 四大开源 RLHF 框架的深度分析
> 面向 LLM 算法实习岗 (MS在读)，聚焦可深挖的面试考察点

---

## 目录

1. [框架全景对比](#1-框架全景对比)
2. [核心算法原理与实现细节](#2-核心算法原理与实现细节)
3. [分布式训练架构深度解析](#3-分布式训练架构深度解析)
4. [Agent RL / 多轮交互训练](#4-agent-rl--多轮交互训练)
5. [Reward Model 与奖励工程](#5-reward-model-与奖励工程)
6. [关键工程细节与"坑"](#6-关键工程细节与坑)
7. [高频面试问题与深挖点](#7-高频面试问题与深挖点)
8. [代码走读指南](#8-代码走读指南)
9. [面试者自我介绍模板](#9-面试者自我介绍模板)
10. [参考文献](#10-参考文献)

---

## 1. 框架全景对比

### 1.1 四大框架定位

| 框架 | 出品方 | 核心定位 | 训练后端 | 推理后端 | 开源时间 |
|------|--------|---------|---------|---------|---------|
| **OpenRLHF** | 社区 (被Google/字节/腾讯等采用) | Ray+vLLM的生产级RLHF框架 | DeepSpeed | vLLM | 2024 |
| **slime** | 清华THUDM (GLM系列背后) | Megatron+SGLang的RL scaling框架 | Megatron-LM | SGLang | 2025 |
| **verl** | 字节Seed (Doubao团队) | HybridFlow灵活高效RL库 | FSDP/Megatron/VeOmni | vLLM/SGLang/TRT-LLM | 2024 |
| **ROLL** | 阿里淘天&AI Engine | 多角色分布式RL后训练框架 | Megatron-Core/FSDP2 | vLLM/SGLang | 2025 |

### 1.2 架构设计哲学对比

**OpenRLHF**: "Agent-based统一管线"
- 核心创新: Token-in-Token-out的Agent执行范式
- 任何RL算法(PPO/GRPO/REINFORCE++) × 任何执行模式(单轮/多轮) = 正交组合
- Ray单控制器 + vLLM异步推理 + DeepSpeed训练

**slime**: "单后端深度优化"
- 哲学: 多后端框架会隐藏每个引擎的最强特性
- Megatron原生透传(上游改进立即可用) + SGLang深度集成
- Data Buffer作为rollout->train的桥梁模块
- 生产验证: GLM-5.x系列模型

**verl**: "3D-HybridEngine + 算法超市"
- 同一组GPU交替做训练和推理，消除内存冗余
- 15+ advantage estimator + 12+ policy loss函数
- Hydra配置系统 + 插件化硬件抽象层

**ROLL**: "多角色协作 + Agentic RL一等公民"
- Cluster/Worker抽象 + @register装饰器定义分发模式
- 6种PG变体 + 6种优势估计器
- 10+环境类型, step/trajectory粒度训练

### 1.3 关键技术选型对比表

| 维度 | OpenRLHF | slime | verl | ROLL |
|------|----------|-------|------|------|
| **分布式调度** | Ray ActorGroup | Ray + sglang_router | Ray Single-Controller | Ray Cluster/Worker |
| **训练并行** | DeepSpeed ZeRO-1/2/3, RingAttn, AutoTP | Megatron TP/PP/CP/EP/VPP | FSDP/FSDP2/Megatron | Megatron-Core 5D + FSDP2 |
| **推理引擎** | vLLM (AsyncLLMEngine) | SGLang (原生router) | vLLM/SGLang/TRT-LLM | vLLM/SGLang |
| **Weight同步** | NCCL broadcast / CUDA IPC | NCCL/delta sync/磁盘/tensor copy | NCCL/NIXL/Mooncake/Kimi | NCCL/delta sync |
| **GPU共置** | sleep/wake交替 | offload/onload交替 | sleep/wake交替 | partial GPU shrink/expand |
| **异步训练** | 生产者-消费者+信号量回压 | fully-async rollout worker | sync/colocate_async/separate_async/fully_async | async_generation_ratio |
| **Agentic RL** | AgentExecutorBase (单/多轮) | TrajectoryManager + adapters | AgentLoopManager + ToolAgentLoop | RolloutScheduler + EnvManager |
| **算法数量** | ~6种advantage estimator | ~7种 | ~15种advantage + 12种loss | ~6种PG + 6种advantage |
| **配置系统** | argparse (CLI) | argparse (CLI, 2010行) | Hydra YAML + dataclass | Hydra YAML + dataclass |

---

## 2. 核心算法原理与实现细节

### 2.1 PPO (Proximal Policy Optimization)

**核心公式**:
```
L^CLIP = E_t[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
```
其中 `r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)` 是重要性采样比率。

**OpenRLHF实现** (`openrlhf/models/loss.py:PolicyLoss`):
```python
raw_policy_log_ratio = log_probs - old_log_probs
policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)  # 防溢出
ratio = policy_log_ratio.exp()

surr1 = ratio * advantages
surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
loss = -torch.min(surr1, surr2)
```

**面试深挖点**:
1. **为什么log_ratio要clamp到[-20, 20]?** — 防止exp()溢出导致NaN。log_ratio过大说明策略偏移严重，clamp后ratio≈exp(20)≈4.8e8，足够表达"很大"的含义。
2. **Dual-clip PPO是什么?** — 当advantage<0时，额外加一个下界`dual_clip * advantages`(dual_clip>1)，防止策略在负advantage时过度惩罚好的token。
3. **非对称clip范围?** — `eps_clip_low_high`允许上下clip不同，比如(0.1, 0.3)，对负advantage更保守。

### 2.2 GRPO (Group Relative Policy Optimization)

**核心思想**: 同一个prompt生成N个response，用组内统计量做baseline，不需要critic。

**公式**:
```
A_i = (R_i - mean(R_1..R_N)) / std(R_1..R_N)
```

**OpenRLHF实现** (`openrlhf/trainer/ppo_utils/experience_maker.py`):
```python
# group_norm
rewards = rewards.reshape(-1, n_samples_per_prompt)
rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
```

**面试深挖点**:
1. **GRPO vs PPO的核心区别?** — GRPO不需要critic网络，用组内多response的reward统计量替代value function。优势: 节省显存(critic通常和actor一样大)；劣势: 需要对每个prompt生成多个response，增加推理开销。
2. **GRPO的方差问题?** — 当组内所有response的reward相同时(std=0)，advantage全部为0，梯度为0。这就是为什么需要DAPO dynamic filtering过滤全对/全错的prompt。
3. **n_samples_per_prompt如何选择?** — 太小(如2)方差大，太大(如64)推理成本高。实践中常用8-16。数学推理任务因为答案二值化(0/1)，需要更多samples来获得有意义的方差。
4. **Dr. GRPO是什么?** — 去掉std normalization，只减mean。因为std normalization可能过度缩放advantage，特别是当所有response都正确时std趋近0。

### 2.3 REINFORCE++ / REINFORCE++-baseline

**REINFORCE++**: 累积回报 + PPO式clipping，不需要critic
```
G_t = Σ_{t'>=t} γ^{t'-t} * r_{t'}
A_t = G_t  (直接用累积回报作为advantage)
```

**REINFORCE++-baseline**: 组内均值baseline减除
```
A_i = G_i - mean(G_1..G_N)   // 同prompt的N个response做baseline
```

**面试深挖点**:
1. **为什么REINFORCE++-baseline适合reasoning任务?** — 数学推理的reward通常是二值的(对/错)，reward scale不敏感。baseline减除使得即使reward全是0/1，也能产生有意义的梯度。被NVIDIA ProRL V2和Mistral Magistral采用。
2. **与GRPO的区别?** — GRPO做std normalization，REINFORCE++-baseline只减mean。baseline更鲁棒，不会因为std→0而梯度爆炸。

### 2.4 RLOO (Reward Leave-One-Out)

**公式**:
```
baseline_i = (Σ_{j≠i} R_j) / (N-1)
A_i = R_i - baseline_i
```

**面试深挖点**:
1. **RLOO vs GRPO?** — RLOO用leave-one-out均值做baseline，GRPO用组内均值+std normalization。RLOO的baseline更"公平"——每个sample的baseline都不包含自己的reward。
2. **计算开销?** — 和GRPO一样需要n_samples_per_prompt>1，但不需要额外的critic网络。

### 2.5 DPO (Direct Preference Optimization)

**公式**:
```
L_DPO = -E[ log σ( β * (log π/π_ref)(y_w) - log π/π_ref)(y_l)) ]
```

**OpenRLHF实现** (`openrlhf/models/loss.py:DPOLoss`):
```python
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps
logits = pi_logratios - ref_logratios
losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) \
         -F.logsigmoid(-beta * logits) * label_smoothing
```

**面试深挖点**:
1. **DPO的隐式reward是什么?** — `r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))`。DPO本质上是在做reward model learning + policy optimization的联合优化。
2. **DPO vs PPO的trade-off?** — DPO: 简单稳定，不需要在线采样，但受限于离线数据分布。PPO: 在线采样，能探索新策略空间，但训练不稳定。
3. **β的含义?** — 控制对preference的"信任度"。β越大，越严格遵循preference数据；β越小，越保守，接近reference policy。
4. **IPO vs DPO?** — IPO用MSE loss `(logits - 1/(2β))²` 替代log-sigmoid，避免DPO在preference很强时的过拟合问题。

### 2.6 KL散度估计器

**三种估计器** (Schulman blog):
```
k1: KL ≈ log(π/π_ref)                    # 无偏，可能为负
k2: KL ≈ (log(π/π_ref))² / 2             # 非负，平方近似
k3: KL ≈ exp(log π_ref - log π) - 1 - (log π_ref - log π)  # 非负，无偏
```

**面试深挖点**:
1. **k1为什么可能为负?** — 因为log(π/π_ref)在某些token上可能为负(π<π_ref)，虽然整体KL非负，但逐token估计可能为负。
2. **k3的推导?** — 来自reverse KL的Taylor展开: `KL(q||p) = E_q[log q/p] = E_q[exp(log q/p) - 1 - log q/p]`的近似。
3. **实际使用建议?** — k3(low_var_kl)最稳定，推荐用于reward shaping中的KL penalty。

### 2.7 重要性采样修正 (Off-Policy Correction)

**问题**: vLLM异步推理时，rollout的策略可能和当前训练策略不一致(因为权重更新有延迟)。

**三种修正方式** (OpenRLHF `loss.py`):
```python
# TIS (Truncated IS): token级clamp
vllm_is = exp(old_log_probs - rollout_log_probs).clamp(low, high)

# ICEPOP: token级过滤
mask = (vllm_is >= low) & (vllm_is <= high)
loss = (vllm_is * mask) * loss

# seq-mask-tis: 序列级几何均值过滤 + token级修正
seq_is = exp(mean(log_ratio))  # 序列级
seq_mask = (seq_is >= low) & (seq_is <= high)  # 序列级过滤
vllm_is = exp(log_ratio)  # token级修正
loss = seq_mask * vllm_is * loss
```

**面试深挖点**:
1. **为什么需要IS修正?** — 异步训练中，generation和training并行，生成样本用的权重可能比当前训练权重旧几个step。如果不修正，off-policy程度太大会导致训练不稳定。
2. **ICEPOP vs TIS?** — TIS是clamp(软截断)，ICEPOP是mask(硬过滤)。ICEPOP更激进，直接丢弃off-policy程度大的token。
3. **seq-mask-tis的优势?** — 用序列级几何均值判断整条序列的off-policy程度，避免逐token判断的噪声。

---

## 3. 分布式训练架构深度解析

### 3.1 OpenRLHF: Ray单控制器 + vLLM

**架构图**:
```
PPOTrainer (Ray Actor, 单控制器)
    |
    ├── PolicyModelActor (N个, DeepSpeed训练)
    |       └── broadcast_to_vllm() ──NCCL──> vLLM engines
    |
    ├── CriticModelActor (N个, DeepSpeed训练)
    |
    ├── ReferenceModelActor (N个, 冻结推理)
    |
    ├── RewardModelActor (N个, 冻结推理)
    |
    └── RolloutRayActor (M个, vLLM AsyncLLMEngine)
```

**Weight同步协议** (`openrlhf/trainer/ray/ppo_actor.py`):

```
1. 创建torch process group: DS rank 0 <-> 所有vLLM engine workers
2. 对每个参数:
   a. ZeRO-3: allgather参数到rank 0
   b. rank 0调用engine.update_weight.remote(name, dtype, shape)
   c. rank 0 broadcast参数tensor (src=0)
   d. vLLM worker接收并load_weights
```

**Sleep/Wake机制**:
```
训练阶段:
  vLLM engines sleep() → 释放KV cache显存
  Actor reload_states() → CPU→GPU
  Actor fit() → PPO训练
  Actor offload_states() → GPU→CPU

生成阶段:
  vLLM wake_up(tags=["weights"]) → 加载权重
  broadcast_to_vllm() → 同步新权重
  vLLM wake_up(tags=["kv_cache"]) → 分配KV cache
  generate() → 生成response
```

**面试深挖点**:
1. **为什么用Ray而不是纯torch分布式?** — Ray提供细粒度的资源调度(placement group)、actor生命周期管理、异步任务编排。对于需要协调4种不同模型+推理引擎的RLHF，Ray的actor模型比torch的SPMD模型更灵活。
2. **ZeRO-3下weight同步的开销?** — 需要allgather所有参数到rank 0，通信量≈模型大小。这是ZeRO-3的主要瓶颈之一，所以OpenRLHF默认用ZeRO-2。
3. **CUDA IPC vs NCCL broadcast?** — CUDA IPC是零拷贝共享内存，适合同节点共置场景；NCCL broadcast适合跨节点。CUDA IPC更快但要求vLLM和训练在同一GPU上。

### 3.2 verl: 3D-HybridEngine

**核心思想**: 同一组GPU交替做训练(FSDP)和推理(vLLM)，通过weight resharding切换。

```
训练阶段: FSDP分片 → 每GPU持有模型的1/N
推理阶段: vLLM TP分片 → 每GPU持有模型的1/M (M可能≠N)
Weight同步: FSDP allgather → vLLM broadcast → vLLM TP shard
```

**面试深挖点**:
1. **FSDP到vLLM的weight resharding怎么做?** — FSDP按flat param分片，需要先allgather恢复完整参数，然后按vLLM的TP方式重新分片。verl通过CheckpointEngine抽象了这个过程。
2. **和OpenRLHF的sleep/wake有什么区别?** — OpenRLHF的sleep/wake是在同一分片策略下offload/reload；verl是不同分片策略间的resharding，更复杂但也更灵活。

### 3.3 slime: Megatron + SGLang + Data Buffer

**三模块架构**:
```
训练模块 (Megatron-LM)  ←→  Data Buffer  ←→  推理模块 (SGLang)
    |                        |                    |
    | TP/PP/CP/EP            | Sample数据结构      | sglang_router
    | TensorBackuper         | DataSource管理      | PD分离
```

**面试深挖点**:
1. **为什么slime只支持SGLang?** — 多后端框架会隐藏每个引擎的最强特性。SGLang的routing、caching、disaggregation等特性需要深度集成才能发挥最大价值。
2. **TensorBackuper是什么?** — 在GPU显存中维护多个模型副本(actor/ref/old_actor/teacher)的备份，通过backup/restore切换，避免重复加载checkpoint。
3. **Delta Weight Sync?** — 检测权重的字节级变化，只传输变化的位置+值，大幅减少大模型的通信量。

### 3.4 ROLL: Cluster/Worker + @register装饰器

**核心抽象**:
```python
@register(dispatch_mode=DispatchMode.DP_MP_COMPUTE)
def forward_step(self, data):
    # 自动按DP分片发送数据，收集TP rank 0 + 最后PP stage的结果
    ...
```

**面试深挖点**:
1. **@register装饰器做了什么?** — 声明方法的分发模式(ONE_TO_ALL, DP_MP_COMPUTE等)，框架自动处理数据分片、结果收集、跨rank通信。
2. **Partial GPU Overlap?** — 训练时"inference GPU"可以"shrink"部分显存给训练用，推理时再"expand"回来。比sleep/wake更细粒度。

---

## 4. Agent RL / 多轮交互训练

### 4.1 OpenRLHF的Agent执行范式

**核心设计** (`openrlhf/utils/agent.py`):

```python
class AgentExecutorBase(ABC):
    async def execute(self, prompt, label, sampling_params, max_length, tokenizer, llm_engine, images=None):
        ...

class SingleTurnAgentExecutor(AgentExecutorBase):
    # 单轮: prompt → vLLM生成 → reward function → 返回

class MultiTurnAgentExecutor(AgentExecutorBase):
    # 多轮: prompt → [generate → step() → feedback → generate → ...] → 返回
```

**关键数据结构**:
```python
{
    "observation_tokens": [token_ids...],      # 完整轨迹的token
    "action_ranges": [(start, end), ...],      # 策略生成的token区间
    "rollout_log_probs": [log_probs...],       # 每个token的log概率
    "reward": float,                           # 总reward
    "scores": float,                           # DAPO过滤用
    "extra_logs": {...}                        # 额外日志
}
```

**面试深挖点**:
1. **action_ranges的作用?** — 区分哪些token是策略生成的(action)，哪些是环境提供的(observation/feedback)。只有action token参与loss计算。
2. **环境反馈token的log_prob怎么处理?** — 设为0.0，因为它们不是策略生成的。
3. **token预算管理?** — 每轮动态计算`max_tokens = max_length - len(current_obs_tokens)`，确保不超出上下文窗口。

### 4.2 slime的TrajectoryManager

**核心功能**: 从多轮对话构建per-session message tree，处理token drift (CLEAN/REALIGN/FORK)，线性化为训练Sample。

**面试深挖点**:
1. **Token drift是什么?** — 多轮对话中，前一轮的tokenization和后一轮拼接后的tokenization可能不一致(因为BPE的上下文依赖)。需要检测并修复。
2. **CLEAN/REALIGN/FORK三种策略?** — CLEAN: 丢弃不一致的token；REALIGN: 重新tokenize对齐；FORK: 分叉成新的trajectory。

### 4.3 verl的AgentLoopManager

**设计**: 与SGLang的多轮能力深度集成，支持tool calling agent。

**面试深挖点**:
1. **和OpenRLHF的AgentExecutor有什么区别?** — verl更贴近SGLang的原生多轮能力，OpenRLHF更通用(任何LLM engine都可以)。
2. **ReTool recipe?** — 训练模型使用代码沙箱执行工具调用，reward来自代码执行结果的正确性。

### 4.4 ROLL的RolloutScheduler

**核心功能**: 管理环境交互式rollout，支持step-wise和trajectory-wise训练。

```python
# Step-wise: 每个step都计算loss
# Trajectory-wise: 整个episode结束后才计算loss
```

**面试深挖点**:
1. **Step-wise vs Trajectory-wise的trade-off?** — Step-wise: 信号更密集，训练更快收敛，但可能引入偏差。Trajectory-wise: 信号更准确，但延迟大。
2. **GroupQueue?** — 同一seed/config的环境放在一组，减少variance。

---

## 5. Reward Model 与奖励工程

### 5.1 Reward Model架构

**OpenRLHF** (`openrlhf/models/model.py`):
```python
# Transformer + Linear(1) value head
# Reward提取在EOS位置
values = hidden_states[:, -1, :]  # 最后一个token
reward = self.value_head(values)   # Linear(hidden_size, 1)
```

**面试深挖点**:
1. **为什么reward在EOS位置提取?** — 因为reward是对整个response的评分，EOS token的hidden state包含了整个序列的信息(causal attention)。
2. **Reward normalization?** — 训练时统计reward_mean和reward_std，保存到config.json。推理时用这些统计量做z-score normalization。

### 5.2 Critic Model vs Reward Model

| 维度 | Reward Model | Critic Model |
|------|-------------|-------------|
| 输出 | 标量reward (EOS位置) | 每个token的value |
| 用途 | 评分response | GAE的baseline |
| 训练 | PairWise loss (chosen vs rejected) | Value loss (MSE with clipping) |
| 推理 | 冻结 | 训练中更新 |

### 5.3 自定义Reward Function

**OpenRLHF** (`examples/python/reward_func.py`):
```python
def reward_func(queries, prompts, labels, **kwargs):
    return {
        "rewards": reward,    # 用于advantage计算
        "scores": reward,     # 用于DAPO dynamic filtering
        "extra_logs": {...},  # 用于wandb日志
    }
```

**面试深挖点**:
1. **rewards vs scores的区别?** — rewards是原始reward信号，直接参与梯度计算。scores是归一化到[0,1]的值，用于过滤太容易/太难的prompt(DAPO)。可以不同。
2. **Reward hacking?** — 模型可能学会利用reward function的漏洞获得高分但实际质量差。缓解: reward ensemble, KL penalty, human evaluation。

### 5.4 常见Reward设计模式

| 任务类型 | Reward设计 | 框架示例 |
|---------|-----------|---------|
| **数学推理** | 答案匹配 0/1 | OpenRLHF math_reward_func |
| **代码生成** | 测试用例通过率 | ROLL code sandbox, slime rm_hub |
| **对话质量** | LLM-as-judge | ROLL LLM judge, verl reward_manager |
| **工具使用** | 任务完成度 | ROLL GEM environment |
| **多轮交互** | 环境反馈的累积reward | OpenRLHF multi-turn agent |

---

## 6. 关键工程细节与"坑"

### 6.1 Action Mask的偏移

**关键代码** (`samples_generator.py:267`):
```python
action_mask = action_mask[1:truncate_length]  # 右移1位
```

**原因**: 自回归训练中，位置t的loss预测token t。第一个token没有预测目标，最后一个生成token的预测在位置T-1。所以mask需要右移1位对齐。

**面试深挖点**: 如果不移位会怎样? — 最后一个response token的loss会被忽略，而第一个prompt token(不该参与loss的)会被错误计入。

### 6.2 序列长度平衡

**OpenRLHF** (`openrlhf/utils/seqlen_balancing.py`):
- Karmarkar-Karp算法将序列分组到balanced micro-batch
- `balance_experiences()`: 将长短序列交错分配到不同DP rank

**面试深挖点**: 为什么需要? — 如果一个DP rank全是长序列，其他全是短序列，那个rank会成为瓶颈(计算时间最长)，其他rank空等。平衡后GPU利用率更高。

### 6.3 Sample Packing

**原理**: 用FlashAttention 2将多条短序列pack到一个tensor中，消除padding浪费。

**面试深挖点**:
1. **怎么处理attention mask?** — FlashAttention 2支持block-diagonal attention mask，pack的序列之间不互相attend。
2. **位置编码怎么处理?** — 每条序列独立计算position ids，从0开始。

### 6.4 VLM (Vision-Language Model) RL

**挑战**:
- 图像token和文本token的对齐
- 多轮交互中的图像累积(如截图)
- M-RoPE (Qwen3.5) 和 Gemma4的位置编码特殊处理

**面试深挖点**: 图像pad token去重? — vLLM会展开图像placeholder token，如果多轮交互累积多张图片，需要去重避免重复展开。

### 6.5 DAPO Dynamic Filtering

**逻辑** (`samples_generator.py:169-195`):
```python
for batch in experiences_per_prompt:
    avg_score = mean(scores)
    if avg_score < min_score or avg_score > max_score:
        discard(batch)  # 太难或太容易
        dispatch_replacement_prompt()
```

**面试深挖点**:
1. **为什么要过滤?** — 全对的prompt没有学习价值(advantage=0)，全错的prompt梯度噪声大。保留"有挑战但可学习"的prompt最有效。
2. **过滤率多少合适?** — 过滤率太高说明数据太简单或太难，需要调整数据分布。实践中20-50%的过滤率较健康。

---

## 7. 高频面试问题与深挖点

### 7.1 算法理解类

**Q1: PPO、GRPO、REINFORCE++的核心区别是什么?各有什么优缺点?**

**参考答案**:
- PPO: 需要critic网络做GAE，训练稳定但显存开销大(需要存actor+critic+ref+reward共4个模型)
- GRPO: 不需要critic，用组内多response的reward统计量做baseline。节省显存但需要n_samples>1，推理开销大
- REINFORCE++-baseline: 组内均值baseline，不需要critic，不需要std normalization，更鲁棒

**深挖**: 面试官可能追问"GRPO的advantage如果std很小怎么办?"→ 用eps=1e-9防除零，或用Dr. GRPO(不做std norm)。

**Q2: KL散度在RLHF中的作用是什么?有哪些估计方式?**

**参考答案**:
- 作用: 作为reward shaping防止策略偏离reference policy太远，避免reward hacking
- 放置位置: 每个response token都有KL penalty，外部reward只放在EOS位置
- 三种估计器: k1(无偏可能为负), k2(非负平方), k3(非负无偏，推荐)
- Adaptive KL Controller: 根据实际KL与target的比值动态调整kl_coef

**深挖**: "KL penalty放在reward里和放在loss里有什么区别?" → 放在reward里是reward shaping(影响advantage计算)，放在loss里是显式正则化项(直接加到loss)。OpenRLHF用reward shaping方式。

**Q3: DPO和PPO的本质区别是什么?什么时候用DPO，什么时候用PPO?**

**参考答案**:
- DPO: 离线算法，直接从preference数据学习，不需要在线采样。简单稳定但受限于数据分布
- PPO: 在线算法，需要和环境交互采样。能探索新策略空间但训练不稳定
- 用DPO: 有高质量preference数据，追求简单稳定
- 用PPO: 需要在线探索，有reward function(而非静态数据)

**深挖**: "DPO的隐式reward是什么?" → `r(x,y) = β * log(π_θ(y|x)/π_ref(y|x))`，DPO本质上在同时学reward model和优化policy。

### 7.2 工程实现类

**Q4: 如何在有限GPU上跑RLHF的4个模型?**

**参考答案**:
- Sleep/Wake机制: vLLM引擎sleep释放KV cache → Actor训练 → vLLM wake恢复生成
- ZeRO offload: optimizer states和gradient offload到CPU
- 模型共置: 用Ray placement group让4种模型共享GPU，通过时间片轮转
- LoRA/QLoRA: 减少可训练参数，降低显存

**深挖**: "sleep模式下训练完怎么把权重同步到vLLM?" → wake_up(tags=["weights"])只分配权重显存，broadcast完再wake_up(tags=["kv_cache"])分配推理显存。

**Q5: vLLM和训练框架之间的权重同步怎么做?**

**参考答案**:
- NCCL broadcast: DS rank 0 → 所有vLLM workers。适合跨节点
- CUDA IPC: 同节点零拷贝共享内存。更快但要求共置
- 对于ZeRO-3: 先allgather到rank 0再broadcast
- Delta sync: 只传输变化的权重字节(slime特有)

**深挖**: "ZeRO-3下allgather的通信量多大?" → 约等于模型参数量×2(BF16)。这是ZeRO-3在RLHF场景的主要瓶颈，所以很多框架默认用ZeRO-2。

**Q6: 异步训练的架构设计?如何保证训练稳定性?**

**参考答案**:
- 生产者-消费者: GenerateSamplesActor生成 → Queue → TrainingActor训练
- 回压机制: rollout_slots信号量控制in-flight batch数量
- 互斥锁: VLLMLock确保生成和权重广播不重叠
- IS修正: 修正off-policy的rollout数据分布偏移

**深挖**: "Partial rollout模式为什么更快?有什么代价?" → 生成时不加锁，可以和训练并行。代价是in-flight样本可能包含新旧混合权重生成的token，off-policy程度更大。

### 7.3 Agent/多轮交互类

**Q7: 多轮Agent RL和单轮RLHF有什么核心区别?**

**参考答案**:
- 数据结构: 多轮需要tracking action_ranges区分策略token和环境token
- Token预算: 每轮动态计算剩余token budget
- Token drift: 多轮拼接的tokenization不一致问题
- Reward: 可以是每步的中间reward或最终的trajectory reward
- 训练粒度: step-wise(每步训练) vs trajectory-wise(整个episode训练)

**深挖**: "环境反馈的token参与loss计算吗?" → 不参与。action_mask标记只有策略生成的token才计算loss。环境反馈token的log_prob设为0。

**Q8: 如何设计一个好的reward function?**

**参考答案**:
- 二值reward(数学): 简单但方差大，需要n_samples>1
- 连续reward(代码): 测试通过率，更平滑
- LLM-as-judge: 灵活但可能有bias
- Reward shaping: 加入格式bonus、长度penalty等
- 多维度reward: 不同维度加权组合

**深挖**: "Reward hacking怎么发现和缓解?" → 监控reward和实际metric的gap；用reward ensemble；KL penalty限制策略偏移；定期human evaluation。

### 7.4 系统设计类

**Q9: 如果让你从零设计一个RLHF框架，你会怎么做?**

**参考答案框架**:
1. 训练后端选择: FSDP(简单) vs Megatron(大规模)
2. 推理引擎选择: vLLM(通用) vs SGLang(高级特性)
3. 分布式调度: Ray(灵活) vs torch(简单)
4. GPU共置策略: sleep/wake vs partial overlap
5. 算法支持: 至少支持PPO/GRPO/REINFORCE++
6. Agent支持: AgentExecutor抽象，支持单/多轮

**深挖**: "为什么大多数框架都选Ray?" — 因为RLHF需要协调多种异构组件(训练模型、推理引擎、reward model)，Ray的actor模型比torch的SPMD模型更适合这种场景。

**Q10: 如何debug RLHF训练?**

**参考答案**:
- 检查reward分布: mean/std是否合理，是否有reward hacking
- 监控KL散度: 是否在target附近
- 检查clip_ratio: 被clip的token比例，太高说明策略更新太激进
- 检查advantage分布: 是否有有意义的正负区分
- 用rollout mock/dump复现问题(ROLL支持)
- 可视化生成样本质量

---

## 8. 代码走读指南

### 8.1 OpenRLHF 核心代码路径

| 文件 | 内容 | 面试重要度 |
|------|------|-----------|
| `openrlhf/models/loss.py` | PPO/GRPO/DPO/GSPO loss实现 | ⭐⭐⭐⭐⭐ |
| `openrlhf/trainer/ppo_utils/experience_maker.py` | 经验收集+优势估计 | ⭐⭐⭐⭐⭐ |
| `openrlhf/utils/agent.py` | Agent执行框架 | ⭐⭐⭐⭐ |
| `openrlhf/trainer/ray/ppo_actor.py` | 权重同步到vLLM | ⭐⭐⭐⭐ |
| `openrlhf/trainer/ppo_trainer.py` | 同步训练主循环 | ⭐⭐⭐⭐ |
| `openrlhf/trainer/ppo_trainer_async.py` | 异步训练架构 | ⭐⭐⭐⭐ |
| `openrlhf/trainer/ppo_utils/samples_generator.py` | vLLM采样+DAPO过滤 | ⭐⭐⭐ |
| `openrlhf/models/model.py` | RM/Critic模型定义 | ⭐⭐⭐ |
| `openrlhf/trainer/ppo_utils/kl_controller.py` | KL自适应控制 | ⭐⭐⭐ |
| `openrlhf/utils/seqlen_balancing.py` | 序列长度平衡 | ⭐⭐ |

### 8.2 推荐阅读顺序

1. **先理解算法**: `loss.py` → `experience_maker.py` → `kl_controller.py`
2. **再理解架构**: `ppo_trainer.py` → `ppo_actor.py` → `vllm_engine.py`
3. **然后理解Agent**: `agent.py` → `samples_generator.py`
4. **最后理解工程**: `seqlen_balancing.py` → `deepspeed.py`

### 8.3 slime 关键代码路径

| 文件 | 内容 |
|------|------|
| `slime/backends/megatron_utils/loss.py` | 所有loss函数和advantage计算 |
| `slime/utils/ppo_utils.py` | KL散度、PPO loss、GAE、GRPO数学 |
| `slime/ray/rollout.py` | RolloutManager, SGLang引擎编排 |
| `slime/agent/trajectory.py` | TrajectoryManager多轮token管理 |
| `slime/rollout/fully_async_rollout.py` | 全异步rollout worker |

### 8.4 verl 关键代码路径

| 文件 | 内容 |
|------|------|
| `verl/trainer/ppo/core_algos.py` | 15+ advantage estimator + 12+ policy loss |
| `verl/workers/engine_workers.py` | 统一TrainingWorker, ActorRolloutRefWorker |
| `verl/single_controller/` | HybridFlow单控制器编程模型 |
| `verl/checkpoint_engine/` | 多种weight同步后端 |
| `verl/experimental/fully_async_policy/` | 全异步训练 |

### 8.5 ROLL 关键代码路径

| 文件 | 内容 |
|------|------|
| `roll/pipeline/rlvr/actor_pg_worker.py` | 6种PG变体实现 |
| `roll/distributed/scheduler/decorator.py` | @register分发模式 |
| `roll/distributed/strategy/strategy.py` | 后端策略抽象 |
| `roll/pipeline/agentic/env/` | 10+环境实现 |
| `roll/distributed/executor/cluster.py` | Cluster/Worker抽象 |

---

## 9. 面试者自我介绍模板

### 9.1 技术背景介绍 (1-2分钟)

"我是XX大学MS在读，研究方向是LLM后训练和Agent RL。我系统性地学习了主流的RLHF框架，包括OpenRLHF、slime、verl和ROLL，深入理解了从算法到工程的完整链路。

在算法层面，我熟悉PPO、GRPO、REINFORCE++、DPO等主流算法的原理和实现差异，理解KL散度控制、重要性采样修正、advantage estimation等核心技术。

在工程层面，我理解Ray分布式调度、vLLM/SGLang推理引擎、DeepSpeed/FSDP训练策略、weight同步协议、sleep/wake显存管理等关键组件的设计和实现。

在Agent RL方面，我理解多轮交互训练的核心挑战，包括token-level trajectory tracking、action mask对齐、token drift处理、step-wise vs trajectory-wise训练等。"

### 9.2 项目经验包装建议

即使没有直接的RLHF项目经验，可以从以下角度包装:

1. **"我阅读并分析了X万行RLHF框架源码"** — 展示代码理解和系统设计能力
2. **"我对比了4个框架的架构设计差异"** — 展示技术选型和trade-off分析能力
3. **"我理解从reward设计到模型训练的完整链路"** — 展示端到端的系统思维
4. **"我关注最新的RL训练方法(DAPO/GRPO/REINFORCE++)"** — 展示技术敏感度

### 9.3 面试中的"加分"表达

- "在OpenRLHF的实现中，这个是通过XXX实现的，我理解它这样做是因为..."
- "slime和verl在这个问题上有不同的设计选择，slime选择了XXX因为..."
- "这个算法的公式是XXX，但在工程实现中需要注意XXX..."
- "如果我要改进这个，我会考虑XXX，因为..."

---

## 10. 参考文献

### 核心论文
- [InstructGPT] Ouyang et al., "Training language models to follow instructions with human feedback", NeurIPS 2022
- [PPO] Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- [DPO] Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023
- [GRPO] Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning", 2024
- [REINFORCE++] Hu, "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models", 2025
- [OpenRLHF] Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework", 2024
- [HybridFlow/verl] Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework", EuroSys 2025
- [RLOO] Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs", ACL 2024
- [DAPO] Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System", 2025

### 框架文档
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- slime: https://github.com/THUDM/slime
- verl: https://github.com/volcengine/verl
- ROLL: https://github.com/alibaba/ROLL

### 补充阅读
- Schulman blog on KL approximators: http://joschu.net/blog/kl-approx.html
- Dual-clip PPO: arXiv:1912.09729
- GSPO: arXiv:2507.18071
- CISPO: Minimax-M1 paper
- ProRL V2: NVIDIA reasoning model training

---

> **最后的建议**: 面试前重点复习§2(算法原理)和§7(面试问题)，能够手写PPO/GRPO的伪代码，理解loss函数中每一行代码的作用。对于§3(分布式架构)，重点理解weight同步和sleep/wake机制。对于§4(Agent RL)，理解action_ranges和token alignment的核心设计。
