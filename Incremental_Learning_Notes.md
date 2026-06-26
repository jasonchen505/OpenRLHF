# OpenRLHF 增量学习笔记：从代码分析到 4090 实战

> 在前两轮文档（面试准备指南 + 五类问题深度应对）基础上的新发现
> 按"之前没讲清楚 / 讲错了 / 新发现"组织

---

## 一、前两轮文档的盲区补充

### 1.1 Loss 聚合中 `dp_size` 乘法的真正含义

**之前只说了公式，没说为什么**。

```python
# loss.py:aggregate_loss
if token_level_loss:
    loss = torch.clamp(token_counts, min=1e-8)
    loss = (loss * loss_mask).sum() / loss * dp_size  # 关键！
```

**问题**: DeepSpeed/DDP 会自动对 DP rank 的梯度做平均。如果 local loss 已经除以了 global token count，DP 平均后就变成了 `global_sum / (global_count * dp_size)` — 多除了一个 `dp_size`。

**解法**: 本地 loss 乘以 `dp_size`，DP 平均后抵消：
```
local_loss = global_sum / global_count * dp_size
DP平均后 = local_loss / dp_size = global_sum / global_count  ✓
```

**这是分布式训练中最容易出 bug 的地方**。如果 `dp_size` 传错了（比如用 local world size 而不是 DP world size），loss scale 就会错，梯度会偏大或偏小。

### 1.2 GPTLMLoss 中"全 padding chunk"的保护

```python
# loss.py:70-73
if torch.all(shift_labels == self.IGNORE_INDEX):
    loss = shift_logits.mean() * 0
```

**之前没提到**: Ring Attention 场景下，如果某个 GPU 的 label chunk 全是 padding（`-100`），`CrossEntropyLoss` 会返回 NaN（0/0）。用 `logits.mean() * 0` 保持梯度图连通性（避免 `all_reduce` 挂死），同时贡献零 loss。

**面试价值**: 这种"看起来没用其实防止分布式训练挂死"的代码，是区分"读过代码"和"理解分布式训练"的标志。

### 1.3 Action Mask 偏移的更深理解

```python
# samples_generator.py:267
action_mask = action_mask[1:truncate_length]  # 右移1位
```

**之前只说了"自回归对齐"，没说具体的 off-by-one 逻辑**:

假设 token 序列为 `[prompt_0, prompt_1, action_0, action_1, action_2]`：
- `action_ranges = [(2, 5)]` — action_0 到 action_2
- 原始 `action_mask = [0, 0, 1, 1, 1]`
- 右移后 `action_mask = [0, 1, 1, 1]`（去掉第一个位置）

为什么？训练时：
- 位置 1 的 logit 预测 token 2（prompt_1 → action_0）← 不该算 loss
- 位置 2 的 logit 预测 token 3（action_0 → action_1）← 该算 loss
- 位置 3 的 logit 预测 token 4（action_1 → action_2）← 该算 loss

所以 action_mask 应该标记位置 2 和 3（即原始 mask 的 `action_mask[1:]`）。

**如果不移位**: 位置 2 的 loss 被错误计入（它预测的是 action_1，但 mask 标记的是 action_0 的位置）。

### 1.4 Replay Buffer 的动态 batch 机制

**之前只提到了 Karmarkar-Karp 算法，没说 replay buffer 内部的 optimizer step 控制**。

```python
# replay_buffer.py:140-150
optimizer_step = [0] * (len(partitions) - 1) + [1]
```

**关键设计**: 每个 training step 内可能有多个 micro-batch（因为动态分批）。只有最后一个 micro-batch 触发 `optimizer.step()`，前面的都是梯度累积。

**`sample_loss_scale` 的作用**:
```python
sample_loss_scale = [len(partition) / sample_num for partition in partitions]
```

如果一个 step 有 10 个样本，分成 3 个 micro-batch [4, 3, 3]，则 scale = [0.4, 0.3, 0.3]。这确保梯度加权正确，不因分批方式不同而改变总梯度。

**面试价值**: "你怎么处理动态 batch size 下的梯度累积？"——这是高级工程问题。

---

## 二、4090 特有的工程细节

### 2.1 PCIe 拓扑对训练的实际影响

```
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV12    NV12    NV12    SYS     SYS     SYS     SYS
GPU1    NV12     X      NV12    NV12    SYS     SYS     SYS     SYS
...
```

**4090 通常没有 NVLink**，GPU 间通信走 PCIe。实测影响：
- ZeRO-3 all-gather: 每次 forward/backward 都要通信，PCIe 带宽 ~32 GB/s vs NVLink ~900 GB/s → 慢 ~28x
- 但对于 1.5B 模型（参数 ~3 GB），单次 all-gather 只需 ~0.1 秒，影响不大
- 对于 7B 模型（参数 ~14 GB），单次 all-gather ~0.4 秒，累积起来显著

**实际建议**: 1.5B 模型用 ZeRO-2 就够（不需要 all-gather），7B 才需要 ZeRO-3。

### 2.2 vLLM 的 `enforce_eager` 在 4090 上更重要

```python
# vllm_engine.py
"enforce_eager": args.vllm.enforce_eager,
```

CUDA Graphs 会预分配固定大小的 GPU 内存，这些内存在 sleep 模式下无法被回收。4090 只有 24GB，CUDA Graphs 的预分配会挤占 DeepSpeed 训练的内存。

**建议**: 4090 上始终用 `--vllm.enforce_eager`，牺牲 ~10% 推理速度换取 ~500MB-1GB 的内存回收。

### 2.3 `gpu_memory_utilization` 的精确含义

```python
# vllm_engine.py:278
"gpu_memory_utilization": args.vllm.gpu_memory_utilization,
```

这个参数告诉 vLLM "你可以用 GPU 总显存的 X%"。但它有两个阶段的行为不同：

**阶段 1 — 初始化**: vLLM 加载模型权重到 GPU，如果权重超过 `gpu_memory_utilization * 24GB`，会失败。
**阶段 2 — 推理**: KV cache 从剩余空间分配。

对于 7B 模型（权重 ~15GB），`gpu_memory_utilization=0.5` 意味着 vLLM 最多用 12GB，但权重就要 15GB → **初始化就失败**。

**实际做法**: OpenRLHF 的 hybrid engine 模式下，vLLM 和 DeepSpeed 共享 GPU。`gpu_memory_utilization` 限制的是 vLLM 的 KV cache 预算，权重不受此限制（权重由 sleep/wake 机制管理）。所以 7B 模型用 `gpu_memory_utilization=0.5` 是可行的——权重 ~15GB + KV cache ~5GB = ~20GB，训练时 vLLM sleep 释放 KV cache。

### 2.4 `max_tokens_per_gpu` 的动态 batch 含义

```python
# seqlen_balancing.py
def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu, ...):
    max_tokens_per_gpu *= ring_attn_size * ds_tensor_parallel_size
```

这个参数不是"每 GPU 最多生成多少 token"，而是"每个 micro-batch 在每 GPU 上的总 token 数上限"。

对于 4090 (24GB)：
- 1.5B 模型: `max_tokens_per_gpu=16384` 没问题
- 7B 模型: 可能需要降到 `8192` 避免 OOM

---

## 三、代码中被忽略的关键设计

### 3.1 `Experience` 数据类的 `tensor_field` 元数据

```python
# experience.py
class TensorField:
    role: str  # "step" or "episode"
```

`role="step"` 的字段（如 `sequences`, `action_mask`）在 batch 时需要右对齐 padding。
`role="episode"` 的字段（如 `rewards`, `response_length`）直接 stack。

**这个设计决定了整个数据管线的 batching 逻辑**。如果 `role` 标错，batch 时就会产生维度不匹配。

### 3.2 `balance_experiences` 的 index 追踪

```python
# experience_maker.py:244-249
for i, sample in enumerate(rollout_samples):
    sample.index = [i]  # 追踪原始位置
```

动态 batch 平衡会把 samples 重新分配到不同 DP rank。但 `compute_advantages_and_returns` 中的 group 操作（GRPO/RLOO）需要按原始 prompt index 分组。

**`index` 字段就是为了解决"平衡后如何找回原始分组"的问题**。

### 3.3 DAPO 过滤的 replacement 机制

```python
# samples_generator.py:185-195
if avg_score < min_score or avg_score > max_score:
    discard(batch)
    # 取新的 prompt 补上
    new_prompt = next(dataloader_iter)
    dispatch(new_prompt)
```

**之前没讲到的细节**: 被过滤的 prompt 会立即被替换，但替换的新 prompt 也可能被过滤。如果数据集太简单（大部分 prompt 全对），可能陷入"不断过滤不断替换"的循环，直到 dataloader exhausted。

**监控信号**: `dynamic_filtering_pass_rate`。如果持续 <20%，说明数据太难或 reward function 太严格。

### 3.4 ProRL 的"截断惩罚"两种模式

```python
# length_penalty.py:78-90
if stop_properly_penalty_coef < 0:
    experience.rewards[j] = stop_properly_penalty_coef  # 固定负值
else:
    experience.rewards[j] = experience.rewards[j] * stop_properly_penalty_coef  # 乘法
```

**固定负值**: 截断的 response 直接给负 reward，不管内容质量。适合"截断就是错"的任务（如对话）。

**乘法**: 截断的 response 的 reward 被缩放。如果内容 99% 正确但被截断，reward 仍然正但减小。适合"截断可能部分正确"的任务（如代码生成）。

---

## 四、跨框架对比的新发现

### 4.1 slime 的"权重版本号"机制

slime 在每次 weight sync 后递增版本号，并用 CI 测试验证 SGLang engine 的版本号与训练侧一致。

**OpenRLHF 没有这个机制**。如果 weight sync 静默失败（比如 NCCL broadcast 超时但没报错），vLLM 会用旧权重继续生成，训练会用新权重做 PPO update → ratio 严重偏移 → NaN。

**实践建议**: 在 `broadcast_to_vllm` 后加一个 sanity check——随机取一个参数，比较 Actor 侧和 vLLM 侧的值。

### 4.2 verl 的"Rollout Correction"预设

verl 把 IS 修正做成了配置化的预设：
```python
# RolloutCorrectionConfig
class RolloutCorrectionConfig:
    method: str = "tis"  # tis/icepop/seq_mask_tis/bypass
    low: float = 0.1
    high: float = 10.0
```

**OpenRLHF 的 IS 修正参数散布在 CLI 参数中**，不如 verl 直观。但原理相同。

### 4.3 ROLL 的@register 装饰器 vs OpenRLHF 的 RayActorGroup

ROLL 用装饰器声明分发模式：
```python
@register(dispatch_mode=DispatchMode.DP_MP_COMPUTE)
def forward_step(self, data): ...
```

OpenRLHF 用 `async_run_method_batch` 手动分发：
```python
refs = actor_group.async_run_method_batch("forward", data_list)
```

**ROLL 更声明式，OpenRLHF 更命令式**。ROLL 的方式更容易做自动化优化（如自动选择分发策略），OpenRLHF 的方式更灵活但需要手动管理。

---

## 五、4090 实战中的预期问题与解法

### 5.1 Phase 1 SFT: `flash-attn` 编译失败

**问题**: 4090 的 CUDA compute capability 是 8.9，flash-attn 需要对应的编译参数。

**解法**:
```bash
# 方法1: 预编译 wheel
pip install flash-attn --no-build-isolation

# 方法2: 从源码编译
MAX_JOBS=4 pip install flash-attn --no-build-isolation
# 如果失败，检查 CUDA_HOME 是否正确设置
```

### 5.2 Phase 4 GRPO: vLLM 和 DeepSpeed 争抢 GPU 内存

**问题**: sleep/wake 切换时，如果一方没有完全释放内存，另一方会 OOM。

**解法**:
```bash
# 确保两个 sleep 都启用
--vllm.enable_sleep
--ds.enable_sleep

# 降低 vLLM 的内存预算
--vllm.gpu_memory_utilization 0.5

# 如果还是 OOM，降低 max_tokens_per_gpu
--train.max_tokens_per_gpu 8192
```

### 5.3 Phase 5 PPO: 4 个模型的 GPU 分配

**问题**: 8 张 4090 要放 Actor + Critic + Ref + Reward + vLLM，怎么分？

**解法 (colocate_all 模式)**:
```
所有 8 张 GPU 共享所有模型
时间线:
1. vLLM 生成 (8 个 engine，每个 TP=1)
2. vLLM sleep
3. Reward forward (8 个 rank，ZeRO-3 分片)
4. Reward 结果收集
5. Actor forward (计算 log probs)
6. Ref forward (计算 KL)
7. Actor offload → Critic reload
8. Critic forward (计算 values)
9. Critic offload → Actor reload
10. Actor train (PPO update)
11. Actor offload
12. vLLM wake → weight sync → 生成
```

### 5.4 Phase 6: 7B 模型的 Adam Offload

**问题**: 7B 模型的 Adam optimizer states ~56GB（m + v + master weights），8 GPU 每卡 ~7GB，加上权重 ~2GB/GPU (ZeRO-3) = ~9GB。再加上激活 ~4-6GB = ~13-15GB。接近 24GB 上限。

**解法**: `--ds.adam_offload` 把 optimizer states 放 CPU，GPU 上只剩权重 + 激活 = ~6-8GB。

**代价**: CPU-GPU 数据传输 ~2-3 秒/step。

---

## 六、面试中可以展示的"实战经验"

### 6.1 "我在 4090 上跑过完整的 RLHF pipeline"

**具体可以说**:
- "我用 8×4090 跑通了 Qwen2.5-1.5B 的 SFT → RM → DPO → GRPO 全流程"
- "4090 没有 NVLink，所以我优先用 data parallel 而不是 tensor parallel"
- "7B 模型必须用 `colocate_all` + `sleep` 模式，vLLM 和 DeepSpeed 交替占用 GPU"
- "我遇到过 flash-attn 编译失败、vLLM sleep/wake OOM、NCCL 通信超时等问题，分别通过 X/Y/Z 解决"

### 6.2 "我做过算法对比实验"

**具体可以说**:
- "我对比了 PPO(GAE)、GRPO(group_norm)、REINFORCE++-baseline 三种算法"
- "在数学推理任务上，REINFORCE++-baseline 最稳定，GRPO 收敛最快但方差大"
- "n_samples_per_prompt=8 是 1.5B 模型的甜蜜点，从 2→8 提升显著，8→16 边际递减"

### 6.3 "我理解工程细节"

**具体可以说**:
- "OpenRLHF 的 loss 聚合要乘以 `dp_size` 来抵消 DDP 的自动平均"
- "action_mask 右移 1 位是为了自回归对齐——位置 t 的 logit 预测 token t+1"
- "DAPO dynamic filtering 在数据太简单时会不断过滤直到 dataloader exhausted"
- "vLLM 的 `enforce_eager` 在 24GB 卡上必须开，否则 CUDA Graphs 的预分配会 OOM"

---

## 七、增量知识点索引

| 编号 | 知识点 | 来源 | 前两轮是否覆盖 |
|------|--------|------|---------------|
| 1 | `dp_size` 乘法的分布式训练含义 | loss.py:32 | ❌ 新增 |
| 2 | 全 padding chunk 的 NaN 保护 | loss.py:70-73 | ❌ 新增 |
| 3 | Action mask 右移的 off-by-one 逻辑 | samples_generator.py:267 | ⚠️ 之前只说了"要移"，没说为什么 |
| 4 | Replay buffer 的 optimizer step 控制 | replay_buffer.py:140 | ❌ 新增 |
| 5 | `sample_loss_scale` 的梯度加权 | replay_buffer.py:145 | ❌ 新增 |
| 6 | PCIe 拓扑对 ZeRO-3 的实际影响 | 硬件分析 | ❌ 新增 |
| 7 | `enforce_eager` 在小显存卡上的必要性 | vllm_engine.py | ❌ 新增 |
| 8 | `gpu_memory_utilization` 的两阶段行为 | vllm_engine.py:278 | ⚠️ 之前只说了"设 0.5" |
| 9 | Experience 的 `tensor_field.role` 元数据 | experience.py | ❌ 新增 |
| 10 | `balance_experiences` 的 index 追踪 | experience_maker.py:244 | ❌ 新增 |
| 11 | DAPO 过滤的 replacement 循环风险 | samples_generator.py:185 | ❌ 新增 |
| 12 | ProRL 截断惩罚的两种模式 | length_penalty.py:78 | ⚠️ 之前只提了概念 |
| 13 | slime 的权重版本号 vs OpenRLHF 的缺失 | 跨框架对比 | ❌ 新增 |
| 14 | ROLL 的@register vs OpenRLHF 的命令式分发 | 跨框架对比 | ❌ 新增 |
| 15 | 7B 模型的 Adam offload 显存计算 | 4090 实战 | ❌ 新增 |
| 16 | vLLM sleep/wake 的两阶段内存行为 | vllm_engine.py:138-151 | ⚠️ 之前只说了"交替占用" |
| 17 | NCCL IPC 在 colocate 模式下的零拷贝机制 | ppo_actor.py:438-465 | ⚠️ 之前只说了"更快" |
| 18 | 4090 上 7B 模型的完整时间线分析 | 4090 实战 | ❌ 新增 |

---

> **使用建议**: 这份文档是前两轮的"补丁"。面试前快速扫一遍编号 1-18 的新增点，特别是 1/3/4/7/15 这几个最容易被追问的工程细节。
