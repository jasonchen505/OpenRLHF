# OpenRLHF 8×4090 全流程复现 Plan

> 硬件：8× RTX 4090 (24GB VRAM, PCIe Gen4, 无 NVLink)
> 目标：完整复现 OpenRLHF 的 SFT → RM → DPO → PPO/GRPO 全流程
> 模型选择：以 1.5B-7B 为主，最大化学习覆盖面

---

## 一、硬件约束分析

### 1.1 RTX 4090 关键参数

| 参数 | 值 | 影响 |
|------|-----|------|
| VRAM | 24 GB | 限制模型大小和 batch size |
| 互联 | PCIe Gen4 x16 (~32 GB/s) | TP 通信慢，不推荐 DeepSpeed TP |
| BF16 | 支持 (Ada Lovelace) | 可用 bf16 训练 |
| FP8 | 支持 | 可用于推理加速 |
| TFLOPS (BF16) | ~82.6 | 训练速度尚可 |

### 1.2 各任务模型大小上限

| 任务 | 全量微调 | LoRA | QLoRA |
|------|---------|------|-------|
| SFT | 8B ✓ / 13B △ | 13B ✓ / 34B △ | 34B ✓ / 70B △ |
| RM | 8B ✓ / 13B △ | 13B ✓ / 34B △ | 34B ✓ |
| DPO | 8B ✓ (需ref offload) | 13B ✓ / 34B △ | 34B ✓ |
| PPO (含Critic) | 7B ✓ (colocate+sleep) | 13B △ | - |
| GRPO (无Critic) | 8B ✓ (colocate+sleep) | 13B ✓ | - |
| REINFORCE++ | 8B ✓ | 13B ✓ | - |

✓ = 可行，△ = 需要仔细调参

### 1.3 关键约束

1. **无 NVLink**: 不推荐 `--ds.tensor_parallel_size > 1` (DeepSpeed TP 通信慢)
2. **24 GB 限制**: 7B+ 模型必须用 `colocate_all` + `sleep` 模式
3. **PCIe 带宽**: vLLM 的 `--vllm.tensor_parallel_size` 可以用(推理通信较少)，但会慢 20-30%

---

## 二、推荐模型选择

### 2.1 主力模型：Qwen2.5-1.5B / Qwen2.5-3B

**选择理由**:
- 1.5B/3B 在 8×4090 上非常舒适，所有任务都能跑
- 支持长上下文训练(配合 Ring Attention)
- 社区活跃，权重公开可用
- 从 1.5B 开始验证 pipeline，再升级到 3B/7B

### 2.2 进阶模型：Qwen2.5-7B / Llama-3-8B

**选择理由**:
- 7B/8B 是业界主流推理/后训练的基准模型
- 验证 sleep/wake、colocate 等高级特性
- 需要仔细调参 (ZeRO-3 + Adam offload)

### 2.3 模型 ID

| 模型 | HuggingFace ID | 参数量 | BF16 大小 |
|------|---------------|--------|----------|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | 1.5B | ~3 GB |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | 3B | ~6 GB |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | 7B | ~14 GB |
| DeepSeek-R1-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3 GB |

---

## 三、分阶段复现 Plan

### Phase 0: 环境搭建 (Day 1)

**目标**: 安装 OpenRLHF 及所有依赖，验证 GPU 识别。

**步骤**:

```bash
# 0.1 克隆项目
cd /data/home/yizhou
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF

# 0.2 创建 conda 环境
conda create -n openrlhf python=3.10 -y
conda activate openrlhf

# 0.3 安装 PyTorch (CUDA 12.1)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 0.4 安装 OpenRLHF
pip install -e .

# 0.5 安装 Flash Attention 2
pip install flash-attn --no-build-isolation

# 0.6 验证 GPU
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# 0.7 验证 Ray
ray status

# 0.8 验证 vLLM
python -c "import vllm; print(vllm.__version__)"
```

**验证标准**: 8 张 4090 全部识别，vLLM 和 DeepSpeed 版本正确。

**预期问题与解决**:
- Flash Attention 编译失败: 确保 CUDA toolkit 版本匹配
- vLLM 版本不兼容: 按 requirements.txt 指定版本
- Ray 集群初始化失败: 检查端口占用

---

### Phase 1: SFT — 指令微调 (Day 2-3)

**目标**: 用 OpenRLHF 的 SFT pipeline 微调 Qwen2.5-1.5B，理解训练流程。

**1.1 准备数据集**

OpenRLHF 支持 HuggingFace datasets 格式。使用 `OpenRLHF/SFT` 格式的数据：

```python
# 数据格式示例 (JSON)
{
    "instruction": "请解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
}
```

或者直接用 HuggingFace 上的公开 SFT 数据集，如 `OpenRLHF/SFT`。

**1.2 运行 SFT 训练**

```bash
# 基于 examples/scripts/train_sft.sh 改编
deepspeed --module openrlhf.cli.train_sft \
   --pretrain Qwen/Qwen2.5-1.5B \
   --dataset <your_dataset> \
   --input_key instruction \
   --output_key output \
   --max_len 2048 \
   --micro_batch_size 4 \
   --batch_size 64 \
   --max_epochs 1 \
   --save_path ./checkpoint/sft-qwen2.5-1.5b \
   --save_steps 100 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --learning_rate 5e-6 \
   --lr_scheduler cosine \
   --gradient_checkpointing \
   --packing_samples \
   --bf16 \
   --flash_attn
```

**1.3 学习要点**

| 学习点 | 要理解的内容 | 代码位置 |
|--------|-------------|---------|
| DeepSpeed ZeRO-2 | 优化器状态分片如何节省显存 | `utils/deepspeed/deepspeed.py` |
| Sample Packing | FlashAttention 2 如何 pack 多条序列 | `datasets/sft_dataset.py` |
| 梯度检查点 | 用计算换显存的机制 | `--gradient_checkpointing` |
| Loss 计算 | `GPTLMLoss` 如何处理 ignore_index | `models/loss.py:60-80` |

**验证标准**: Loss 正常下降，checkpoint 正确保存，eval 指标合理。

---

### Phase 2: Reward Model 训练 (Day 4-5)

**目标**: 训练一个 Reward Model，理解偏好学习。

**2.1 准备偏好数据**

```json
{
    "instruction": "请解释量子计算",
    "chosen": "量子计算利用量子力学原理...",
    "rejected": "量子计算就是很快的计算机..."
}
```

使用 `Anthropic/hh-rlhf` 或 `OpenRLHF/rm-mixture` 等公开数据集。

**2.2 运行 RM 训练**

```bash
deepspeed --module openrlhf.cli.train_rm \
   --pretrain Qwen/Qwen2.5-1.5B \
   --dataset <preference_dataset> \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_len 2048 \
   --micro_batch_size 2 \
   --batch_size 64 \
   --max_epochs 1 \
   --save_path ./checkpoint/rm-qwen2.5-1.5b \
   --save_steps 100 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --bf16 \
   --flash_attn
```

**2.3 学习要点**

| 学习点 | 要理解的内容 | 代码位置 |
|--------|-------------|---------|
| PairWise Loss | chosen vs rejected 的 Bradley-Terry 模型 | `models/loss.py:270-300` |
| Value Head | `nn.Linear(hidden_size, 1)` 如何添加 | `models/model.py:177-242` |
| EOS 提取 | 为什么 reward 在 EOS 位置提取 | `models/model.py:220-230` |
| Reward Normalization | 如何统计 mean/std 并保存到 config | `trainer/rm_trainer.py` |
| ZeRO-3 | 参数分片如何工作 | `utils/deepspeed/deepspeed.py` |

**验证标准**: RM 在 held-out 数据上的 accuracy > 65%。

---

### Phase 3: DPO — 直接偏好优化 (Day 6-7)

**目标**: 用 DPO 训练，理解离线偏好学习。

**3.1 运行 DPO 训练**

```bash
deepspeed --module openrlhf.cli.train_dpo \
   --pretrain Qwen/Qwen2.5-1.5B \
   --dataset <preference_dataset> \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_len 2048 \
   --micro_batch_size 1 \
   --batch_size 32 \
   --max_epochs 1 \
   --save_path ./checkpoint/dpo-qwen2.5-1.5b \
   --beta 0.1 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --gradient_checkpointing \
   --packing_samples \
   --bf16 \
   --ref.offload
```

**3.2 学习要点**

| 学习点 | 要理解的内容 | 代码位置 |
|--------|-------------|---------|
| DPO Loss | 隐式 reward 的计算 | `models/loss.py:301-336` |
| Reference Model | 为什么需要 ref，如何 offload | `train_dpo.py:42-50` |
| β 参数 | 控制策略偏移程度 | `--beta 0.1` |
| IPO 变体 | MSE loss 替代 log-sigmoid | `loss.py:324` |

**验证标准**: DPO loss 下降，chosen reward > rejected reward。

---

### Phase 4: GRPO — 无 Critic 的 RL (Day 8-12)

**目标**: 用 GRPO 训练 reasoning 模型，这是当前最主流的后训练方法。

**4.1 为什么先 GRPO 而不是 PPO**

- 不需要 Critic 网络 → 省显存，更容易在 4090 上跑
- 被 DeepSeek-R1、Qwen3 等模型采用
- 是当前 reasoning 模型训练的主流方案

**4.2 准备 Reward Function**

对于数学推理任务，使用 rule-based reward:

```python
# examples/python/math_reward_func.py 的简化版
def reward_func(queries, prompts, labels, **kwargs):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        # 提取答案并比较
        pred = extract_answer(query)
        gold = extract_answer(label)
        rewards.append(1.0 if pred == gold else 0.0)
    return {
        "rewards": torch.tensor(rewards),
        "scores": torch.tensor(rewards),
        "extra_logs": {}
    }
```

**4.3 准备推理数据集**

使用 GSM8K 或 MATH 数据集：
```json
{"prompt": "Solve: If x + 3 = 7, what is x?", "label": "4"}
```

**4.4 运行 GRPO 训练 (1.5B 模型, 8 GPU 全量)**

```bash
python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 --ref.num_gpus_per_node 8 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 8 \
   --vllm.num_engines 8 --vllm.tensor_parallel_size 1 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.75 \
   --vllm.enable_sleep --ds.enable_sleep \
   --actor.model_name_or_path Qwen/Qwen2.5-1.5B \
   --reward.model_name_or_path <your_rm_path> \
   --critic.model_name_or_path None \
   --algo.advantage.estimator group_norm \
   --rollout.n_samples_per_prompt 8 \
   --rollout.temperature 1.0 \
   --rollout.max_new_tokens 1024 \
   --data.max_len 2048 \
   --train.batch_size 128 \
   --rollout.batch_size 1024 \
   --actor.optim adam \
   --actor.adam.lr 1e-6 \
   --ds.zero_stage 3 \
   --ds.packing_samples \
   --actor.gradient_checkpointing_enable \
   --train.dynamic_batch_enable \
   --train.max_tokens_per_gpu 16384 \
   --reward.clip_range -10 10 \
   --algo.kl.init_coef 0.01 \
   --ckpt.save_steps 50 \
   --save_path ./checkpoint/grpo-qwen2.5-1.5b
```

**4.5 学习要点**

| 学习点 | 要理解的内容 | 代码位置 |
|--------|-------------|---------|
| Group Norm Advantage | 如何用组内统计量替代 Critic | `experience_maker.py:284-291` |
| Sleep/Wake 生命周期 | vLLM 和 DeepSpeed 如何交替占用 GPU | `ppo_trainer.py:266-320` |
| Weight 同步 | Actor 权重如何广播到 vLLM | `ray/ppo_actor.py:408-489` |
| DAPO Dynamic Filtering | 如何过滤全对/全错的 prompt | `samples_generator.py:169-195` |
| KL 控制 | Adaptive KL Controller 如何工作 | `kl_controller.py` |
| Ray 分布式 | ActorGroup 如何管理多个 Ray Actor | `ray/launcher.py:202-373` |
| 序列长度平衡 | Karmarkar-Karp 算法 | `utils/seqlen_balancing.py` |

**验证标准**: reward_mean 逐步上升，KL 在 target 附近，response_length 稳定。

---

### Phase 5: PPO — 完整 RLHF (Day 13-17)

**目标**: 跑通含 Critic 的完整 PPO 流程。

**5.1 需要额外训练 Critic Model**

Critic Model 和 RM 架构相同(Transformer + value head)，但训练方式不同：
- RM: 冻结，用于评分
- Critic: 训练中更新，用于 GAE 价值估计

**5.2 运行 PPO 训练 (1.5B 模型)**

```bash
python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 --ref.num_gpus_per_node 8 \
   --reward.num_nodes 1 --reward.num_gpus_per_node 8 \
   --critic.num_nodes 1 --critic.num_gpus_per_node 8 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 8 \
   --vllm.num_engines 4 --vllm.tensor_parallel_size 2 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.5 \
   --vllm.enable_sleep --ds.enable_sleep \
   --actor.model_name_or_path Qwen/Qwen2.5-1.5B \
   --critic.model_name_or_path <your_rm_path> \
   --reward.model_name_or_path <your_rm_path> \
   --algo.advantage.estimator gae \
   --algo.advantage.gamma 1.0 \
   --algo.advantage.lambd 1.0 \
   --rollout.n_samples_per_prompt 1 \
   --data.max_len 2048 \
   --train.batch_size 128 \
   --rollout.batch_size 1024 \
   --ds.zero_stage 3 \
   --ds.packing_samples \
   --actor.gradient_checkpointing_enable \
   --train.dynamic_batch_enable \
   --train.max_tokens_per_gpu 16384 \
   --actor.adam.lr 1e-6 \
   --critic.adam.lr 1e-6 \
   --actor.eps_clip 0.2 \
   --ckpt.save_steps 50 \
   --save_path ./checkpoint/ppo-qwen2.5-1.5b
```

**5.3 学习要点**

| 学习点 | 要理解的内容 | 代码位置 |
|--------|-------------|---------|
| GAE 计算 | γ 和 λ 如何影响 bias-variance | `experience_maker.py:332-377` |
| Critic 训练 | Value Loss 的 clipping 机制 | `models/loss.py:238-260` |
| PPO Clipping | 双向 clip + dual-clip | `models/loss.py:166-194` |
| Experience Making | 4 个模型的 forward 如何编排 | `experience_maker.py:112-233` |
| Sleep 模式下 Critic 和 Actor 的交替训练 | Sequential vs Parallel | `ppo_trainer.py:266-298` |

**验证标准**: policy_loss 下降，value_loss 下降，clip_ratio 在 0.1-0.4。

---

### Phase 6: 进阶 — 7B 模型 + 高级特性 (Day 18-25)

**6.1 升级到 Qwen2.5-7B**

```bash
# SFT: 7B 全量微调
deepspeed --module openrlhf.cli.train_sft \
   --pretrain Qwen/Qwen2.5-7B \
   --dataset <dataset> \
   --max_len 2048 \
   --micro_batch_size 1 \
   --batch_size 32 \
   --zero_stage 3 \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --bf16
```

**6.2 尝试 LoRA 训练**

```bash
deepspeed --module openrlhf.cli.train_sft \
   --pretrain Qwen/Qwen2.5-7B \
   --dataset <dataset> \
   --max_len 4096 \
   --micro_batch_size 4 \
   --batch_size 32 \
   --zero_stage 2 \
   --lora_rank 64 \
   --lora_alpha 64 \
   --gradient_checkpointing \
   --packing_samples \
   --bf16
```

**6.3 尝试 REINFORCE++-baseline (长上下文)**

```bash
python3 -m openrlhf.cli.train_ppo_ray \
   --actor.model_name_or_path Qwen/Qwen2.5-3B \
   --ref.num_nodes 1 --ref.num_gpus_per_node 4 \
   --actor.num_nodes 1 --actor.num_gpus_per_node 4 \
   --vllm.num_engines 2 --vllm.tensor_parallel_size 2 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.7 \
   --vllm.enable_sleep --ds.enable_sleep \
   --algo.advantage.estimator reinforce_baseline \
   --rollout.n_samples_per_prompt 8 \
   --ds.ring_attn_size 2 \
   --ds.ring_attn_head_stride 2 \
   --ds.packing_samples \
   --data.max_len 8192 \
   --rollout.max_new_tokens 4096 \
   --ds.zero_stage 3 \
   --actor.gradient_checkpointing_enable \
   --train.dynamic_batch_enable \
   --train.max_tokens_per_gpu 16192
```

**6.4 尝试异步训练**

```bash
# 在 GRPO 脚本基础上添加:
--train.async_enable \
--train.async_generate_queue_size 2
```

**6.5 学习要点**

| 学习点 | 要理解的内容 |
|--------|-------------|
| Adam Offload | 优化器状态 CPU offload 的开销和收益 |
| LoRA | 低秩适配的原理和工程实现 |
| QLoRA | 4-bit 量化 + LoRA 的组合 |
| Ring Attention | 长序列如何跨 GPU 分片 |
| 异步训练 | 生产者-消费者架构、回压机制 |
| IS 修正 | off-policy 训练的重要性采样修正 |
| GSPO | 序列级 ratio 的设计动机 |

---

### Phase 7: 高级实验与对比 (Day 26-30)

**7.1 算法对比实验**

```
固定: Qwen2.5-1.5B, 同一数据集, 同一 reward function
变量: 优势估计器
├── PPO (GAE)
├── GRPO (group_norm)
├── REINFORCE++-baseline
└── RLOO
比较: reward 曲线, eval pass@k, 训练稳定性, 显存占用
```

**7.2 n_samples 对比实验**

```
GRPO with n_samples ∈ {2, 4, 8, 16}
比较: 收敛速度, 推理成本, reward 方差
```

**7.3 KL 系数对比实验**

```
kl_coef ∈ {0.001, 0.01, 0.05, 0.1}
比较: KL 轨迹, reward vs KL Pareto 曲线
```

**7.4 生成超参对比**

```
temperature ∈ {0.5, 0.7, 1.0, 1.3}
top_p ∈ {0.9, 0.95, 1.0}
比较: 生成多样性, reward 分布, 探索效率
```

---

## 四、关键超参速查表 (4090 专用)

### 4.1 SFT

| 参数 | 1.5B | 3B | 7B |
|------|------|-----|-----|
| zero_stage | 2 | 2 | 3 + adam_offload |
| micro_batch_size | 8 | 4 | 1 |
| batch_size | 64 | 64 | 32 |
| max_len | 2048 | 2048 | 2048 |
| gradient_checkpointing | 可选 | 推荐 | 必须 |
| packing_samples | 是 | 是 | 是 |
| lora_rank | 0 (全量) | 0 | 64 (可选) |

### 4.2 RM

| 参数 | 1.5B | 3B | 7B |
|------|------|-----|-----|
| zero_stage | 3 | 3 | 3 |
| micro_batch_size | 4 | 2 | 1 |
| max_len | 2048 | 2048 | 2048 |

### 4.3 DPO

| 参数 | 1.5B | 3B | 7B |
|------|------|-----|-----|
| zero_stage | 3 | 3 | 3 |
| micro_batch_size | 4 | 2 | 1 |
| ref.offload | 可选 | 推荐 | 必须 |
| beta | 0.1 | 0.1 | 0.1 |

### 4.4 GRPO/PPO (colocate_all + sleep)

| 参数 | 1.5B | 3B | 7B |
|------|------|-----|-----|
| vllm.num_engines | 8 | 4 | 4 |
| vllm.tensor_parallel_size | 1 | 1 | 2 |
| gpu_memory_utilization | 0.75 | 0.7 | 0.5 |
| zero_stage | 3 | 3 | 3 |
| adam_offload | 否 | 否 | 是 |
| n_samples_per_prompt | 8-16 | 8 | 8 |
| max_tokens_per_gpu | 16384 | 16384 | 16384 |

---

## 五、常见问题与解决

### 5.1 OOM (Out of Memory)

```
症状: RuntimeError: CUDA out of memory
排查:
1. 确认用了 --zero_stage 3
2. 确认用了 --gradient_checkpointing
3. 减小 --micro_batch_size
4. 减小 --max_len
5. 添加 --adam_offload
6. 减小 --vllm.gpu_memory_utilization
7. 使用 LoRA (--lora_rank 16)
```

### 5.2 vLLM 启动失败

```
症状: vLLM engine initialization failed
排查:
1. 检查 vLLM 版本 (>0.8.5)
2. 检查 CUDA 版本匹配
3. 减小 gpu_memory_utilization
4. 检查是否有其他进程占用 GPU
```

### 5.3 NCCL 通信超时

```
症状: NCCL timeout / watchdog timeout
排查:
1. 设置 NCCL_DEBUG=INFO 查看详细日志
2. 检查 PCIe 拓扑: nvidia-smi topo -m
3. 减小 batch_size 降低通信频率
4. 确认没有其他进程干扰
```

### 5.4 训练 Loss 为 NaN

```
症状: loss 变成 NaN 或 Inf
排查:
1. 降低 learning rate
2. 检查 reward 是否有极端值 (加 --reward.clip_range -10 10)
3. 检查 gradient norm (加 --actor.max_norm 1.0)
4. 检查数据是否有问题
```

### 5.5 Ray Actor 崩溃

```
症状: Ray actor died unexpectedly
排查:
1. 检查单个 actor 的日志: /tmp/ray/session_latest/logs/
2. 确认 GPU 内存没有被其他进程占用
3. 减小 num_engines 降低 GPU 争用
4. 检查是否 OOM 导致进程被 kill
```

---

## 六、学习检查清单

### Phase 0-1 (环境 + SFT)
- [ ] 理解 DeepSpeed ZeRO-0/1/2/3 的区别
- [ ] 理解 gradient checkpointing 的 trade-off
- [ ] 理解 sample packing 的实现原理
- [ ] 能解释 SFT loss 和普通 CE loss 的区别

### Phase 2-3 (RM + DPO)
- [ ] 理解 Bradley-Terry 偏好模型
- [ ] 理解 DPO 的隐式 reward 推导
- [ ] 理解 reference model 的作用
- [ ] 理解 β 参数的含义

### Phase 4 (GRPO)
- [ ] 能画出 sleep/wake 的时序图
- [ ] 理解 weight 同步的 NCCL/CUDA IPC 两条路径
- [ ] 理解 group normalization advantage 的方差问题
- [ ] 理解 DAPO dynamic filtering 的动机
- [ ] 理解 KL penalty 在 reward shaping 中的作用

### Phase 5 (PPO)
- [ ] 能手写 GAE 的伪代码
- [ ] 理解 PPO clipping 和 dual-clip 的区别
- [ ] 理解 critic 和 reward model 的区别
- [ ] 能解释 experience making 的 4 步流程

### Phase 6-7 (进阶)
- [ ] 理解 LoRA 的数学原理和工程实现
- [ ] 理解 Ring Attention 如何处理长序列
- [ ] 理解异步训练的生产者-消费者架构
- [ ] 能对比 GRPO/PPO/REINFORCE++/RLOO 的 trade-off
- [ ] 能设计一个完整的 RLHF 实验方案

---

## 七、文件清单

完成所有 Phase 后，你应该有以下产出:

```
checkpoint/
├── sft-qwen2.5-1.5b/       # Phase 1: SFT 模型
├── rm-qwen2.5-1.5b/        # Phase 2: Reward Model
├── dpo-qwen2.5-1.5b/       # Phase 3: DPO 模型
├── grpo-qwen2.5-1.5b/      # Phase 4: GRPO 模型
└── ppo-qwen2.5-1.5b/       # Phase 5: PPO 模型

logs/
├── sft/                     # WandB 日志
├── rm/
├── dpo/
├── grpo/
└── ppo/

results/
├── algorithm_comparison/    # Phase 7: 算法对比
├── n_samples_ablation/      # Phase 7: n_samples 消融
└── kl_coef_ablation/        # Phase 7: KL 系数消融
```

---

## 八、时间线总结

| 阶段 | 时间 | 任务 | 关键产出 |
|------|------|------|---------|
| Phase 0 | Day 1 | 环境搭建 | 可运行的 OpenRLHF 环境 |
| Phase 1 | Day 2-3 | SFT | 微调后的 1.5B 模型 |
| Phase 2 | Day 4-5 | RM | 训练好的 Reward Model |
| Phase 3 | Day 6-7 | DPO | DPO 对齐后的模型 |
| Phase 4 | Day 8-12 | GRPO | GRPO 训练的推理模型 |
| Phase 5 | Day 13-17 | PPO | 完整 PPO 训练的模型 |
| Phase 6 | Day 18-25 | 进阶 | 7B 模型 + LoRA + 异步训练 |
| Phase 7 | Day 26-30 | 实验对比 | 消融实验结果 |

**总计**: ~30 天，完整复现 OpenRLHF 的全流程。
