# Experiment Plan

## SVHN 参数配置记录

本项目中 SVHN 实验分为两类：一类是全监督 sanity check / 对照组，用于确认数据读取、模型结构、云边同步和 FedAvg 聚合是否正常；另一类是少标签半监督实验，用于验证 EdgeQGFed 的主要方法效果。两类实验不应混在同一组结论中比较。全监督实验只作为代码正确性和上限参考，论文主实验应以 `labeled_ratio = 0.1` 的半监督设置为主。

### A. SVHN 全监督对照组配置

用途：验证 `identity + small-cnn` 在 SVHN 上是否能正常训练，并检查云端模型和边缘模型的精度是否同步。该组不作为 EdgeQGFed 主方法结果，只用于 sanity check。

```yaml
datasets:
  name: "svhn"
  normalize: "half"
  batch_size: 64
  labeled_ratio: 1.0
  distributed: "IID"
  partition_mode: "client_dirichlet"
  partition_alpha: 0.3
  min_client_size: 16
  edge_num_workers: 4
  eval_num_workers: 2
  pin_memory: True

models:
  encoder_name: "identity"
  classifier_name: "small-cnn"
  ema_decay: 0.0
  ema_dynamic_decay: False
  algorithm: "FedAvg"
  attack_rate: 0.0
  graph:
    use: False

train:
  learning_rate: 1.0e-3
  total_steps: 1000
  log_step: 20
  evaluate_step: 100
  ema_update_step: 1
  class_fn: "CE"
  consis_fn: "MSE"
  optimizer: "AdamW"
  initial_weight: 0.0
  final_weight: 0.0
  ramp_up_steps: 1000
  pseudo:
    use: False
    weight: 0.0

topology:
  num_edges: 4
  num_clients: 128
  clients_per_round: 32
  edges_per_round: 4
  client_sample_mode: "balanced"
  assignment: "round_robin"
  client_drop_rate: 0.0

network:
  round_comm_budget_mb: 1200
  client_image_size: 32
  client_input_channels: 3
  client_view_count: 3
  bytes_per_value: 4
  bandwidth:
    mode: "edge_profile"
    client_uplink_by_edge_mbps: [40, 60, 80, 120]
    edge_uplink_by_edge_mbps: [200, 300, 400, 600]
    cloud_downlink_by_edge_mbps: [400, 600, 800, 1000]
  mobility:
    mode: "static"
  max_parallel_uploads_per_edge: 8
  target_accuracy: 90.0
```

记录指标：

```text
cloud_accuracy
edge_avg_accuracy
cloud_loss
edge_avg_loss
edge_total_loss
edge_class_loss
total_comm_mb
formal_round_latency_s
cumulative_estimated_latency_s
round_wall_clock_s
```

判断标准：

```text
1. edge_total_loss 和 edge_class_loss 应明显下降。
2. edge_avg_accuracy 应显著高于随机水平。
3. cloud_accuracy 应与 edge_avg_accuracy 接近；如果 cloud 仍接近 10%，优先检查云边 state_dict 同步和 BatchNorm buffer。
4. avg_pseudo_ratio、avg_pseudo_weight、avg_agreement_ratio 可以为 0，因为该组没有无标签样本且 pseudo.use=False。
```

建议表格：

```text
行: Full-supervised FedAvg sanity
列: Dataset, Labeled Ratio, Partition, Graph, Pseudo, Cloud Accuracy, Edge Avg Accuracy, Cloud Loss, Edge Avg Loss
```

建议绘图：

```text
图 1:
横轴: Communication Round
纵轴: Accuracy
曲线: cloud_accuracy, edge_avg_accuracy

图 2:
横轴: Communication Round
纵轴: Loss
曲线: cloud_loss, edge_avg_loss, edge_class_loss
```

### B. SVHN 半监督主实验配置

用途：作为论文主实验的默认配置，验证少标签、non-IID、通信受限和云端图注意力聚合条件下的 EdgeQGFed 效果。

```yaml
datasets:
  name: "svhn"
  normalize: "half"
  batch_size: 64
  labeled_ratio: 0.1
  distributed: "nonIID"
  partition_mode: "client_dirichlet"
  partition_alpha: 0.3
  min_client_size: 16
  edge_num_workers: 4
  eval_num_workers: 2
  pin_memory: True

models:
  encoder_name: "identity"
  classifier_name: "small-cnn"
  ema_decay: 0.0
  ema_dynamic_decay: False
  algorithm: "FedAvg"
  attack_rate: 0.0
  graph:
    use: True
    temperature: 0.5
    prototype_weight: 0.35
    reliability_weight: 0.35
    label_ratio_weight: 0.2
    confidence_weight: 0.1
    self_bias: 0.3
    min_attention: 1.0e-4

train:
  learning_rate: 1.0e-3
  total_steps: 2000
  log_step: 20
  evaluate_step: 100
  ema_update_step: 1
  class_fn: "CE"
  consis_fn: "MSE"
  optimizer: "AdamW"
  initial_weight: 0.0
  final_weight: 1.0
  ramp_up_steps: 300
  warm_mode: "gaussian"
  pseudo:
    use: True
    weight: 0.5
    edge_threshold: 0.55
    cloud_threshold: 0.60
    temperature: 1.0
    weight_temperature: 0.1
    min_weight: 0.05

topology:
  num_edges: 4
  num_clients: 128
  clients_per_round: 32
  edges_per_round: 4
  client_sample_mode: "resource_aware"
  assignment: "round_robin"
  client_drop_rate: 0.0

network:
  round_comm_budget_mb: 1200
  client_image_size: 32
  client_input_channels: 3
  client_view_count: 3
  bytes_per_value: 4
  bandwidth:
    mode: "edge_profile"
    client_uplink_by_edge_mbps: [40, 60, 80, 120]
    edge_uplink_by_edge_mbps: [200, 300, 400, 600]
    cloud_downlink_by_edge_mbps: [400, 600, 800, 1000]
  mobility:
    mode: "static"
  resource_sampling:
    exploration_rate: 0.05
    label_ratio_weight: 0.35
    data_size_weight: 0.25
    bandwidth_weight: 0.25
    availability_weight: 0.15
    latency_cost_weight: 1.0
  max_parallel_uploads_per_edge: 8
  target_accuracy: 75.0
```

半监督实验的 non-IID 强度至少包含两组：

```text
partition_alpha = 0.3: medium non-IID
partition_alpha = 0.1: strong non-IID
```

半监督对照顺序：

```text
1. labeled_ratio=0.1, pseudo=False, graph=False: 少标签监督下限
2. labeled_ratio=0.1, pseudo=True, graph=False: 半监督伪标签贡献
3. labeled_ratio=0.1, pseudo=True, graph=True: Full EdgeQGFed
4. labeled_ratio=0.1, pseudo=True, graph=True, client_sample_mode=balanced: 资源感知采样消融
```

记录指标：

```text
cloud_accuracy
edge_avg_accuracy
cloud_loss
edge_avg_loss
edge_total_loss
edge_class_loss
edge_consistency_loss
edge_pseudo_loss
avg_pseudo_ratio
avg_pseudo_weight
avg_agreement_ratio
avg_cloud_confidence
avg_edge_confidence
graph_attention_mean
graph_attention_diag
graph_reliability_mean
graph_label_ratio_mean
graph_confidence_mean
total_comm_mb
formal_round_latency_s
cumulative_estimated_latency_s
time_to_target_accuracy_s
samples_per_estimated_second
client_drop_ratio
```

建议表格：

```text
行: FedAvg, w/o Pseudo, w/o Graph, w/o Resource Sampling, Full EdgeQGFed
列: Method, Alpha, Labeled Ratio, Cloud Accuracy, Edge Avg Accuracy, Time-to-75%, Total Comm MB, Formal Round Latency, Avg Pseudo Weight, Graph Attention Diag
```

建议绘图：

```text
图 1:
横轴: Communication Round
纵轴: Cloud Accuracy
曲线: FedAvg, w/o Pseudo, w/o Graph, Full EdgeQGFed

图 2:
横轴: Cumulative Estimated Latency
纵轴: Cloud Accuracy
曲线: FedAvg, w/o Pseudo, w/o Graph, Full EdgeQGFed

图 3:
横轴: Communication Round
纵轴: Loss
曲线: cloud_loss, edge_avg_loss, edge_consistency_loss, edge_pseudo_loss

图 4:
横轴: Communication Round
纵轴: Pseudo-label Quality
曲线: avg_pseudo_ratio, avg_pseudo_weight, avg_agreement_ratio
```

注意：当前实现中边缘节点上传到云端的是模型状态与质量感知摘要的组合，包括 `parameters/state_dict`、类别原型、类别原型计数、伪标签质量、标签比例和置信度。因此论文表述应写成“边缘节点上传质量感知模型摘要”，不要写成“边缘节点完全不上传模型参数”。

本文实验按论文呈现顺序组织，目标是证明 EdgeQGFed 在 non-IID、少标签、通信受限、带宽异构和客户端掉线条件下的有效性，重点突出通信效率、鲁棒性和 time-to-accuracy。

## 0. 统一实验设置

除特殊说明外，所有实验默认使用以下配置。

```yaml
datasets:
  name: "svhn"
  batch_size: 64
  labeled_ratio: 0.1
  distributed: "nonIID"
  partition_mode: "client_dirichlet"
  partition_alpha: 0.3
  min_client_size: 16

topology:
  num_edges: 4
  num_clients: 128
  clients_per_round: 32
  edges_per_round: 4
  client_sample_mode: "resource_aware"
  client_drop_rate: 0.0

train:
  learning_rate: 1e-4
  total_steps: 2000
  log_step: 20
  evaluate_step: 100
  ema_update_step: 50
  pseudo:
    use: True

models:
  graph:
    use: True

network:
  round_comm_budget_mb: 1200
  bandwidth:
    mode: "edge_profile"
    client_uplink_by_edge_mbps: [40, 60, 80, 120]
    edge_uplink_by_edge_mbps: [200, 300, 400, 600]
    cloud_downlink_by_edge_mbps: [400, 600, 800, 1000]
  mobility:
    mode: "static"
  max_parallel_uploads_per_edge: 8
  target_accuracy: 75.0
```

每组实验建议运行：

```text
seeds = 1, 2, 3
```

汇报均值和标准差。核心指标包括：

```text
Cloud Accuracy
Edge Avg Accuracy
Total Communication MB
Formal Round Latency
Cumulative Estimated Latency
Time-to-Target-Accuracy
Budget Used Ratio
Client Drop Ratio
Samples per Estimated Second
```

## 实验一：主结果对比

目的：证明 EdgeQGFed 相比普通分层联邦学习和关键消融版本更有效。

固定参数：

```yaml
datasets.partition_alpha: 0.3
datasets.labeled_ratio: 0.1
network.round_comm_budget_mb: 1200
network.bandwidth.mode: "edge_profile"
topology.client_drop_rate: 0.0
```

对比方法：

```text
FedAvg / Hierarchical FedAvg
EdgeQGFed w/o Graph
EdgeQGFed w/o Resource Sampling
EdgeQGFed w/o Pseudo Label
Full EdgeQGFed
```

对应配置：

```yaml
# w/o Graph
models.graph.use: False

# w/o Resource Sampling
topology.client_sample_mode: "balanced"

# w/o Pseudo Label
train.pseudo.use: False

# Full EdgeQGFed
models.graph.use: True
topology.client_sample_mode: "resource_aware"
train.pseudo.use: True
```

建议表格：

```text
Method | Accuracy | Time-to-75% | Total Comm MB | Formal Latency | Drop Ratio
```

表格设置：

```text
行: FedAvg / Hierarchical FedAvg, w/o Graph, w/o Resource Sampling, w/o Pseudo Label, Full EdgeQGFed
列: Method, Cloud Accuracy, Edge Avg Accuracy, Time-to-75%, Total Comm MB, Formal Round Latency, Budget Used Ratio, Drop Ratio
```

建议曲线：

```text
Accuracy vs Communication MB
Accuracy vs Cumulative Estimated Latency
```

绘图设置：

```text
图1:
横轴: Cumulative Communication MB
纵轴: Cloud Accuracy
曲线: 各对比方法

图2:
横轴: Cumulative Estimated Latency
纵轴: Cloud Accuracy
曲线: 各对比方法
```

## 实验二：non-IID 强度实验

目的：验证数据异构增强时，图注意力聚合和质量感知机制的作用。

重点比较：

```text
partition_alpha = 0.3
partition_alpha = 0.1
```

含义：

```text
alpha = 0.3: medium non-IID
alpha = 0.1: strong non-IID
```

固定参数：

```yaml
datasets.labeled_ratio: 0.1
network.round_comm_budget_mb: 1200
network.bandwidth.mode: "edge_profile"
topology.client_drop_rate: 0.0
```

对比方法：

```text
Hierarchical FedAvg
EdgeQGFed w/o Graph
Full EdgeQGFed
```

建议表格：

```text
Alpha | Method | Accuracy | Edge Avg Acc | Time-to-75% | Total Comm MB
0.3   | ...
0.1   | ...
```

表格设置：

```text
行: alpha 与 method 的组合
列: Alpha, Method, Cloud Accuracy, Edge Avg Accuracy, Time-to-75%, Total Comm MB, Formal Round Latency
```

绘图设置：

```text
图1:
横轴: Communication Round
纵轴: Cloud Accuracy
曲线: alpha=0.3 与 alpha=0.1 下的各方法

图2:
横轴: Dirichlet Alpha
纵轴: Cloud Accuracy
柱/点: 各对比方法
```

预期论点：当 Dirichlet alpha 从 0.3 降到 0.1 时，数据异构增强，普通平均聚合性能下降更明显，而 EdgeQGFed 的图注意力聚合能更稳定地协调边缘节点。

## 实验三：通信预算敏感性实验

目的：证明 EdgeQGFed 在通信受限条件下更有效。

扫描参数：

```text
round_comm_budget_mb = 300, 600, 900, 1200, 1800
```

固定参数：

```yaml
datasets.partition_alpha: 0.3
datasets.labeled_ratio: 0.1
network.bandwidth.mode: "edge_profile"
topology.client_drop_rate: 0.0
```

对比方法：

```text
Balanced Sampling + Graph
Resource-aware Sampling + Graph
Full EdgeQGFed
```

建议图：

```text
Budget vs Accuracy
Budget vs Time-to-Target
Budget vs Selected Clients
```

建议表格：

```text
Budget MB | Method | Accuracy | Selected Clients | Total Comm MB | Time-to-75%
```

表格设置：

```text
行: round_comm_budget_mb 与 method 的组合
列: Budget MB, Method, Cloud Accuracy, Selected Clients, Active Clients, Total Comm MB, Budget Used Ratio, Time-to-75%
```

绘图设置：

```text
图1:
横轴: Round Communication Budget MB
纵轴: Cloud Accuracy
曲线: 各对比方法

图2:
横轴: Round Communication Budget MB
纵轴: Time-to-75%
曲线: 各对比方法

图3:
横轴: Round Communication Budget MB
纵轴: Selected Clients
曲线: 各对比方法
```

预期论点：在低通信预算下，资源感知采样能够优先选择单位通信收益更高的客户端，从而提高通信效率。

## 实验四：异构带宽实验

目的：评估 EdgeQGFed 在异构链路条件下的训练效率。

### 4.1 固定边缘异构带宽

```yaml
network.bandwidth.mode: "edge_profile"
```

Mild:

```yaml
client_uplink_by_edge_mbps: [60, 80, 100, 120]
edge_uplink_by_edge_mbps: [300, 400, 500, 600]
cloud_downlink_by_edge_mbps: [600, 700, 800, 900]
```

Medium:

```yaml
client_uplink_by_edge_mbps: [40, 60, 80, 120]
edge_uplink_by_edge_mbps: [200, 300, 400, 600]
cloud_downlink_by_edge_mbps: [400, 600, 800, 1000]
```

Severe:

```yaml
client_uplink_by_edge_mbps: [10, 30, 60, 120]
edge_uplink_by_edge_mbps: [80, 150, 300, 600]
cloud_downlink_by_edge_mbps: [200, 400, 700, 1000]
```

### 4.2 随机带宽分布

```yaml
network.bandwidth.mode: "sampled_distribution"
```

Uniform-1:

```yaml
client_distribution:
  type: "uniform"
  min_mbps: 10
  max_mbps: 120
```

Uniform-2:

```yaml
client_distribution:
  type: "uniform"
  min_mbps: 5
  max_mbps: 80
```

LogNormal-1:

```yaml
client_distribution:
  type: "lognormal"
  median_mbps: 40
  sigma: 0.35
  min_mbps: 5
  max_mbps: 120
```

LogNormal-2:

```yaml
client_distribution:
  type: "lognormal"
  median_mbps: 40
  sigma: 0.75
  min_mbps: 5
  max_mbps: 120
```

固定参数：

```yaml
datasets.partition_alpha: 0.3
datasets.labeled_ratio: 0.1
network.round_comm_budget_mb: 1200
```

指标：

```text
Accuracy
Formal Round Latency
Time-to-75%
Samples per Estimated Second
```

表格设置：

```text
行: bandwidth setting 与 method 的组合
列: Bandwidth Setting, Method, Cloud Accuracy, Formal Round Latency, Time-to-75%, Samples per Estimated Second
```

绘图设置：

```text
图1:
横轴: Bandwidth Heterogeneity Level
纵轴: Cloud Accuracy
曲线: 各对比方法

图2:
横轴: Bandwidth Heterogeneity Level
纵轴: Formal Round Latency
曲线: 各对比方法

图3:
横轴: Bandwidth Heterogeneity Level
纵轴: Time-to-75%
曲线: 各对比方法
```

## 实验五：客户端掉线与移动性实验

目的：验证客户端掉线和移动性条件下的鲁棒性。

### 5.1 静态掉线率

```yaml
network.mobility.mode: "static"
topology.client_drop_rate: [0.0, 0.1, 0.3, 0.5]
```

### 5.2 移动性掉线

```yaml
network.mobility.mode: "mobility"
network.mobility.handoff_probability: [0.05, 0.10, 0.20]
network.mobility.handoff_drop_boost: 0.20
network.mobility.drop_jitter: 0.03
network.mobility.edge_drop_rates: [0.02, 0.05, 0.10, 0.15]
```

固定参数：

```yaml
datasets.partition_alpha: 0.3
datasets.labeled_ratio: 0.1
network.round_comm_budget_mb: 1200
network.bandwidth.mode: "edge_profile"
```

对比方法：

```text
Balanced Sampling
Resource-aware Sampling
Full EdgeQGFed
```

建议图：

```text
Drop Rate vs Accuracy
Drop Rate vs Time-to-Target
Drop Rate vs Active Clients
```

表格设置：

```text
行: dropout / mobility setting 与 method 的组合
列: Drop Setting, Method, Cloud Accuracy, Active Clients, Client Drop Ratio, Time-to-75%, Formal Round Latency
```

绘图设置：

```text
图1:
横轴: Client Drop Rate 或 Handoff Probability
纵轴: Cloud Accuracy
曲线: 各对比方法

图2:
横轴: Client Drop Rate 或 Handoff Probability
纵轴: Time-to-75%
曲线: 各对比方法

图3:
横轴: Client Drop Rate 或 Handoff Probability
纵轴: Active Clients
曲线: 各对比方法
```

## 实验六：标签比例敏感性实验

目的：验证少标签条件下伪标签机制和质量控制的作用。

扫描参数：

```text
labeled_ratio = 0.01, 0.05, 0.1, 0.2
```

固定参数：

```yaml
datasets.partition_alpha: 0.3
network.round_comm_budget_mb: 1200
network.bandwidth.mode: "edge_profile"
```

对比方法：

```text
w/o Pseudo Label
Full EdgeQGFed
```

额外记录：

```text
avg_pseudo_ratio
avg_pseudo_weight
avg_agreement_ratio
avg_cloud_confidence
avg_edge_confidence
```

建议表格：

```text
Label Ratio | Method | Accuracy | Pseudo Ratio | Pseudo Weight | Time-to-75%
```

表格设置：

```text
行: labeled_ratio 与 method 的组合
列: Label Ratio, Method, Cloud Accuracy, Avg Pseudo Ratio, Avg Pseudo Weight, Avg Agreement Ratio, Time-to-75%
```

绘图设置：

```text
图1:
横轴: Labeled Ratio
纵轴: Cloud Accuracy
曲线: w/o Pseudo Label, Full EdgeQGFed

图2:
横轴: Labeled Ratio
纵轴: Avg Pseudo Weight
曲线: Full EdgeQGFed

图3:
横轴: Labeled Ratio
纵轴: Avg Agreement Ratio
曲线: Full EdgeQGFed
```

## 实验七：图注意力聚合消融实验

目的：证明云端图注意力聚合机制中各质量因子的贡献。

固定参数：

```yaml
datasets.partition_alpha: 0.1
datasets.labeled_ratio: 0.1
network.round_comm_budget_mb: 1200
network.bandwidth.mode: "edge_profile"
```

消融配置：

```text
Full EdgeQGFed
w/o Graph: models.graph.use = False
w/o Prototype: models.graph.prototype_weight = 0.0
w/o Reliability: models.graph.reliability_weight = 0.0
w/o Label Ratio: models.graph.label_ratio_weight = 0.0
w/o Confidence: models.graph.confidence_weight = 0.0
```

建议表格：

```text
Variant | Accuracy | Edge Avg Acc | Graph Attention Diag | Time-to-75%
```

表格设置：

```text
行: Full EdgeQGFed 与各消融版本
列: Variant, Cloud Accuracy, Edge Avg Accuracy, Graph Attention Diag, Graph Reliability Mean, Time-to-75%, Total Comm MB
```

绘图设置：

```text
图1:
横轴: Ablation Variant
纵轴: Cloud Accuracy
柱: 各消融版本

图2:
横轴: Ablation Variant
纵轴: Time-to-75%
柱: 各消融版本

图3:
横轴: Ablation Variant
纵轴: Graph Attention Diag
柱: 各启用图聚合的版本
```

该实验建议使用 `partition_alpha = 0.1`，因为强 non-IID 下图聚合优势更容易体现。

## 实验八：规模扩展实验

目的：验证 EdgeQGFed 在不同客户端规模和边缘节点规模下的可扩展性。

客户端数量扫描：

```text
num_clients = 64, 128, 256
```

边缘数量扫描：

```text
num_edges = 2, 4, 8
```

建议组合：

```text
64 clients, 4 edges
128 clients, 4 edges
256 clients, 4 edges
128 clients, 2 edges
128 clients, 8 edges
```

固定参数：

```yaml
clients_per_round: 32
edges_per_round: min(4, num_edges)
datasets.partition_alpha: 0.3
datasets.labeled_ratio: 0.1
network.round_comm_budget_mb: 1200
```

指标：

```text
Accuracy
Formal Round Latency
Total Comm MB
Graph Aggregation Metrics
Wall-clock Time
```

表格设置：

```text
行: scale setting 与 method 的组合
列: Scale Setting, Method, Num Clients, Num Edges, Cloud Accuracy, Formal Round Latency, Total Comm MB, Wall-clock Time
```

绘图设置：

```text
图1:
横轴: Num Clients
纵轴: Cloud Accuracy
曲线: 各对比方法

图2:
横轴: Num Clients
纵轴: Formal Round Latency
曲线: 各对比方法

图3:
横轴: Num Edges
纵轴: Total Comm MB
曲线: 各对比方法
```

## 推荐论文图表顺序

```text
Table 1: Main comparison under alpha=0.3
Figure 1: Accuracy vs Communication MB
Figure 2: Accuracy vs Cumulative Estimated Latency
Table 2: Non-IID comparison under alpha=0.3 and alpha=0.1
Figure 3: Communication budget sensitivity
Figure 4: Heterogeneous bandwidth sensitivity
Figure 5: Dropout / mobility robustness
Figure 6: Label ratio sensitivity
Table 3: Graph aggregation ablation
Table 4: Scalability study
```

## 最小必跑实验

如果实验时间有限，优先运行：

```text
1. Main comparison, alpha=0.3
2. Main comparison, alpha=0.1
3. Communication budget = 300, 600, 1200
4. Drop rate = 0.0, 0.3, 0.5
5. Graph ablation under alpha=0.1
```

这组最小实验可以形成完整证据链：强 non-IID 下有效、通信预算下有效、掉线条件下稳健、图注意力机制确实有贡献。
