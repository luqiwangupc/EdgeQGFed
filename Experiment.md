# Experiment Plan

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
