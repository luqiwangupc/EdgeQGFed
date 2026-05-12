# EdgeQGFed 实验设计与结果填表模板

本文实验按照 INFOCOM 风格组织，重点不是单纯追求集中式分类最高精度，而是证明 EdgeQGFed 在通信受限、链路异构、客户端不稳定和少标签条件下，能够实现更高效、更稳健的云-边-端协同学习。

建议所有结果至少运行 3 个随机种子，并报告 `mean ± std`。若训练成本较高，主结果表保留 3 次运行，敏感性实验可先报告单次结果，并在论文中说明。

## 1. Experimental Setup

### 1.1 Datasets and Tasks

| Dataset | Modality | Classes | Partition | Role in Experiments |
|---|---:|---:|---|---|
| HARBox | Sensor time series | 5 | User-level split | Main edge intelligence task |
| SVHN | Image | 10 | Dirichlet non-IID | Lightweight visual task |
| CIFAR-10 | Image | 10 | Dirichlet non-IID | Standard image benchmark |
| CIFAR-100 | Image | 100 | Dirichlet non-IID | Harder large-class benchmark |

说明：HARBox 使用用户级划分，体现真实边缘用户差异，不使用 `edge_dirichlet_alpha` 表示边缘异构。SVHN、CIFAR-10 和 CIFAR-100 使用 Dirichlet 划分，主要报告 `partition_alpha=0.3` 和 `partition_alpha=0.1`。

### 1.2 Default Training Settings

| Parameter | HARBox | SVHN | CIFAR-10 | CIFAR-100 |
|---|---:|---:|---:|---:|
| `labeled_ratio` | 0.1 | 0.1 | 0.1 | 0.1 |
| `num_edges` | 5 | 4 | 4 | 4 |
| `num_clients` | 100 | 128 | 128 | 128 |
| `clients_per_round` | 50 | 32 | 32 | 32 |
| `batch_size` | 16 | 64 | 64 | 64 |
| `total_steps` | 6000 | 3000 | 5000 | 6000 |
| `learning_rate` | 8e-4 | 1e-3 | 1e-3 | 1e-3 |
| `encoder` | identity | identity | ResNet-50 | ResNet-50 |
| `classifier` | har-cnn | small-cnn | small-mlp | small-mlp |
| `round_comm_budget_mb` | 300 | 1200 | 1200 | 1200 |

### 1.3 Compared Methods

| Method | Graph Aggregation | Pseudo Labeling | Client Sampling | Description |
|---|---|---|---|---|
| H-FedAvg | No | No | Balanced | Basic cloud-edge hierarchical FedAvg |
| H-FedAvg + PL | No | Yes | Balanced | FedAvg with pseudo-label learning |
| EdgeQGFed w/o Graph | No | Yes | Resource-aware | Remove cloud graph attention |
| EdgeQGFed w/o RS | Yes | Yes | Balanced | Remove resource-aware sampling |
| EdgeQGFed | Yes | Yes | Resource-aware | Full method |

### 1.4 Main Metrics

| Category | Metrics |
|---|---|
| Accuracy | `cloud_accuracy`, `edge_avg_accuracy` |
| Convergence | `cloud_loss`, `edge_avg_loss`, `edge_class_loss` |
| Communication | `total_comm_mb`, `cumulative_comm_mb`, `budget_used_ratio` |
| Latency | `formal_round_latency_s`, `cumulative_estimated_latency_s`, `time_to_target_accuracy_s` |
| Robustness | `client_drop_ratio`, `active_clients`, `samples_per_estimated_second` |
| Semi-supervised quality | `avg_pseudo_ratio`, `avg_pseudo_weight`, `avg_agreement_ratio`, `avg_consistency_ratio` |
| Graph aggregation | `graph_attention_diag`, `graph_reliability_mean`, `graph_confidence_mean` |

### 1.5 Target Accuracy for Time-to-Accuracy

| Dataset | Target Accuracy |
|---|---:|
| HARBox | 83.79 |
| SVHN | 90.32 |
| CIFAR-10 | ____ |
| CIFAR-100 | ____ |

建议初始值：HARBox 80%，SVHN 75%，CIFAR-10 70%，CIFAR-100 40%。若某方法未达到目标精度，填 `N/A`。

## 2. Experiment 1: Overall Performance

目的：展示 EdgeQGFed 在四个数据集上的总体性能，并与分层 FedAvg 及关键消融方法比较。这是论文主结果，建议放在实验部分最前面。

### Parameter Setting

| Parameter | Value |
|---|---|
| `labeled_ratio` | 0.1 |
| SVHN/CIFAR `partition_alpha` | 0.3 |
| HARBox partition | user-level |
| `bandwidth.mode` | edge_profile |
| `client_drop_rate` | 0.0 |
| `round_comm_budget_mb` | HARBox=300, others=1200 |

### Table 1: Overall Performance

| Dataset | Method | Cloud Acc (%) | Edge Avg Acc (%) | Cloud Loss | Edge Avg Loss | Time-to-Target (s) | Total Comm (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| HARBox | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| HARBox | H-FedAvg + PL | ____ | ____ | ____ | ____ | ____ | ____ |
| HARBox | EdgeQGFed w/o Graph | ____ | ____ | ____ | ____ | ____ | ____ |
| HARBox | EdgeQGFed w/o RS | ____ | ____ | ____ | ____ | ____ | ____ |
| HARBox | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | H-FedAvg + PL | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | EdgeQGFed w/o Graph | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | EdgeQGFed w/o RS | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | H-FedAvg + PL | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | EdgeQGFed w/o Graph | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | EdgeQGFed w/o RS | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | H-FedAvg + PL | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | EdgeQGFed w/o Graph | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | EdgeQGFed w/o RS | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 1(a) | Line chart | Communication Round | Cloud Accuracy (%) | Five methods |
| Fig. 1(b) | Line chart | Communication Round | Edge Avg Accuracy (%) | Five methods |
| Fig. 1(c) | Grouped bar chart | Dataset | Final Cloud Accuracy (%) | Five methods |
| Fig. 1(d) | Grouped bar chart | Dataset | Time-to-Target (s) | Five methods |

### Expected Claim

EdgeQGFed should achieve the best or near-best accuracy while reducing time-to-target and communication cost. If absolute accuracy is close to other methods, emphasize its efficiency and robustness under network constraints.

## 3. Experiment 2: Communication Efficiency

目的：验证 EdgeQGFed 在相同通信量或相同通信预算下是否能更快达到目标精度。这一实验最贴近 INFOCOM 的网络系统关注点。

### Parameter Sweep

| Dataset | `round_comm_budget_mb` |
|---|---|
| HARBox | 100, 200, 300, 500 |
| SVHN | 300, 600, 900, 1200 |
| CIFAR-10 | 300, 600, 900, 1200 |
| CIFAR-100 | 600, 900, 1200, 1800 |

### Table 2: Communication Budget Sensitivity

| Dataset | Budget (MB) | Method | Cloud Acc (%) | Edge Avg Acc (%) | Selected Clients | Budget Used Ratio | Time-to-Target (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| HARBox | 100 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| HARBox | 100 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 200 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| HARBox | 200 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 300 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| HARBox | 300 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| SVHN | 300 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| SVHN | 300 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 300 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 300 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | 600 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | 600 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |

完整表格按所有预算展开。论文主文可只放 HARBox 和 CIFAR-10，其他数据集放附录或补充表。

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 2(a) | Line chart | Cumulative Communication (MB) | Cloud Accuracy (%) | Five methods |
| Fig. 2(b) | Line chart | Round Communication Budget (MB) | Final Cloud Accuracy (%) | H-FedAvg, w/o RS, EdgeQGFed |
| Fig. 2(c) | Line chart | Round Communication Budget (MB) | Time-to-Target (s) | H-FedAvg, w/o RS, EdgeQGFed |
| Fig. 2(d) | Grouped bar chart | Dataset | Communication Saved at Target (%) | EdgeQGFed vs best baseline |

### Expected Claim

EdgeQGFed should require less cumulative communication to reach the same accuracy. Resource-aware sampling should be especially useful under small communication budgets.

## 4. Experiment 3: Latency Breakdown and Time-to-Accuracy

目的：展示端-边-云三层时延模型的作用，证明方法不只提升精度，也缩短实际边缘网络中的收敛时间。

### Parameter Setting

使用 Experiment 1 的默认设置，记录每轮通信与计算时延分量。

### Table 3: Latency Components

| Dataset | Method | Terminal-to-Edge Upload (s) | Edge Compute (s) | Edge-to-Cloud Upload (s) | Cloud Aggregation (s) | Cloud-to-Edge Downlink (s) | Formal Round Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| HARBox | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| HARBox | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| SVHN | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | H-FedAvg | ____ | ____ | ____ | ____ | ____ | ____ |
| CIFAR-100 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 3(a) | Stacked bar chart | Dataset | Latency Components (s) | Terminal-edge, edge-cloud, cloud-edge, compute |
| Fig. 3(b) | Line chart | Cumulative Estimated Latency (s) | Cloud Accuracy (%) | Five methods |
| Fig. 3(c) | Grouped bar chart | Dataset | Time-to-Target (s) | Five methods |

### Expected Claim

Compared with baselines, EdgeQGFed should improve time-to-accuracy by selecting higher-value clients and reducing ineffective communication rounds.

## 5. Experiment 4: Bandwidth Heterogeneity

目的：验证链路异构增强时，资源感知采样和质量感知聚合是否能保持训练效率。

### 5.1 Deterministic Edge Profile

For SVHN, CIFAR-10, CIFAR-100:

| Level | Client Uplink (Mbps) | Edge Uplink (Mbps) | Cloud Downlink (Mbps) |
|---|---|---|---|
| Mild | [60, 80, 100, 120] | [300, 400, 500, 600] | [600, 700, 800, 900] |
| Medium | [40, 60, 80, 120] | [200, 300, 400, 600] | [400, 600, 800, 1000] |
| Severe | [10, 30, 60, 120] | [80, 150, 300, 600] | [200, 400, 700, 1000] |

For HARBox:

| Level | Client Uplink (Mbps) | Edge Uplink (Mbps) | Cloud Downlink (Mbps) |
|---|---|---|---|
| Mild | [20, 25, 30, 40, 50] | [150, 180, 220, 300, 400] | [250, 300, 400, 500, 600] |
| Medium | [5, 10, 20, 30, 50] | [80, 120, 200, 300, 400] | [150, 250, 400, 500, 600] |
| Severe | [2, 5, 10, 20, 50] | [40, 80, 120, 200, 400] | [80, 150, 250, 400, 600] |

### 5.2 Stochastic Bandwidth Distribution

| Setting | Client Distribution | Edge Distribution | Cloud Downlink Distribution |
|---|---|---|---|
| Uniform | min=10, max=120 | min=100, max=1000 | min=300, max=1200 |
| LogNormal-Mild | median=60, sigma=0.35 | median=400, sigma=0.35 | min=300, max=1200 |
| LogNormal-Severe | median=40, sigma=0.75 | median=300, sigma=0.75 | min=100, max=1000 |

### Table 4: Bandwidth Heterogeneity

| Dataset | Bandwidth Setting | Method | Cloud Acc (%) | Formal Latency (s) | Time-to-Target (s) | Samples / Estimated Second |
|---|---|---|---:|---:|---:|---:|
| HARBox | Mild | H-FedAvg | ____ | ____ | ____ | ____ |
| HARBox | Mild | EdgeQGFed | ____ | ____ | ____ | ____ |
| HARBox | Medium | H-FedAvg | ____ | ____ | ____ | ____ |
| HARBox | Medium | EdgeQGFed | ____ | ____ | ____ | ____ |
| HARBox | Severe | H-FedAvg | ____ | ____ | ____ | ____ |
| HARBox | Severe | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-10 | Mild | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | Mild | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-10 | Severe | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | Severe | EdgeQGFed | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 4(a) | Line chart | Bandwidth Heterogeneity Level | Cloud Accuracy (%) | H-FedAvg, w/o RS, EdgeQGFed |
| Fig. 4(b) | Line chart | Bandwidth Heterogeneity Level | Time-to-Target (s) | H-FedAvg, w/o RS, EdgeQGFed |
| Fig. 4(c) | Box plot | Bandwidth Distribution | Formal Round Latency (s) | Methods |
| Fig. 4(d) | Line chart | Communication Round | Samples per Estimated Second | Methods |

### Expected Claim

The performance gap should become larger under severe bandwidth heterogeneity, showing that EdgeQGFed is more network-aware than standard hierarchical FedAvg.

## 6. Experiment 5: Client Dropout and Mobility

目的：验证客户端掉线和移动性条件下的鲁棒性。这部分非常适合 INFOCOM，因为它直接体现真实无线边缘网络中的不稳定参与。

### Static Dropout

| Parameter | Values |
|---|---|
| `mobility.mode` | static |
| `client_drop_rate` | 0.0, 0.1, 0.3, 0.5 |

### Mobility-Induced Dropout

| Parameter | Values |
|---|---|
| `mobility.mode` | mobility |
| `handoff_probability` | 0.05, 0.10, 0.20 |
| `handoff_drop_boost` | 0.20 |
| `drop_jitter` | 0.03 |

### Table 5: Dropout and Mobility Robustness

| Dataset | Drop Setting | Method | Cloud Acc (%) | Active Clients | Client Drop Ratio | Time-to-Target (s) | Accuracy Drop (%) |
|---|---|---|---:|---:|---:|---:|---:|
| HARBox | 0.0 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.0 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.3 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.3 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.5 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.5 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.0 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.0 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.3 | H-FedAvg | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.3 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 5(a) | Line chart | Client Drop Rate | Cloud Accuracy (%) | Methods |
| Fig. 5(b) | Line chart | Client Drop Rate | Time-to-Target (s) | Methods |
| Fig. 5(c) | Line chart | Handoff Probability | Cloud Accuracy (%) | Methods |
| Fig. 5(d) | Grouped bar chart | Drop Setting | Active Clients | Methods |

### Expected Claim

EdgeQGFed should show smaller accuracy degradation and more stable convergence as dropout or mobility increases.

## 7. Experiment 6: Statistical Heterogeneity

目的：验证 non-IID 程度增强时，云端图注意力聚合能否缓解低质量边缘更新的负面影响。

### Parameter Sweep

| Dataset | Heterogeneity Parameter |
|---|---|
| SVHN | `partition_alpha=0.3, 0.1` |
| CIFAR-10 | `partition_alpha=0.3, 0.1` |
| CIFAR-100 | `partition_alpha=0.3, 0.1` |
| HARBox | `HARBOX_TRAIN_USERS=120, 90` |

### Table 6: Non-IID Robustness

| Dataset | Heterogeneity | Method | Cloud Acc (%) | Edge Avg Acc (%) | Accuracy Drop (%) | Graph Attention Diag |
|---|---|---|---:|---:|---:|---:|
| SVHN | alpha=0.3 | H-FedAvg | ____ | ____ | ____ | ____ |
| SVHN | alpha=0.3 | EdgeQGFed | ____ | ____ | ____ | ____ |
| SVHN | alpha=0.1 | H-FedAvg | ____ | ____ | ____ | ____ |
| SVHN | alpha=0.1 | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-10 | alpha=0.3 | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | alpha=0.3 | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-10 | alpha=0.1 | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | alpha=0.1 | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-100 | alpha=0.3 | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-100 | alpha=0.1 | EdgeQGFed | ____ | ____ | ____ | ____ |
| HARBox | users=120 | H-FedAvg | ____ | ____ | ____ | ____ |
| HARBox | users=90 | EdgeQGFed | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 6(a) | Grouped bar chart | Dataset and Heterogeneity | Cloud Accuracy (%) | H-FedAvg, w/o Graph, EdgeQGFed |
| Fig. 6(b) | Line chart | Dirichlet Alpha | Accuracy Drop (%) | Methods |
| Fig. 6(c) | Heatmap | Edge Node | Edge Node | Graph Attention Weights |
| Fig. 6(d) | Line chart | Communication Round | Graph Attention Diag | EdgeQGFed |

### Expected Claim

When data heterogeneity becomes stronger, EdgeQGFed should suffer a smaller accuracy drop. The attention heatmap can visually support that the cloud learns non-uniform edge relationships.

## 8. Experiment 7: Label Efficiency

目的：验证 EdgeQGFed 能否在少标签条件下有效利用未标注边缘数据。

### Parameter Sweep

| Parameter | Values |
|---|---|
| `labeled_ratio` | 0.05, 0.1, 0.2, 1.0 |

HARBox 可额外跑 `0.01`，但如果波动较大，主文只放 `0.05, 0.1, 0.2, 1.0`。

### Table 7: Label Ratio Sensitivity

| Dataset | Label Ratio | Method | Cloud Acc (%) | Edge Avg Acc (%) | Avg Pseudo Ratio | Avg Pseudo Weight | Avg Agreement Ratio |
|---|---:|---|---:|---:|---:|---:|---:|
| HARBox | 0.05 | Supervised Only | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.05 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.1 | Supervised Only | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.1 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.2 | Supervised Only | ____ | ____ | ____ | ____ | ____ |
| HARBox | 0.2 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | 1.0 | Fully Supervised | ____ | ____ | N/A | N/A | N/A |
| CIFAR-10 | 0.05 | Supervised Only | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.05 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.1 | Supervised Only | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | 0.1 | EdgeQGFed | ____ | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 7(a) | Line chart | Label Ratio | Cloud Accuracy (%) | Supervised Only, PL Only, EdgeQGFed |
| Fig. 7(b) | Line chart | Label Ratio | Edge Avg Accuracy (%) | Supervised Only, PL Only, EdgeQGFed |
| Fig. 7(c) | Line chart | Label Ratio | Avg Pseudo Weight | EdgeQGFed |
| Fig. 7(d) | Line chart | Communication Round | Weighted Pseudo Loss | EdgeQGFed |

### Expected Claim

At `labeled_ratio=0.1`, EdgeQGFed should approach its full-supervised upper bound while using far fewer labels. For HARBox, this result supports the claim that the method is useful when edge labels are scarce.

## 9. Experiment 8: Ablation Study

目的：验证 EdgeQGFed 中各关键模块的贡献，包括图注意力、资源感知采样、伪标签学习和质量信号。

### Ablation Variants

| Variant | Configuration |
|---|---|
| Full EdgeQGFed | All modules enabled |
| w/o Graph | `models.graph.use=False` |
| w/o Prototype | `prototype_weight=0.0` |
| w/o Reliability | `reliability_weight=0.0` |
| w/o Label Ratio | `label_ratio_weight=0.0` |
| w/o Confidence | `confidence_weight=0.0` |
| w/o Resource Sampling | `client_sample_mode=balanced` |
| w/o Pseudo Label | `train.pseudo.use=False` |

### Table 8: Ablation Results

| Dataset | Variant | Cloud Acc (%) | Edge Avg Acc (%) | Time-to-Target (s) | Total Comm (MB) | Graph Attention Diag |
|---|---|---:|---:|---:|---:|---:|
| HARBox | Full EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| HARBox | w/o Graph | ____ | ____ | ____ | ____ | ____ |
| HARBox | w/o Reliability | ____ | ____ | ____ | ____ | ____ |
| HARBox | w/o Resource Sampling | ____ | ____ | ____ | ____ | ____ |
| HARBox | w/o Pseudo Label | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | Full EdgeQGFed | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | w/o Graph | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | w/o Reliability | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | w/o Resource Sampling | ____ | ____ | ____ | ____ | ____ |
| CIFAR-10 | w/o Pseudo Label | ____ | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 8(a) | Grouped bar chart | Ablation Variant | Cloud Accuracy (%) | HARBox, CIFAR-10 |
| Fig. 8(b) | Grouped bar chart | Ablation Variant | Time-to-Target (s) | HARBox, CIFAR-10 |
| Fig. 8(c) | Line chart | Communication Round | Graph Reliability Mean | Graph variants |
| Fig. 8(d) | Heatmap | Edge Node | Edge Node | Learned attention matrix |

### Expected Claim

Removing graph aggregation should hurt accuracy under non-IID settings. Removing resource-aware sampling should mainly hurt communication efficiency and time-to-target. Removing pseudo labeling should hurt low-label performance.

## 10. Experiment 9: Scalability

目的：验证客户端数量和边缘节点数量变化时，EdgeQGFed 的训练效率和聚合开销是否可控。

### Scale Settings

| Setting | `num_clients` | `num_edges` | `clients_per_round` |
|---|---:|---:|---:|
| Small | 64 | 4 | 16 |
| Default | 128 | 4 | 32 |
| Large | 256 | 4 | 64 |
| Few Edges | 128 | 2 | 32 |
| Many Edges | 128 | 8 | 32 |

HARBox 受用户数限制时使用：

| Setting | `num_clients` | `num_edges` | `clients_per_round` |
|---|---:|---:|---:|
| HARBox-Small | 50 | 5 | 25 |
| HARBox-Default | 100 | 5 | 50 |

### Table 9: Scalability Results

| Dataset | Scale Setting | Method | Cloud Acc (%) | Formal Latency (s) | Total Comm (MB) | Wall-clock Time (s) |
|---|---|---|---:|---:|---:|---:|
| CIFAR-10 | Small | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | Small | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-10 | Default | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | Default | EdgeQGFed | ____ | ____ | ____ | ____ |
| CIFAR-10 | Large | H-FedAvg | ____ | ____ | ____ | ____ |
| CIFAR-10 | Large | EdgeQGFed | ____ | ____ | ____ | ____ |
| HARBox | HARBox-Small | H-FedAvg | ____ | ____ | ____ | ____ |
| HARBox | HARBox-Small | EdgeQGFed | ____ | ____ | ____ | ____ |
| HARBox | HARBox-Default | H-FedAvg | ____ | ____ | ____ | ____ |
| HARBox | HARBox-Default | EdgeQGFed | ____ | ____ | ____ | ____ |

### Figures

| Figure | Type | X-axis | Y-axis | Curves / Groups |
|---|---|---|---|---|
| Fig. 9(a) | Line chart | Num Clients | Cloud Accuracy (%) | H-FedAvg, EdgeQGFed |
| Fig. 9(b) | Line chart | Num Clients | Formal Round Latency (s) | H-FedAvg, EdgeQGFed |
| Fig. 9(c) | Line chart | Num Edges | Total Communication (MB) | H-FedAvg, EdgeQGFed |
| Fig. 9(d) | Bar chart | Num Edges | Graph Aggregation Overhead (s) | EdgeQGFed |

### Expected Claim

EdgeQGFed should maintain stable accuracy and acceptable overhead as clients or edge nodes increase. If graph aggregation overhead grows, emphasize that the cost is at the cloud and remains small relative to communication latency.

## 11. Minimum Required Experiments

如果时间有限，优先完成下面实验。这个组合已经能形成完整 INFOCOM 证据链。

| Priority | Experiment | Dataset | Required Figures / Tables |
|---:|---|---|---|
| 1 | Overall performance | HARBox, SVHN, CIFAR-10, CIFAR-100 | Table 1, Fig. 1 |
| 2 | Communication efficiency | HARBox, CIFAR-10 | Table 2, Fig. 2 |
| 3 | Latency breakdown | HARBox, CIFAR-10 | Table 3, Fig. 3 |
| 4 | Dropout and mobility | HARBox, CIFAR-10 | Table 5, Fig. 5 |
| 5 | Statistical heterogeneity | SVHN, CIFAR-10, CIFAR-100 | Table 6, Fig. 6 |
| 6 | Ablation study | HARBox, CIFAR-10 | Table 8, Fig. 8 |

## 12. Suggested Paper Figure Order

| Order | Figure/Table | Content |
|---:|---|---|
| Table 1 | Overall performance | Four datasets and five methods |
| Fig. 1 | Accuracy curves | Accuracy vs communication round |
| Fig. 2 | Communication efficiency | Accuracy vs cumulative communication |
| Fig. 3 | Time-to-accuracy | Accuracy vs estimated latency and latency components |
| Fig. 4 | Bandwidth heterogeneity | Performance under mild, medium, severe bandwidth |
| Fig. 5 | Dropout and mobility | Robustness under unstable clients |
| Table 6 | Non-IID robustness | alpha=0.3 and alpha=0.1 |
| Fig. 7 | Label efficiency | Accuracy under different label ratios |
| Table 8 | Ablation | Contributions of graph, sampling, pseudo labels |
| Fig. 8 | Graph attention visualization | Attention heatmap and graph metrics |
| Table 9 | Scalability | Clients and edge nodes |

## 13. Result Writing Notes

HARBox 可作为主数据集，因为它更贴近真实边缘智能中的用户级数据异构和轻量终端参与。不要把 HARBox 结果写成超过已有集中式 HAR 方法，而应强调：

> Under user-level partition, limited labels, communication budgets, and unstable participation, EdgeQGFed achieves strong accuracy while reducing communication cost and time-to-accuracy.

SVHN、CIFAR-10 和 CIFAR-100 用于说明方法可迁移到不同边缘任务。论文中可以说 EdgeQGFed supports cross-modal edge intelligence tasks，但不要说它是多模态融合模型。
