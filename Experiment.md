# EdgeQGFed 实验执行与填表说明

本文档与 `Experiment_Record_Template.xlsx` 一一对应，用于说明每个实验应该跑什么、修改哪些配置参数、记录哪些指标，以及结果应填写到 xlsx 的哪个 sheet 和哪个区域。

当前论文主实验数据集为 `CIFAR-10`、`CIFAR-100`、`HARBox` 和 `NSL-KDD`。SVHN 仍可作为代码支持的数据集保留，但不作为当前 xlsx 主实验的一部分。若后续需要补充 SVHN，可作为额外图像泛化结果单独加行。

## 0. 通用设置

### 0.1 基础配置文件

| 数据集 | 配置文件 | 运行命令 |
|---|---|---|
| CIFAR-10 | `config/formated_config.yaml` | `python EdgeQGFed.py --config config/formated_config.yaml` |
| CIFAR-100 | `config/100_basic_config.yaml` | `python EdgeQGFed.py --config config/100_basic_config.yaml` |
| HARBox | `config/harbox_config.yaml` | `python EdgeQGFed.py --config config/harbox_config.yaml` |
| NSL-KDD | `config/nslkdd_config.yaml` | `python EdgeQGFed.py --config config/nslkdd_config.yaml` |

### 0.2 主要记录指标

| 论文含义 | 建议记录字段 |
|---|---|
| 云端准确率 | `cloud_accuracy` |
| 边缘平均准确率 | `edge_avg_accuracy` |
| 云端 loss | `cloud_loss` |
| 边缘平均 loss | `edge_avg_loss` |
| 达到目标精度时间 | `time_to_target_accuracy_s` 或 `time_to_target_s` |
| wall-clock 达到目标精度时间 | `wall_time_to_target_accuracy_s` |
| 累计通信量 | `cumulative_comm_mb` |
| 总通信量 | `total_comm_mb` |
| 每轮形式化时延 | `formal_round_latency_s` |
| 预算使用率 | `budget_used_ratio` |
| 选中客户端数 | `selected_clients` |
| 活跃客户端数 | `active_clients` |
| 客户端掉线比例 | `client_drop_ratio` |
| 图注意力对角占比 | `graph_attention_diag` |
| 图可靠性均值 | `graph_reliability_mean` |
| 图置信度均值 | `graph_confidence_mean` |
| NSL-KDD Macro-F1 | `cloud_macro_f1` |

建议主结果至少运行 3 个随机种子，填写 `mean±std`。如果训练成本较高，系统类实验可先填单次结果，并在论文中说明。

### 0.3 方法配置

| 方法 | 配置修改 |
|---|---|
| H-FedAvg | `models.graph.use=False`; `train.pseudo.use=False`; `topology.client_sample_mode=balanced` |
| H-FedAvg+PL | `models.graph.use=False`; `train.pseudo.use=True`; `topology.client_sample_mode=balanced` |
| EdgeQGFed | `models.graph.use=True`; `train.pseudo.use=True`; `topology.client_sample_mode=resource_aware` |
| w/o Graph | `models.graph.use=False`; 其余保持 EdgeQGFed |
| w/o Resource Sampling | `topology.client_sample_mode=balanced`; 其余保持 EdgeQGFed |
| w/o Pseudo Label | `train.pseudo.use=False`; 其余保持 EdgeQGFed |
| w/o Consistency | `train.initial_weight=0.0`; `train.final_weight=0.0`; 其余保持 EdgeQGFed |

实验一中的 `FedAvg`、`FedProx`、`FedAsync`、`FedBuff`、`FedAC`、`ASAFL`、`FedAMP`、`pFedGraph`、`FedSaC` 等现有方法用于论文横向大表。若当前代码未完整实现这些方法，可先通过复现脚本、已有结果或后续独立 baseline 代码补齐。

## 1. 实验一：跨模态总体性能

对应 xlsx sheet：`实验一 跨模态总体`

### 1.1 实验目的

该实验是论文主结果，对比 EdgeQGFed 与现有大量联邦学习方法，证明本文方法不仅在单一图像任务有效，也能覆盖图像、传感序列和网络流量任务。

### 1.2 数据集与参数

| 数据集 | 参数设置 |
|---|---|
| CIFAR-10 | `partition_mode=client_dirichlet`; `partition_alpha=0.3` 和 `0.1`; `labeled_ratio=0.1` |
| CIFAR-100 | `partition_mode=client_dirichlet`; `partition_alpha=0.3` 和 `0.1`; `labeled_ratio=0.1` |
| HARBox | `partition_mode=user`; `labeled_ratio=0.1`; `round_comm_budget_mb=300` |
| NSL-KDD | 五分类；`labeled_ratio=0.1`; `label_sampling=stratified`; `class_fn=weighted_focal` |

NSL-KDD 说明：原始二分类可接近 `99%` 准确率，任务过于饱和。因此论文采用 `Normal, DoS, Probe, R2L, U2R` 五分类，并记录 `Acc` 和 `Macro-F1`。

### 1.3 对比方法

填写大表时，方法行包括：

`FedAvg`, `FedProx`, `FedAsync`, `FedBuff`, `FedAC`, `ASAFL`, `FedAMP`, `pFedGraph`, `FedSaC`, `FedAMP-Async`, `pFedGraph-Async`, `EdgeQGFed`。

### 1.4 填写位置

#### 表1：跨模态边缘任务下与现有方法的总体性能对比

xlsx 位置：`实验一 跨模态总体` → `表1 跨模态边缘任务下与现有方法的总体性能对比`

| xlsx 列 | 填写内容 |
|---|---|
| `CIFAR-10 Dir(0.3)` | CIFAR-10 在 `partition_alpha=0.3` 下的最终/best 云端准确率 |
| `CIFAR-10 Dir(0.1)` | CIFAR-10 在 `partition_alpha=0.1` 下的最终/best 云端准确率 |
| `CIFAR-100 Dir(0.3)` | CIFAR-100 在 `partition_alpha=0.3` 下的最终/best 云端准确率 |
| `CIFAR-100 Dir(0.1)` | CIFAR-100 在 `partition_alpha=0.1` 下的最终/best 云端准确率 |
| `HARBox Real-world` | HARBox 用户级划分下的最终/best 云端准确率 |
| `NSL-KDD Acc` | NSL-KDD 五分类最终/best 云端准确率 |
| `NSL-KDD Macro-F1` | NSL-KDD 五分类 macro-F1 |

推荐填写格式：`83.21±0.42`。

#### 图1(a)：不同数据集的模型收敛情况

xlsx 位置：`实验一 跨模态总体` → `图1(a) 不同数据集的模型收敛情况`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| 训练步数 `0, 500, ..., 6000` | `Cloud Accuracy (%)` | 一个数据集，记录完整 EdgeQGFed |

填写 `CIFAR-10`、`CIFAR-100`、`HARBox`、`NSL-KDD` 四条曲线。

#### 图1(b)：主要方法最终准确率对比

xlsx 位置：`实验一 跨模态总体` → `图1(b) 主要方法最终准确率对比`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `CIFAR-10 Dir(0.3)`, `CIFAR-100 Dir(0.3)`, `HARBox`, `NSL-KDD` | `Final Cloud Accuracy (%)` | 一种方法 |

建议只放代表性方法：`FedAvg`, `FedProx`, `FedAMP`, `pFedGraph`, `EdgeQGFed`。

#### 图1(c)：Time-to-Target 对比

xlsx 位置：`实验一 跨模态总体` → `图1(c) Time-to-Target 对比`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| 数据集 | `Time-to-Target (s)` | `H-FedAvg`, `H-FedAvg+PL`, `EdgeQGFed` |

如果某方法没有达到目标精度，填 `N/A`。

## 2. 实验二：通信效率与时延

对应 xlsx sheet：`实验二 通信时延`

### 2.1 实验目的

证明 EdgeQGFed 在通信预算受限时能更有效地利用客户端上传，并通过三层时延模型反映端-边-云训练成本。

只在 `HARBox` 和 `CIFAR-10` 上做，避免所有数据集重复展开。

### 2.2 改变的参数

HARBox：

```yaml
network:
  round_comm_budget_mb: 100  # 也跑 200, 300, 500
```

CIFAR-10：

```yaml
network:
  round_comm_budget_mb: 300  # 也跑 600, 900, 1200
```

方法：

| 方法 | 配置 |
|---|---|
| H-FedAvg | `graph=False`, `pseudo=False`, `client_sample_mode=balanced` |
| EdgeQGFed w/o RS | `graph=True`, `pseudo=True`, `client_sample_mode=balanced` |
| EdgeQGFed | `graph=True`, `pseudo=True`, `client_sample_mode=resource_aware` |

### 2.3 填写位置

#### 表2：通信预算与时延敏感性

xlsx 位置：`实验二 通信时延` → `表2 通信预算与时延敏感性`

| xlsx 列 | 填写内容 |
|---|---|
| `Cloud Acc (%)` | 当前预算下的 best 云端准确率 |
| `Time-to-Target (s)` | 达到目标精度所需估计时间 |
| `Budget Used Ratio` | 当前轮或累计预算使用率 |
| `Formal Round Latency (s)` | 形式化每轮时延 |
| `Total Comm (MB)` | 总通信量 |

#### 图2(a)：累计通信量下的准确率收敛

xlsx 位置：`实验二 通信时延` → `图2(a) 累计通信量下的准确率收敛`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| 累计通信量 MB | `Cloud Accuracy (%)` | `HARBox-H-FedAvg`, `HARBox-EdgeQGFed`, `CIFAR-10-H-FedAvg`, `CIFAR-10-EdgeQGFed` |

#### 图2(b)：三层时延分解

xlsx 位置：`实验二 通信时延` → `图2(b) 三层时延分解`

| 横轴 | 纵轴 | 每列分量 |
|---|---|---|
| 数据集-方法 | `Latency Components (s)` | 端到边上传、边缘计算、边到云上传、云端聚合、云到边下发 |

#### 图2(c)：通信预算与 Time-to-Target

xlsx 位置：`实验二 通信时延` → `图2(c) 通信预算与 Time-to-Target`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `round_comm_budget_mb` | `Time-to-Target (s)` | 数据集-方法 |

## 3. 实验三：系统鲁棒性

对应 xlsx sheet：`实验三 系统鲁棒`

### 3.1 实验目的

验证客户端掉线、带宽异构和移动性条件下，资源感知采样和图注意力聚合是否提升稳定性。

主实验只记录 `HARBox`，因为其用户级划分更接近真实边缘设备参与。若需要图像补充，可额外跑 CIFAR-10，但不作为主表必须项。

### 3.2 改变的参数

客户端掉线率：

```yaml
topology:
  client_drop_rate: 0.0  # 也跑 0.1, 0.2, 0.3
network:
  mobility:
    mode: "static"
```

带宽异构强度：

```yaml
network:
  bandwidth:
    mode: "edge_profile"
```

HARBox 建议三档：

| 强度 | `client_uplink_by_edge_mbps` | `edge_uplink_by_edge_mbps` | `cloud_downlink_by_edge_mbps` |
|---|---|---|---|
| Mild | `[15, 20, 25, 30, 35]` | `[150, 180, 220, 260, 300]` | `[300, 350, 400, 450, 500]` |
| Medium | `[5, 10, 20, 30, 50]` | `[80, 120, 200, 300, 400]` | `[150, 250, 400, 500, 600]` |
| Severe | `[2, 5, 10, 25, 50]` | `[40, 80, 150, 250, 400]` | `[100, 180, 300, 450, 600]` |

### 3.3 填写位置

#### 表3：链路异构与客户端掉线鲁棒性

xlsx 位置：`实验三 系统鲁棒` → `表3 链路异构与客户端掉线鲁棒性`

| xlsx 列 | 填写内容 |
|---|---|
| `Cloud Acc (%)` | 掉线率下 best 云端准确率 |
| `Active Clients` | 实际活跃客户端数 |
| `Client Drop Ratio` | 实际掉线比例 |
| `Time-to-Target (s)` | 达到目标精度时间 |

#### 图3(a)：客户端掉线率与云端准确率

xlsx 位置：`实验三 系统鲁棒` → `图3(a) 客户端掉线率与云端准确率`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `client_drop_rate = 0.0, 0.1, 0.2, 0.3` | `Cloud Accuracy (%)` | `H-FedAvg`, `EdgeQGFed w/o RS`, `EdgeQGFed` |

#### 图3(b)：带宽异构强度与 Time-to-Target

xlsx 位置：`实验三 系统鲁棒` → `图3(b) 带宽异构强度与 Time-to-Target`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `Mild`, `Medium`, `Severe` | `Time-to-Target (s)` | `H-FedAvg`, `EdgeQGFed w/o RS`, `EdgeQGFed` |

## 4. 实验四：消融实验

对应 xlsx sheet：`实验四 消融`

### 4.1 实验目的

分析 EdgeQGFed 各模块贡献：云端图注意力、资源感知采样、伪标签学习和云边一致性。

数据集只跑：

| 数据集 | 作用 |
|---|---|
| HARBox | 传感序列边缘任务 |
| CIFAR-10 | 图像边缘任务 |

### 4.2 改变的参数

| 变体 | 修改 |
|---|---|
| Full EdgeQGFed | 默认完整配置 |
| w/o Graph | `models.graph.use=False` |
| w/o Resource Sampling | `topology.client_sample_mode=balanced` |
| w/o Pseudo Label | `train.pseudo.use=False` |
| w/o Consistency | `train.initial_weight=0.0`; `train.final_weight=0.0` |

### 4.3 填写位置

#### 表4：消融实验

xlsx 位置：`实验四 消融` → `表4 消融实验`

| xlsx 列 | 填写内容 |
|---|---|
| `Cloud Acc (%)` | best 云端准确率 |
| `Time-to-Target (s)` | 达到目标精度时间 |
| `Total Comm (MB)` | 总通信量 |

#### 图4(a)：消融变体与云端准确率

xlsx 位置：`实验四 消融` → `图4(a) 消融变体与云端准确率`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| 消融变体 | `Cloud Accuracy (%)` | `HARBox`, `CIFAR-10` |

#### 图4(b)：消融变体与 Time-to-Target

xlsx 位置：`实验四 消融` → `图4(b) 消融变体与 Time-to-Target`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| 消融变体 | `Time-to-Target (s)` | `HARBox`, `CIFAR-10` |

## 5. 实验五：质量敏感性

对应 xlsx sheet：`实验五 质量敏感`

### 5.1 实验目的

分析 EdgeQGFed 对数据异构程度、标签比例和图质量信号的敏感性。

该实验不对所有数据集重复展开：

| 数据集 | 分析内容 |
|---|---|
| CIFAR-10 | Dirichlet non-IID 敏感性 |
| CIFAR-100 | Dirichlet non-IID 敏感性 |
| HARBox | 标签比例敏感性 |
| NSL-KDD | 只作为图质量信号中的一个跨模态点，不单独分析 |

### 5.2 改变的参数

图像任务 non-IID：

```yaml
datasets:
  partition_mode: "client_dirichlet"
  partition_alpha: 0.1  # 也跑 0.3, 0.5, 1.0
```

HARBox 标签比例：

```yaml
datasets:
  labeled_ratio: 0.05  # 也跑 0.1, 0.2, 1.0
```

当 `labeled_ratio=1.0` 作为全监督对照时，建议：

```yaml
train:
  pseudo:
    use: False
```

### 5.3 填写位置

#### 表5：跨模态质量敏感性

xlsx 位置：`实验五 质量敏感` → `表5 跨模态质量敏感性`

| xlsx 列 | 填写内容 |
|---|---|
| `Cloud Acc (%)` | 对应参数下 best 云端准确率 |
| `Edge Avg Acc (%)` | 对应参数下 best 边缘平均准确率 |
| `Time-to-Target (s)` | 达到目标精度时间 |

#### 图5(a)：Dirichlet alpha 与图像任务准确率

xlsx 位置：`实验五 质量敏感` → `图5(a) Dirichlet alpha 与图像任务准确率`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `partition_alpha = 0.1, 0.3, 0.5, 1.0` | `Cloud Accuracy (%)` | `CIFAR-10-H-FedAvg`, `CIFAR-10-EdgeQGFed`, `CIFAR-100-H-FedAvg`, `CIFAR-100-EdgeQGFed` |

#### 图5(b)：HARBox 标签比例与准确率

xlsx 位置：`实验五 质量敏感` → `图5(b) HARBox 标签比例与准确率`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `labeled_ratio = 0.05, 0.1, 0.2, 1.0` | `Cloud Accuracy (%)` | `H-FedAvg`, `EdgeQGFed` |

#### 图5(c)：图注意力质量信号

xlsx 位置：`实验五 质量敏感` → `图5(c) 图注意力质量信号`

| 横轴 | 纵轴 | 每行 |
|---|---|---|
| `CIFAR-10`, `CIFAR-100`, `HARBox`, `NSL-KDD` | Graph Metric | `graph_attention_diag`, `graph_reliability_mean`, `graph_confidence_mean` |

## 6. 建议实验优先级

如果时间有限，建议按以下顺序跑：

1. 实验一中 EdgeQGFed 在四个数据集上的主结果。
2. 实验一中关键 baseline：`FedAvg`, `FedProx`, `FedAMP`, `pFedGraph`。
3. 实验二通信效率：HARBox 和 CIFAR-10。
4. 实验四消融：HARBox 和 CIFAR-10。
5. 实验三系统鲁棒性：HARBox。
6. 实验五敏感性：CIFAR-10/CIFAR-100 的 `alpha=0.1,0.3`，HARBox 的 `labeled_ratio=0.05,0.1,0.2,1.0`。

## 7. 写论文时的对应关系

| 论文位置 | 对应 xlsx |
|---|---|
| 数据集与实验设置 | `总览与配置` |
| 主结果大表 | `实验一 跨模态总体` → 表1 |
| 收敛曲线 | `实验一 跨模态总体` → 图1(a) |
| 方法柱状对比 | `实验一 跨模态总体` → 图1(b) |
| Time-to-Target | `实验一 跨模态总体` → 图1(c)，以及 `实验二 通信时延` |
| 通信效率 | `实验二 通信时延` |
| 三层时延模型 | `实验二 通信时延` → 图2(b) |
| 掉线和带宽异构 | `实验三 系统鲁棒` |
| 模块贡献 | `实验四 消融` |
| non-IID 与标签比例敏感性 | `实验五 质量敏感` |
