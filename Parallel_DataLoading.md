# 多客户端并行取数：在原有服务器 YAML 上启用

更新：2026-10-22。这里只优化数据供给，不增加E1第2/3/4组。

## 改了什么

旧版每个ES一次提交一个大batch，整个batch由一个DataLoader任务处理；训练worker=0时主进程逐样本增强，GPU等待CPU。单纯增加旧loader的worker数，也不能把这一个大batch拆给多个worker。

新版将每个成功client的batch作为一个任务，多个spawn worker并行完成原图/weak/strong准备。主进程按既定ES/client顺序拼接张量，再执行原有encoder、loss、optimizer和graph/EMA。GPU不放进worker，不并行更新ES模型；模拟中的并行上传和通信成本也不会自动变成真实多设备实测。

| 保持的实验含义 | 新的数据准备方式 |
|---|---|
| 每成功client原有batch大小、抽样分布和标签mask | 主进程用现有MutableEdgeBatchSampler抽索引，worker不重新分区/抽样 |
| 原有三视图和增强算子 | worker调用同一ReusableEdgeBatchDataset，不缩图、不删view、不换encoder |
| ES更新顺序、teacher固定时点、loss、prototype、聚合、EMA | 只提前准备CPU数据，不提前更新模型 |
| 同一新协议的配对输入 | availability/batch/views分别按seed/client/step定随机流，0-worker和多worker可核对 |
| 有界预取 | 全进程共享一个client worker池；默认每worker预取1个client任务，不是每ES都创建4个worker |

## 直接运行：不需要重生成服务器已有 YAML

先将本次修改的`EdgeQGFed.py`、`tree/tree.py`、`utils/e1_logging.py`及新增`datasets/parallel_clients.py`同步到服务器的同一项目版本。自测还需同步`tests/test_parallel_client_loading.py`与更新后的`tests/test_e1_aggregation.py`；基准工具为`tools/benchmark_client_loading.py`。其余原有项目文件保留，不只复制单个入口。

下面以第5组seed=0为例；应换成实际尚未完成或明确重新验证的配置，不重复启动已完成实验。

```bash
env -u WANDB_API_KEY -u WANDB_MODE \
  WANDB_BASE_URL=https://api.wandb.ai \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u EdgeQGFed.py \
  --config config/e1_cifar10/mp_sample_quality_s0.yaml \
  --client-workers 4 --cpu-threads 2
```

如果当前已采用其他有效认证方式，沿用它即可；这里使用本会话已成功的W&B配置，不写密钥。

新增选项：

| 选项 | 含义 |
|---|---|
| 不写`--client-workers`，YAML也无client_batch_workers | 保持旧加载/随机输入协议；现有命令不被自动切换 |
| `--client-workers 0` | 新client_keyed_v2协议的串行参考，用于公平速度/数值核验 |
| `--client-workers 2`或`4` | 新协议，使用2或4个持久spawn worker；建议先比较2/4，不盲目增大 |
| `--cpu-threads 2` | 主进程PyTorch CPU计算线程数；启用新协议时默认2；各worker固定1线程 |

CLI实际覆盖并保存到`config_resolved.yaml`：`datasets.client_batch_workers`和`train.cpu_threads`。可选YAML字段`datasets.client_batch_prefetch_factor`默认1，通常先不增加。旧`edge_num_workers=0`不必改；新后端优先使用client_batch_workers。

若验证`eval_num_workers>0`，新协议为验证/测试loader显式采用spawn与持久worker，避免CUDA/SDK初始化后fork。设为0仍可用。训练和验证worker总数应一起考虑，不能只计算client池。

## E1当前四组命令

在同一数据输入协议下比较，所有组使用相同的线程设置；示例均为seed=0。

| E1组 | 命令（Python前的环境变量同上） |
|---|---|
| 1 | `python -u EdgeQGFed.py --config config/e1_cifar10/sample_average_s0.yaml --client-workers 4 --cpu-threads 2` |
| 5 | `python -u EdgeQGFed.py --config config/e1_cifar10/mp_sample_quality_s0.yaml --client-workers 4 --cpu-threads 2` |
| 6 | `python -u EdgeQGFed.py --config config/e1_cifar10/full_sample_quality_s0.yaml --client-workers 4 --cpu-threads 2` |
| 7 | `python -u EdgeQGFed.py --config config/e1_cifar10/quality_average_s0.yaml --client-workers 4 --cpu-threads 2` |

其他seed替换文件名；HARBox在既有划分前置条件解决后同样可用。双实验并行时，上述4个worker是每个训练进程各4个，CPU/内存仍共享；GPU分配沿用已有设置，本开关不自动分卡。

## “等价”的准确范围与已有结果

本次是**同一训练方法下的数据调度替换**，没有删减客户端样本/增强、loss或训练步数。但为使增强不受worker调度影响，新后端采用`client_keyed_v2`随机输入协议。

- 新版0-worker与多worker在相同seed/client/step下应得到相同抽样、成功客户端、三视图与ES batch顺序；CPU合成测试覆盖这些项及单步参数更新。
- **不声称与旧edge_step_v1协议逐样本/逐位相同。** 旧版将drop、抽样、增强、模型随机数交织；新版按client分流，具体mask/views可能不同。
- 已完成的旧run保留；新run会进入新的唯一输出目录，不覆盖它们。正式配对比较应采用同一input_protocol。若第1组已用旧版完成，要与新版5/6/7严格配对，应补一份新版第1组；或者保留完整的旧四组作为另一套结果。不能只因seed相同就混为严格配对。
- GPU kernels尚未强制确定性，CPU测试不能保证服务器GPU训练逐位一致。数据/库版本也必须一致。
- 改的是workstation数据准备速度，不会自动修复旧成本统计；记录loader设置和真实wall-clock，不把加速当作算法通信收益。

`inputs.json`新增input_protocol、client_batch_workers、cpu_threads；`participation.jsonl`新增每client的样本数、抽样索引hash与随机流键，便于核对调度和配对。

## 验证与不训练的速度测试

在相同Conda环境：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

新增测试只使用合成数据和CPU：包含真实CIFAR增强算子，但不读取benchmark、不下载、不初始化追踪SDK。一个小型合成forward/backward校验下游一致性，不是实验训练。

数据准备基准（合成32×32图像，使用现有224三视图增强，不训练、不下载）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python tools/benchmark_client_loading.py \
  --workers 0 2 4 --cpu-threads 2 --clients 8 --batch-size 16 --rounds 5
```

分别输出startup_round_s和steady_round_s；spawn第一次启动会慢，比较稳定段，不拿启动时间冒充每轮加速。该工具只测数据准备，不预测真实训练倍数。服务器上再看GPU空闲、Round Wall Time与评估停顿，选择更快的2/4 worker设置；没有测量就不承诺几倍加速。

## 内存和进程资源

一个client的64样本三视图float32图像约115.6MB。4worker×prefetch1仅预取张量量就可达约462MB；还有主进程拼接ES batch、worker数据副本和模型等内存。并非免费缓存。Linux多进程张量通过共享内存传递，执行前看：

```bash
free -h
df -h /dev/shm
```

若出现bus error/shared memory不足，先降低client worker数或在有权限的部署环境调整共享内存；不要关闭三视图或减少batch后仍宣称同一实验。`train()`在正常完成/异常路径都清理新client池和新协议的评估池，避免只停主训练却留下持久worker。

本次未在远端服务器启动或终止任何进程，未改变已有YAML、论文或正在运行的任务。
