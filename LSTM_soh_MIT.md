# LSTM 容量预测（→ SOH）— MIT 数据集

## 概述

基于 MIT-Stanford 电池老化数据集（124 颗 LFP 电池），使用深度学习模型**回归**预测电池当前放电容量和剩余使用寿命（→ SOH, RUL）。

当前提供四个模型：

| 模型 | 架构 | 定位 |
|:-----|:-----|:-----|
| `BiLSTM` | BiLSTM + Attention Pooling | 基准 |
| `CNN_BiLSTM_ECA` | CNN + BiLSTM + ECA + Attention Pooling | 增强 |
| `CNN_BiLSTM` | CNN(单层) + BiLSTM + Attention Pooling | 对照 |
| `BiLSTM_MoE` | BiLSTM + MoE（稀疏专家）+ Attention Pooling | 增强 |

```
5-cycle V/I/dQdV/T + 标量特征 ──→ 模型 ──→ 容量 Q(t)
SOH(t) = Q(t) / Q_max
```

**核心设计**：从局部窗口（5 cycle）回归当前容量，部署时只用现有循环数据（当前窗口 + 标量特征）。

---

## 数据

### 数据源

- 文件：[DATA_bat_dict.pkl](D:\Python\data_bag\MIT_data\processed\DATA_bat_dict.pkl)
- 电池数：124 颗（LFP，多种快充策略 1C–8C）
- 结构：`bat_dict[name]['summary']`（每 cycle 一个点）+ `bat_dict[name]['cycles']`（每 cycle 细粒度采样）

### 关键字段

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `summary['QD_fixed']` | `(N,) ndarray` | 离群点处理 + 插值后的放电容量（Ah） |
| `summary['cycle']` | `(N,) ndarray` | 循环编号 |
| `summary['eol_cycle']` | float | 容量首次低于 80% 额定容量的 cycle |
| `summary['IR']` / `Tavg` / `chargetime` | `(N,) ndarray` | 内阻、平均温度、充电时间 |
| `cycles[str(cn)]['V']` / `I` / `T` / `dQdV` | `(n,) ndarray` | 单个 cycle 的细粒度采样 |

### EOL 定义

MIT 数据集：容量首次低于 **0.88 Ah**（1.1 Ah 额定容量的 80%）。

---

## 特征工程

### 时序特征（1832 维）

| 通道 | 提取方式 | 维度 |
|:-----|:---------|:-----|
| V（电压） | 放电段 `I < -0.01A`，保留原始时间分辨率 | 458 |
| I（电流） | 同上 | 458 |
| T（温度） | 同上 | 458 |
| dQdV | 电压网格上计算，`np.interp` 重采样到放电段长度 | 458 |

**固定长度策略**：取**全局所有 cycle 放电段长度的 99 分位**（= 458 点）作为统一维度。短于该长度的 zero-padding 尾部，长于该长度的截断尾部，保留原始时间分辨率。

**dQdV 特殊处理**：dQdV 在电压网格上计算（长度通常 ≠ V/I/T 长度），用 `np.interp` 线性重采样到与 V/I/T 相同的放电段长度后拼接。

**通道排列**（`(458×4,)` → 1832 维）：
```
[0:458]          V
[458:916]        I
[916:1374]       T
[1374:1832]      dQdV
```

### 标量特征（6 维）

| 特征 | 说明 |
|:-----|:-----|
| ΔQ₁ | 当前容量 − 前 1 cycle 容量 |
| ΔQ₅ | 当前容量 − 前 5 cycle 容量 |
| ΔQ₂₀ | 当前容量 − 前 20 cycle 容量 |
| IR | 当前 cycle 内阻 |
| Tavg | 当前 cycle 平均温度 |
| chargetime | 当前 cycle 充电时间 |



### 总特征维度

```
1832（时序）+ 6（标量）= 1838
```


---

## 数据划分与窗口

### 电池级分层划分

- 按 `cycle_life` 三分位分组（短寿 / 中寿 / 长寿），确保三层在各集合中比例一致
- 70% 训练 / 15% 验证 / 15% 测试

### 滑动窗口

- `window_size = 20` cycles
- `stride = 3` cycles

### Z-score 归一化

- 特征：训练集 `mean` / `std`（每条特征维度独立归一化）
- 容量标签：训练集容量 `mean` / `std`

### 数据集规模

| 集合 | 电池数 | 窗口数 |
|:-----|:-------|:-------|
| Train | 86 | 22,647 |
| Val | 19 | 5,246 |
| Test | 19 | 4,822 |

特征缓存占用约 704 MiB 内存。

---

## 模型架构

### BiLSTM（基准）

```
输入: (B, window_size, 1838)
       │
  ┌────▼──────────────────────────────┐
  │  Input Projection                 │
  │  Linear(input→256) → LayerNorm →  │
  │  ReLU → Dropout                   │
  └────┬──────────────────────────────┘
       │  (B, window_size, 256)   proj_dim = hidden_dim × 2
       │
  ┌────▼──────────────────────────────────┐
  │  BiLSTM(256→128, 2层, bidirectional)  │
  │  dropout=0.1                          │
  └────┬──────────────────────────────────┘
       │  (B, window_size, 256)   256 = 128×2 (双向)
       │
  ┌────▼───────────────┐
  │  Attention Pooling │  Linear(256→1) → softmax → 加权求和
  └────┬───────────────┘
       │  (B, 256)
       │
  ┌────▼──────────────────────┐
  │  Shared: Linear→64, ReLU, │
  │  Dropout(0.1)             │
  └────┬──────────────────────┘
       │  (B, 64)
       │
  ┌────▼──────────────────────┐
  │  cap_head  Linear(64→1)   │  容量预测（归一化值）
  └───────────────────────────┘
  输出: (B,)
```

### CNN_BiLSTM_ECA

```
输入: (B, window_size, 1838) → 拆分为时序(1832) + 标量(6)

  时序 (B, ws, 1832)                       标量 (B, ws, 6)
    │ reshape → (B, ws, 4, 458)                  │
    │                                            │
  ┌─▼─────────────────────────────┐              │
  │ 2D CNN (3层)                  │              │
  │  Conv2d(4→32, (3,7), s(1,2))  │              │
  │  Conv2d(32→64, (3,5), s(1,2)) │              │
  │  Conv2d(64→128, (3,3), s(1,2))│              │
  │  AdaptiveAvgPool2d((20,1))    │              │
  └─┬─────────────────────────────┘              │
    │ (B, ws, 128)              └─────────┬──────┘
    │                          concat → (B, ws, 134)
    │
  ┌─▼──────────────────────────────────┐
  │ BiLSTM(134→128, 2层)               │
  └─┬──────────────────────────────────┘
    │  (B, ws, 256)
    │
  ┌─▼──────────────────────┐
  │ ECA 通道注意力          │  1D Conv over channels → 学习通道依赖
  │ kernel=k_adapative     │
  └─┬──────────────────────┘
    │  (B, ws, 256)
    │
  ┌─▼────────────────┐
  │ Attention Pooling│  Linear(256→1) → softmax → 加权求和（时间维度）
  └─┬────────────────┘
    │  (B, 256)
    │
  ┌─▼───────────────────────┐
  │ Shared: Linear→64, ReLU,│
  │ Dropout(0.1)            │
  └─┬───────────────────────┘
    │  (B, 64)
    │
  ┌─▼──────────────────────┐
  │ cap_head  Linear(64→1) │  容量预测（归一化值）
  └────────────────────────┘
  输出: (B,)
```

**四组件分工**：

| 组件 | 作用 | 操作维度 |
|:-----|:-----|:---------|
| CNN | 从每条放电曲线提取局部形态（dQdV峰、电压平台斜率） | 单个 cycle 的 458 采样点 |
| BiLSTM | 捕捉 ws 个 cycle 之间特征的时间演变 | 时间步 (ws cycles) |
| ECA | 自适应强化对容量敏感的特征通道 | 通道维 (256) |
| Attention Pooling | 学习 ws 个时间步的重要性权重 | 时间维 (ws) |

### CNN_BiLSTM

与 `CNN_BiLSTM_ECA` 的区别：**去掉 ECA** 通道注意力，CNN 改为**单层**、卷积核数量降到 16（`cnn_channels` 构造参数，可调）。改 2 层的方法见 cell 6 代码注释：取消注释 `cnn2d` 里的第二层卷积，并把 `cnn_dim` 改为 `cnn_channels * 2`。

```
输入: (B, window_size, 1838) → 拆分为时序(1832) + 标量(6)

  时序 (B, ws, 1832)                       标量 (B, ws, 6)
    │ reshape → (B, ws, 4, 458)                 │
    │                                            │
  ┌─▼─────────────────────────────┐              │
  │ 2D CNN (单层)                 │              │
  │  Conv2d(4→16, (3,7), s(1,2))  │  曲线轴 458→229→1 │
  │  AdaptiveAvgPool2d((ws,1))    │              │
  └─┬─────────────────────────────┘              │
    │ (B, ws, 16)               └─────────┬──────┘
    │                          concat → (B, ws, 22)
    │
  ┌─▼──────────────────────────────────┐
  │ BiLSTM(22→hidden, 双向)            │  hidden/层数由网格搜索
  └─┬──────────────────────────────────┘
    │  (B, ws, hidden×2)
    │
  ┌─▼────────────────┐
  │ Attention Pooling│  Linear(hidden×2→1) → softmax → 加权求和（时间维度）
  └─┬────────────────┘
    │  (B, hidden×2)
    │
  ┌─▼───────────────────────┐
  │ Shared: Linear→64, ReLU,│
  │ Dropout(0.1)            │
  └─┬───────────────────────┘
    │  (B, 64)
    │
  ┌─▼──────────────────────┐
  │ cap_head  Linear(64→1) │  容量预测（归一化值）
  └────────────────────────┘
  输出: (B,)
```

注：代码中池化用 `AdaptiveAvgPool2d((self.num_steps, 1))`，`num_steps = window_size`（实际运行 `window_size=5`）。

### BiLSTM_MoE

在基准 `BiLSTM` 之上，把 LSTM 输出的每个 cycle 隐状态送入 **Mixture of Experts**（`st_moe_pytorch`，参考 `参考代码/MoE.ipynb`）：

```
输入: (B, window_size, 1838)
       │  Input Projection（同 BiLSTM）
       ▼
       BiLSTM  →  lstm_out (B, ws, hidden×2)
       │
       ├─ LayerNorm（PreNorm，稳定路由）
       ├─ MoE(dim=hidden×2, N 专家, top-2 gating)   每个 cycle token 独立路由
       └─ + 残差连接
       │
       Attention Pooling → Shared → cap_head → 容量 Q(t)
```

**MoE 关键设计**：
- 参数量 ×N，但单 token 计算量仅 ×top_n（稀疏激活，`gating_top_n=2`）——容量大、算力不变
- 路由器学习"不同老化阶段 / 充电策略 → 不同专家"，覆盖多模态老化路径
- 训练时引入**专家负载均衡损失 + router z-loss**，经 `run_epoch` 的 `aux_weight` 加权进总损失；**验证损失保持纯 MSE**（与 BiLSTM 直接可比）
- 评估时库内部自动切换 `threshold_eval` / `capacity_factor_eval`（推理放宽容量、不丢 token）
- `forward` 返回 `cap_pred`（与 BiLSTM 相同），辅助损失记到 `self.last_aux_loss` 供训练使用

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `num_experts` | 8 | 专家个数（网格搜索 [4,8,16]） |
| `expert_hidden_mult` | 2 | 单专家 FFN 隐藏层 = dim × mult |
| `gating_top_n` | 2 | 每 token 激活专家数 |
| `threshold_train` / `threshold_eval` | 0.2 / 0.2 | 路由接受阈值 |
| `capacity_factor_train` / `capacity_factor_eval` | 1.25 / 2.0 | 专家容量因子 |
| `balance_loss_coef` | 1e-2 | 负载均衡损失权重 |
| `router_z_loss_coef` | 1e-3 | router z-loss 权重 |

### 参数量对比

| 模型 | hidden=64, layers=3（本次最佳） | hidden=128, layers=3 |
|:-----|:-------------------------------|:---------------------|
| BiLSTM | ~54.3万 | ~168万 |
| CNN_BiLSTM_ECA | ~42.0万 | ~119万 |
| CNN_BiLSTM (cnn_channels=16) | ~25.4万 | ~96.4万 |
| BiLSTM_MoE (8 experts, mult=2) | ~97万 | ~380万* |

（实测参数量。BiLSTM 的输入投影把 1838 维直接线性压到 hidden×2，同规模下参数反而多于先经 CNN 降维（1832→128）的 CNN_BiLSTM_ECA。）

\* BiLSTM_MoE 的 hidden=128 为估算值。MoE 参数量 ≈ `num_experts × 2 × (dim × dim×mult)`（dim = hidden×2），随专家数近似线性增长：hidden=64 时 4→8→16 专家约 52.8 万→105.7 万→211.3 万（实测 hidden=64、8 专家总参 97 万，其中 MoE 52.8 万）。

---

## 训练

### 训练配置

| 参数 | 值 |
|:-----|:---|
| Batch size | 128 |
| Optimizer | AdamW, weight_decay=5×10⁻⁴ |
| Loss | `MSE_cap`（容量单标签） |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=8) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=50 epochs |
| Max epochs | 1,000 |

### 网格搜索参数
`lr=[1e-4]`, `hidden_dim=[32,64,128]`, `num_layers=[1,2]`, `dropout=[0.1]`, `batch_size=[128]`, `num_experts=[4,8,16]`（`num_experts` 仅 `BiLSTM_MoE` 使用）

**最佳参数（BiLSTM）**：
| 参数 | 最佳值 |
|:-----|:-------|
| LR | 1.0e-04 |
| hidden_dim | 64 |
| num_layers | 1 |
| dropout | 0.1 |
| batch_size | 128 |


**最佳参数（CNN_BiLSTM_ECA）**：
| 参数 | 最佳值 |
|:-----|:-------|
| LR | 6.8e-05 |
| hidden_dim | 64 |
| num_layers | 1 |
| dropout | 0.09 |
| batch_size | 32 |

**最佳参数（CNN_BiLSTM）**：
| 参数 | 最佳值 |
|:-----|:-------|
| LR | 6.8e-05 |
| hidden_dim | 64 |
| num_layers | 1 |
| dropout | 0.09 |
| batch_size | 32 |

**最佳参数（BiLSTM_MoE）**：
| 参数 | 最佳值 |
|:-----|:-------|
| LR |  |
| hidden_dim |  |
| num_layers |  |
| dropout |  |
| batch_size |  |
| num_experts |  |

### 随机种子

```python
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
```

---

## 评估
### BiLSTM ：
| 指标 | 容量 | RUL（窗口）| RUL（电池）|
|:-----|:---|:---|:---|
| MAE | 0.0068 Ah | 1.31 cycle | 1.95 cycle |
| RMSE | 0.0091 Ah | 3.12 cycle | 3.79 cycle |
| MAPE | 0.67% | - | - |

### CNN_BiLSTM_ECA：
| 指标 | 容量 | RUL（窗口）| RUL（电池）|
|:-----|:---|:---|:---|
| MAE | 0.0111 Ah | 3.46 cycles | 5.05 cycles |
| RMSE | 0.0141 Ah | 6.36 cycles | 7.83 cycles |
| MAPE | 1.10% | - | - |

### CNN_BiLSTM：
| 指标 | 容量 | RUL（窗口）| RUL（电池）|
|:-----|:---|:---|:---|
| MAE | 0.0121 Ah | 3.87 cycles | 5.74 cycles |
| RMSE | 0.0157 Ah | 7.36 cycles | 9.02 cycles |
| MAPE | 1.18% | - | - |

### BiLSTM_MoE：
| 指标 | 容量 | RUL（窗口）| RUL（电池）|
|:-----|:---|:---|:---|
| MAE | 0.0055 Ah | 0.89 cycles | 1.32 cycles |
| RMSE | 0.0076 Ah | 2.78 cycles | 3.36 cycles |
| MAPE | 0.54% | - | - |

RUL_true 范围: [0, 1309]


## 切换模型

Cell 9（参数搜索）中切换模型时，改动类名即可。四个模型共享 `input_dim / scalar_dim / window_size / hidden_dim / num_layers / dropout` 等参数：

```python
# 基准
temp_model = BiLSTM(input_dim=meta['feat_dim'], scalar_dim=meta['feat_dim_scalar'],
                    window_size=meta['window_size'],
                    hidden_dim=hd, num_layers=nl, dropout=do).to(device)

# 增强（BiLSTM_MoE：多两个 MoE 参数）
temp_model = BiLSTM_MoE(input_dim=meta['feat_dim'], scalar_dim=meta['feat_dim_scalar'],
                        window_size=meta['window_size'],
                        hidden_dim=hd, num_layers=nl, dropout=do,
                        num_experts=ne, expert_hidden_mult=2).to(device)
```

⚠️ **仅 `BiLSTM_MoE` 支持 `num_experts` / `expert_hidden_mult`**。切回 `BiLSTM` / `CNN_BiLSTM_ECA` / `CNN_BiLSTM` 时，需同步删掉 cell 9 中**网格循环**和**加载最佳模型**两处的这两行，否则抛 `TypeError: unexpected keyword argument`（其余模型无此参数）。

`run_epoch` 支持带 MoE 辅助损失的模型（`aux_weight=1.0`，默认仅训练时加权），对普通模型无任何影响。四个模型共享相同的训练流程与评估函数。

---

## 代码结构 (13 Cells)

| Cell | 内容 |
|:-----|:-----|
| 0 | Markdown 标题与特征方案说明 |
| 1 | 依赖导入 |
| 2 | 数据加载 + 快速结构展示 |
| 3 | 全局 99 分位放电段长度计算 + 分布直方图 |
| 4 | `extract_cycle_timeseries` / `extract_scalar_features` + 特征维度定义 |
| 5 | `build_lstm_dataset`：分层划分 + 特征缓存（窗口 20）+ 滑动窗口 + Z-score 归一化（容量标签） |
| 6 | `BatteryDataset`（单标签）+ `BiLSTM` + `ECAAttention` + `CNN_BiLSTM_ECA` + `CNN_BiLSTM` + `BiLSTM_MoE`（均单头，CNN 按 `window_size` 保留全部时间步） |
| 7 | 训练工具函数 `run_epoch`（容量损失 `MSE_cap` + MoE 辅助损失 `aux_weight`）+ `train_model` + Dataset 实例化 |
| 8 | 随机种子设置 |
| 9 | 参数搜索（含 `num_experts=[4,8,16]`）→ `BiLSTM_MoE` 等一键切换（传入 `scalar_dim` + `window_size`） |
| 10 | 评估：`predict_battery`（分块推理，容量输出）+ 容量指标 |
| 11 | 容量散点图 + 误差分布直方图 |
| 12 | 测试集全部电池容量预测 vs 真实衰减曲线 |
| 13 | 总结打印 |

---

## 依赖

| 包 | 用途 |
|:---|:-----|
| `numpy` | 数值计算、特征缓存 |
| `matplotlib` | 可视化 |
| `torch` / `torch.nn` | BiLSTM / CNN / ECA 模型 |
| `st_moe_pytorch` | `BiLSTM_MoE` 的稀疏专家层（MoE / Top-N gating） |
| `sklearn.model_selection.train_test_split` | 分层数据划分 |
| `copy.deepcopy` | 模型最佳状态保存 |
| `pickle` | 加载 .pkl 数据 |

---

## 关键设计决策

1. **不重采样到固定 SOC 网格**：保留原始时间分辨率 → 99 分位固定长度 + zero-padding
2. **dQdV 与 V/I/T 分离处理**：dQdV 在电压网格上计算，`np.interp` 重采样对齐
3. **Attention Pooling 而非最后时间步**：让模型学习 window_size 个 cycle 中哪些对容量预测最关键
4. **共享接口 + MoE 例外**：四个模型共享 `input_dim / scalar_dim / window_size / hidden_dim / num_layers / dropout`，一键切换；仅 `BiLSTM_MoE` 额外带 `num_experts` / `expert_hidden_mult` 等 MoE 参数，切换回其他模型时需删除（见「切换模型」）
