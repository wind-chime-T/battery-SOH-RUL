# CNN1D + Transformer 容量预测（→ SOH）— MIT 数据集

## 概述

基于 MIT-Stanford 电池老化数据集（124 颗 LFP 电池），使用 **CNN1D + Transformer 编码器** 模型**回归**预测电池当前放电容量和剩余使用寿命（→ SOH, RUL）。

```
5-cycle V/I/T（容量轴固定 200 点重采样）──→ 模型 ──→ 容量 Q(t)
SOH(t) = Q(t) / Q_max
```

**核心设计**：从局部窗口（5 cycle）回归最后一个 cycle 的放电容量，部署时只用现有循环数据（当前窗口）。

与 LSTM 项目（`模型/LSTM/`）的**同一输入口径、同一评估流程**对比，可在 `模型/LSTM/LSTM_soh_MIT.md` 直接横向比较 MAE/MAPE。

**两个注意力模块**（用户指定）：
- **CNN 后**：ECA 通道注意力（复用 LSTM 项目的 `ECAAttention`）
- **Transformer 后**：Attention Pooling（softmax 加权求和）

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
| `cycles[str(cn)]['V']` / `I` / `T` | `(n,) ndarray` | 单个 cycle 的细粒度采样（放电用，`I < -0.01A`） |

> 本模型只用 V / I / T 三个通道，与 LSTM 项目 可视为纯时序输入的对照。

### EOL 定义（评估时）

与 LSTM 项目一致，使用**固定额定容量**阈值（与预处理 `eol_cycle` 字段、`process_MIT.ipynb` 的 `threshold = 0.8 * 1.1` 同口径）：

```
EOL 阈值 = 0.80 × 1.1 Ah = 0.88 Ah    # 额定容量 1.1 Ah（固定，不随电池变化）
RUL(t) = EOL - t                      # 5-cycle 移动平均平滑后求首次低于阈值
```

---

## 特征工程

### 两套预处理方案（cell 4，`PREPROC_MODE` 一键切换）

| 方案 | `PREPROC_MODE` | 做法 | 特征维度 |
|:-----|:---------------|:-----|:---------|
| 1 原始定长 | `'raw'` | 放电段原始时间分辨率，全局 99 分位长度（458）尾部 zero-padding / 截断 | 458×3 = 1374 |
| 2 容量轴重采样（默认） | `'resample'` | 放电段按**累计放电容量 Q=∫(−I)dt** 重采样到固定 200 点 | 200×3 = **600** |

### 注：方案 2 → 容量轴固定重采样（`extract_cycle_timeseries_resample`）

以**放电量**而非时间/SOC 作为重采样轴，天然对齐不同 cycle 的放电深度（老化后放电容量变小，时长也变，容量轴等价于 SOC 轴）：

```
mask = I < -0.01                     # 放电段
Q    = cumsum(-I × dt)               # 累计放电容量（t 缺失时退化按索引）
Q_new = linspace(Q[0], Q[-1], 200)   # 均匀网格
V / I / T  各用 np.interp(Q_new, Q, ·) 重采样到 200 点
```

**通道排列**（`(200×3,)` → 600 维）：
```
[0:200]      V
[200:400]    I
[400:600]    T
```

特征形状 `(5, 600)` = `(window_size, RESAMPLE_LEN×3)`。

### 标签

每个窗口的标签 = 窗口最后一个 cycle 的放电容量 `QD_fixed[last_idx]`（单标签回归，与 LSTM 同口径）。

---

## 数据划分与窗口

### 电池级分层划分

- 按 `cycle_life` 三分位分组（短寿 / 中寿 / 长寿），确保三层在各集合比例一致
- 70% 训练 / 15% 验证 / 15% 测试

### 滑动窗口

- `window_size = 5` cycles
- `stride = 3` cycles

### Z-score 归一化

- 特征：训练集 `mean` / `std`（每条特征维度独立归一化）
- 容量标签：训练集容量 `mean` / `std`（val/test 复用训练集统计量）

### 数据集规模

| 集合 | 电池数 | 窗口数 |
|:-----|:-------|:-------|
| Train | 86 | 23,077 |
| Val | 19 | 5,341 |
| Test | 19 | 4,917 |

特征缓存：124 电池，100,377 窗口，约 172 MiB。容量标签范围：train [0.827, 1.098]，mean=1.032，std=0.052。

---

## 模型架构

### CNN1DTransformer

```
输入: (B, 5, 600)                     # 600 = 3通道(V/I/T) × 200点，布局 [V(200),I(200),T(200)]
       │  reshape → (B, 5, 3, 200) → (B*5, 3, 200)   # 5 个 cycle 共享同一套卷积权重
  ┌────▼───────────────────────────────┐
  │ CNN1D（3 层，逐 cycle 共享权重）     │
  │ Conv1d(3→32, k7, s2, p3) → BN→ReLU │   200→100
  │ Conv1d(32→64, k5, s2, p2) → BN→ReLU│   100→50
  │ Conv1d(64→C, k3, s2, p1) → BN→ReLU │   50→25    (C = conv_channels，默认 = d_model)
  │ AdaptiveAvgPool1d(1)               │   25→1
  └────┬───────────────────────────────┘
       │ (B, 5, C)   5 个 cycle token
  ┌────▼──────────────────┐
  │ ECA 通道注意力         │  ← CNN 后注意力（作用于 token 通道维）
  └────┬──────────────────┘
       │ + learned 位置编码 (B,5,d_model)
  ┌────▼───────────────────────────┐
  │ TransformerEncoder(2层, nhead=4│
  │ dim_feedforward=256, gelu,     │
  │ batch_first)                   │
  └────┬───────────────────────────┘
       │ (B, 5, d_model)
  ┌────▼──────────────────┐
  │ Attention Pooling     │  ← Transformer 后注意力：Linear(d_model→1) + softmax(dim=1) + 加权求和
  └────┬──────────────────┘
       │ (B, d_model)
  ┌────▼──────────────────────┐
  │ Shared: Linear→64, ReLU,  │
  │ Dropout(0.1)              │
  └────┬──────────────────────┘
  ┌────▼──────────────────────┐
  │ cap_head  Linear(64→1)    │  容量预测（归一化值）
  └───────────────────────────┘
  输出: (B,)
```

**组件分工**：

| 组件 | 作用 | 操作维度 |
|:-----|:-----|:---------|
| CNN1D | 从每条放电曲线提取局部形态（电压平台、容量拐弯） | 单个 cycle 的 200 采样点 |
| ECA | 自适应强化对容量敏感的 token 特征通道 | 通道维 (d_model) |
| 位置编码 | 编码 cycle 先后顺序（learned） | 时间步 (5 cycles) |
| Transformer | 建模 5 个 cycle 之间（局部与跨 cycle）的交互 | 时间步 (5 cycles) |
| Attention Pooling | 学习 5 个 cycle 中哪些对容量预测最关键的权重 | 时间维 (5 cycles) |

**参数量**：默认配置（d_model=128, nhead=4, num_layers=2, ffn=512）**441,799**。

> `conv_channels` 默认 `None` → 自动取 `d_model`，保证 CNN 输出通道与 Transformer 输入一致（曾因固定 128 与 d_model 不等导致 reshape 报错，cell 6 已修）。

---

## 网格搜索（cell 9）

与 `模型/LSTM/LSTM_soh_MIT.ipynb` cell 9 同构：`itertools.product` 笛卡尔积，每组合**全量训练到早停**（epochs=1000, patience=50），每组重新播种 seed=1；搜索过程中保存最佳 `state_dict`，结束后**直接加载、不再二次重训**。

### 搜索空间

```
lr = [1e-4, 5e-4]
d_model = [64, 128]
nhead = [4, 8]
num_layers = [1, 2]
dim_feedforward = [256, 512]
```
2×2×2×2×2 = **32 组**。约束：`nhead` 整除 `d_model`。

### 前 5 名（按 val_loss 升序）

| lr | d_model | nhead | num_layers | ffn | val_loss | params |
|:---|:--------|:------|:-----------|:----|:---------|:-------|
| 5e-4 | 128 | 4 | 2 | 256 | **0.0287** | 310,215 |
| 5e-4 | 128 | 8 | 2 | 256 | 0.0297 | 310,215 |
| 5e-4 | 128 | 8 | 2 | 512 | 0.0363 | 441,799 |
| 5e-4 | 64 | 8 | 1 | 256 | 0.0369 | 78,277 |
| 1e-4 | 128 | 4 | 1 | 512 | 0.0380 | 243,527 |

**最佳参数**：

| 参数 | 最佳值 |
|:-----|:-------|
| LR | 5.0e-04 |
| d_model | 128 |
| nhead | 4 |
| num_layers | 2 |
| dim_feedforward | 256 |

→ val_loss **0.0287**。对比基线固定配置（d=128, nhead=4, nl=2, ffn=512, lr=1e-4，val_loss 0.0411）下降约 30%。

规律：**大 lr（5e-4）+ 偏窄 ffn（256）** 显著更优；d_model=128 优于 64。

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

### 随机种子

```python
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
```

---

## 评估（最优模型，测试集）

| 指标 | 容量 | RUL（窗口）| RUL（电池）|
|:-----|:---|:---|:---|
| MAE | 0.0059 Ah | 3.66 cycles | 4.47 cycles |
| RMSE | 0.0082 Ah | 8.04 cycles | 8.39 cycles |
| MAPE | 0.58% | — | — |

### RUL MAPE（cell 10 追加）

RUL 的 MAPE 有**近零分母问题**：真实 RUL 临近 EOL 时趋近 0，`|err|/RUL_true` 爆炸。按电池 RUL 文献惯例，只统计**真实 RUL ≥ 阈值**的窗口（`RUL_MAPE_MIN = 10.0`，改阈值改这一处）：

| 指标 | 值 | 覆盖 |
|:-----|:---|:-----|
| 逐窗口 RUL MAPE（真实RUL≥10）| **2.45%** | 14,379 / 14,557 窗口 |
| 逐电池 EOL MAPE | **0.77%** | 19 电池（EOL 量级大，无近零分母） |

RUL_true 范围：[0, 1309]；19 颗测试电池中 7 颗在记录期内真正跌破 0.88 Ah（`n_genuine`），其余为右删失样本（RUL_pred / RUL_true 使用同一 fallback 末圈）。

### 与 LSTM 项目对比（同评估流程）

| 模型 | 容量 MAE (Ah) | 容量 MAPE | RUL 窗口 MAE |
|:-----|:-------------|:----------|:-------------|
| **CNN1D+Transformer（本 notebook）** | **0.0059** | **0.58%** | 3.66 |
| BiLSTM | 0.0068 | 0.67% | 1.31 |
| BiLSTM_MoE | 0.0055 | 0.54% | 0.89 |

> ⚠️ **RUL 口径不一致，不可直接横比**：本 notebook 用固定额定 **0.88 Ah** 阈值（与预处理 `eol_cycle` 同口径）；LSTM 项目评估代码用**每电池 80% × Q_max**。这实际上是代码`LSTM_soh_MIT.ipynb`和其文档不一致。
> 容量指标不受 EOL 定义影响，可直接对比：CNN1D+Transformer 已接近 BiLSTM_MoE 的水平（纯时序输入、无标量特征、无 MoE）。

---

## 切换预处理方案

cell 4 改一行即可：

```python
PREPROC_MODE = 'resample'   # 'raw' 或 'resample'
```

- `'resample'`：`FEAT_DIM = 600`，模型 `resample_len=200`
- `'raw'`：`FEAT_DIM = 1374`，模型需改 `resample_len` 语义（raw 无重采样轴，为对照方案）

改方案后**必须**重建数据集（cell 5）并重新训练（cell 9 网格 / 训练），`meta` 中的 `feat_dim` / `resample_len` 会自动跟随。

---

## 代码结构 (13 Cells)

| Cell | 内容 |
|:-----|:-----|
| 0 | Markdown 标题与特征方案说明 |
| 1 | 依赖导入 |
| 2 | 数据加载 + 快速结构展示 |
| 3 | 全局 99 分位放电段长度统计 + 分布直方图（FIXED_LEN=458） |
| 4 | `extract_cycle_timeseries`（方案1 raw）/ `extract_cycle_timeseries_resample`（方案2 容量轴 200 点）+ `PREPROC_MODE` 调度 + `FEAT_DIM` |
| 5 | `build_dataset`：分层划分 + 特征缓存 + 滑动窗口(5,3) + Z-score（容量标签） |
| 6 | `BatteryDataset` + `ECAAttention` + **`CNN1DTransformer`** + 形状/参数量自测 |
| 7 | 训练工具 `run_epoch` + `train_model` + Dataset 实例化 |
| 8 | 随机种子设置 |
| 9 | 参数搜索（`product` 网格，lr×d_model×nhead×num_layers×ffn = 32 组）→ 保存最佳 state 并加载 |
| 10 | 评估：`predict_battery`（分块推理）+ 容量指标 + RUL + **RUL MAPE** + EOL MAPE |
| 11 | 容量散点图 + 误差分布直方图 |
| 12 | 测试集全部电池容量预测 vs 真实衰减曲线 |

---

## 依赖

| 包 | 用途 |
|:---|:-----|
| `numpy` | 数值计算、特征重采样 |
| `matplotlib` | 可视化 |
| `torch` / `torch.nn` | CNN / ECA / Transformer 模型 |
| `sklearn.model_selection.train_test_split` | 分层数据划分 |
| `copy.deepcopy` | 模型最佳状态保存 |
| `pickle` | 加载 .pkl 数据 |

（无 `st_moe_pytorch` 依赖，与 LSTM 项目不同。）

---

## 关键设计决策

1. **容量轴固定重采样而非原始时间/SOC**：以 `Q=∫(−I)dt` 为轴统一到 200 点，消除放电时长差异、无 padding/截断，等价于按放电深度对齐（方案 2 默认；方案 1 raw 保留对照）
2. **CNN 逐 cycle 共享权重**：5 个 cycle 用同一套 Conv1d 提取局部曲线形态，池化后得到 5 个 token
3. **CNN 后接 ECA 通道注意力**：自适应强化对容量敏感的特征通道
4. **Transformer 后接 Attention Pooling**：让模型学习 5 个 cycle 中哪些对容量预测最关键的权重（而非取最后时间步）
5. **LSTM 同款网格搜索**：`product` 全量训练 + 保存最佳 state 直接加载（不二次重训），每组合重播种保证可复现
6. **`conv_channels` 默认 = `d_model`**：避免网格搜索调整 d_model 时 CNN 输出通道与 Transformer 输入不匹配（根因修复）
