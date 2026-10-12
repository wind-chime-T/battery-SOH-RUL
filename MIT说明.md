MIT 总共三个mat文件，每个文件结构如下

==summary 是"粗"的（每 cycle 一个点）， cycles 是"细"的（每采样点一个数据）==

#### summary 部分（摘要数据）

| 字段 | 含义 | 数据维度 |
|:-----|:-----|:---------|
| IR | 内阻 | 每个 cycle 1个值 |
| QC | 充电容量 | 每个 cycle 1个值 |
| QD | 放电容量 | 每个 cycle 1个值 |
| Tavg | 平均温度 | 每个 cycle 1个值 |
| Tmin | 最低温度 | 每个 cycle 1个值 |
| Tmax | 最高温度 | 每个 cycle 1个值 |
| chargetime | 充电时间 | 每个 cycle 1个值 |
| cycle | 循环编号 | 每个 cycle 1个值 |
| knee_onset | P1 拐点（减速衰退）cycle 索引 | 1 个值（或 None） |
| knee_point | P2 拐点（加速衰退）cycle 索引 | 1 个值（或 None） |
| eol_cycle | 容量首次低于 80% 额定容量的 cycle | 1 个值 |

#### cycles 部分（详细循环数据）

| 字段 | 含义 | 数据维度 |
|:-----|:-----|:---------|
| t | 时间 | 每个采样点1个值 |
| I | 电流 | 每个采样点1个值 |
| V | 电压 | 每个采样点1个值 |
| Qc | 充电容量 | 每个采样点1个值 |
| Qd | 放电容量 | 每个采样点1个值 |
| T | 温度 | 每个采样点1个值 |
| Qdlin | 线性插值放电容量 | 每个采样点1个值（重采样到统一电压网格） |
| Tdlin | 线性插值温度 | 每个采样点1个值（重采样到统一电压网格） |
| dQdV | 放电 dQ/dV | 每个采样点1个值 |

***

## mat格式
```
f (h5py.File)
└── 'batch'                          # 顶层组
    ├── 'cycle_life'      [num_cells]  # 每个电池的循环寿命（引用）
    ├── 'policy_readable' [num_cells]  # 每个电池的充电策略（引用）
    ├── 'summary'         [num_cells]  # 每个电池的摘要数据（引用）
    │   └── [i] →
    │       ├── 'IR'          [1, N]   # 内阻
    │       ├── 'QCharge'     [1, N]   # 充电容量
    │       ├── 'QDischarge'  [1, N]   # 放电容量
    │       ├── 'Tavg'        [1, N]   # 平均温度
    │       ├── 'Tmin'        [1, N]   # 最低温度
    │       ├── 'Tmax'        [1, N]   # 最高温度
    │       ├── 'chargetime'  [1, N]   # 充电时间
    │       └── 'cycle'       [1, N]   # 循环编号
    └── 'cycles'          [num_cells]  # 每个电池的循环详细数据（引用）
        └── [i] →
            ├── 'I'              [N_cycles]  # 电流
            ├── 'Qc'             [N_cycles]  # 充电容量
            ├── 'Qd'             [N_cycles]  # 放电容量
            ├── 'Qdlin'          [N_cycles]  # 线性插值放电容量
            ├── 'T'              [N_cycles]  # 温度
            ├── 'Tdlin'          [N_cycles]  # 线性插值温度
            ├── 'V'              [N_cycles]  # 电压
            ├── 'discharge_dQdV' [N_cycles]  # 放电 dQ/dV
            └── 't'              [N_cycles]  # 时间

```


## pkl格式:  "DATA_bat_dict.pkl"
```
bat_dict (dict, ~124 个电池)
│
├── 'b1c0' ────────────────────────── 单个电池字典
│   ├── 'cycle_life' : int           → 循环寿命
│   ├── 'charge_policy' : str        → 充电策略
│   │
│   ├── 'summary' ────────────── 摘要数据 dict (每cycle一个点)
│   │   ├── 'IR'      : (N,) float   内阻
│   │   ├── 'QC'      : (N,) float   充电容量
│   │   ├── 'QD'      : (N,) float   放电容量
│   │   ├── 'QD_fixed'   : (N,) float   离群点处理+插值后的放电容量
│   │   ├── 'Tavg'    : (N,) float   平均温度
│   │   ├── 'Tmin'    : (N,) float   最低温度
│   │   ├── 'Tmax'    : (N,) float   最高温度
│   │   ├── 'chargetime' : (N,) float 充电时间
│   │   ├── 'cycle'      : (N,) int     循环编号
│   │   ├── 'knee_onset' : int or None  P1 拐点 cycle 索引
│   │   ├── 'knee_point' : int or None  P2 拐点 cycle 索引
│   │   └── 'eol_cycle'  : float        容量首次低于 80% 的 cycle
│   │   └── 'RUL'        : float        剩余周期数
│   │
│   └── 'cycles' ───────────── 详细循环数据 dict
│       ├── '0'  ───────────── 第0个cycle
│       │   ├── 'I'      : (n,) float   电流
│       │   ├── 'V'      : (n,) float   电压
│       │   ├── 't'      : (n,) float   时间
│       │   ├── 'Qc'     : (n,) float   充电容量
│       │   ├── 'Qd'     : (n,) float   放电容量
│       │   ├── 'T'      : (n,) float   温度
│       │   ├── 'Qdlin'  : (n,) float   插值放电容量
│       │   ├── 'Tdlin'  : (n,) float   插值温度
│       │   └── 'dQdV'   : (n,) float   dQ/dV
│       │
│       ├── '1'  ───────────── 第1个cycle
│       ├── '2'  ───────────── 第2个cycle
│       └── ... (直到 cycle_life-1)
│
├── 'b1c1' ──────────────────────────
├── 'b1c2' ──────────────────────────
├── ...
├── 'b2c0' ──────────────────────────
├── 'b2c1' ──────────────────────────
├── ...
├── 'b3c0' ──────────────────────────
└── ...

```

## 访问 bat_dict 方式
```
bat_dict = pickle.load(open('DATA_bat_dict.pkl', 'rb'))
# 放电容量衰减曲线（优先用 QD_fixed，已去离群点+插值）
bat_dict['b1c0']['summary']['QD_fixed']     # (N,) array
# 第10个cycle的V-Q放电曲线
bat_dict['b1c0']['cycles']['10']['V'] # (n,) array
# P1 / P2 拐点 cycle 索引
bat_dict['b1c0']['summary']['knee_onset']   # int or None
bat_dict['b1c0']['summary']['knee_point']   # int or None
# 寿命终点 cycle（容量首次低于 80% 额定容量）
bat_dict['b1c0']['summary']['eol_cycle']    # float
```
