# AFC-NeuralFoil 工程文档

本文档面向 CFD 工程师和算法工程师，说明当前 AFC 二次开发模块的工程目标、技术路线、数据闭环、训练与推理接口、三维有限翼耦合方式和验证方法。

AFC 是 **Active Flow Control，主动流动控制**。在本项目中，AFC 主要指通过吹气、吸气、合成射流、等离子体激励等方式改变翼型或机翼附近流动，从而影响 `CL`、`CD`、`CM`、转捩、分离和失速行为。当前已实现的是 AFC 代理模型和三维耦合计算的工程骨架，其中一部分是物理/低阶气动模型，一部分是机器学习模型，还有一部分是等待真实 CFD 或实验数据替换的 placeholder。

## 1. 项目目标和技术路线

项目目标是基于现有 AeroSandbox 和 NeuralFoil 架构，扩展出一套可用于 AFC 设计、训练、推理、三维有限翼评估和优化的工程链路。

技术路线：

1. 保留 AeroSandbox 的几何、飞行工况、自动微分、`Opti` 优化、LiftingLine、NonlinearLiftingLine、VLM 等核心能力。
2. 保留 NeuralFoil 的 Kulfan 翼型参数化、轻量 MLP 推理和 `.npz` 权重加载风格。
3. 新增 AFC 相关输入：`C_mu`、`x_jet`、`theta_jet`。
4. 在二维剖面层提供 `airfoil.get_aero_from_afc_neuralfoil(...)`。
5. 在三维有限翼层复用 AeroSandbox 的机翼几何和诱导速度模型，每个展向截面调用 AFC 剖面模型。
6. 用统一 CSV/Parquet schema 管理来自 SU2、RANS、改造 XFoil、风洞或实验的二维 AFC 数据。
7. 用 PyTorch 训练 AFC-NeuralFoil MLP，导出 `.npz` 权重供 `aerosandbox.numpy` 推理器使用。
8. 用主动学习模块根据模型 confidence、梯度、优化关键区域和失速附近自动生成下一批 CFD 任务清单。
9. 通过轻量 pytest、benchmark case 和 validation report 建立回归测试体系。

当前工程定位：

- 已实现完整接口和可运行流程。
- 已实现 placeholder AFC 修正、dummy 权重、synthetic 数据和轻量测试。
- 尚未接入真实 AFC CFD/实验训练数据。
- 尚未宣称 AFC 气动效果具有工程可信度；真实物理可信度依赖后续数据、训练和验证。

## 2. 与 AeroSandbox / NeuralFoil 的关系

本项目不是重写 AeroSandbox 或 NeuralFoil，而是在其上方增加 AFC 扩展层。

### AeroSandbox 负责的部分

物理/工程建模能力：

- `Airfoil`、`KulfanAirfoil`、`Wing`、`Airplane` 等几何对象。
- `OperatingPoint` 飞行工况。
- `LiftingLine`、`NonlinearLiftingLine`、`VortexLatticeMethod` 等三维低阶气动模型。
- `aerosandbox.numpy`，兼容 NumPy 和 CasADi 自动微分。
- `asb.Opti`，用于设计变量、约束和目标函数优化。

### NeuralFoil 负责的部分

机器学习二维剖面基线：

- 从 Kulfan 参数、`alpha`、`Re`、转捩参数等输入预测二维翼型气动。
- 输出 `CL`、`CD`、`CM`、转捩位置、边界层特征和 `analysis_confidence`。
- 使用轻量 MLP 权重文件，适合嵌入优化循环。

### AFC 扩展层负责的部分

新增工程能力：

- 在二维剖面模型中加入 AFC 输入。
- 在三维有限翼中支持展向 `C_mu`、`x_jet`、`theta_jet` 分布。
- 提供 AFC 数据 schema、训练 skeleton、推理器、验证脚本、优化示例和主动学习任务清单生成器。

## 2.1 统一接口与配置规范

当前阶段的原则是 **配置层独立、现有求解逻辑不大改**。AFC 项目统一使用 `C_mu`、`x_jet`、`theta_jet` 作为主动流动控制输入；二维剖面量在需求和文档中写作 `cl/cd/cm`，但当前代码为了兼容 AeroSandbox 和 NeuralFoil，仍用 `CL/CD/CM` 存储二维剖面结果。三维总量统一写作 `CL/CD/CM`。

### 2.1.1 配置文件

示例配置位于：

```text
configs/afc/dataset_config.example.yaml
configs/afc/model_config.example.yaml
configs/afc/aircraft_config.example.yaml
configs/afc/optimization_config.example.yaml
configs/afc/validation_config.example.yaml
```

配置读取工具位于：

```text
aerosandbox/aerodynamics/afc_config.py
```

主要函数：

```python
read_afc_config(...)
write_afc_config(...)
merge_afc_configs(...)
get_afc_config_value(...)
require_afc_config_fields(...)
```

配置优先级建议如下，越靠前优先级越高：

1. 函数调用时显式传入的参数，例如 `model_size="large"`。
2. CLI 或脚本中的运行时 override。
3. `configs/afc/*.yaml` 或 `.json` 配置文件。
4. 模块默认值，例如训练脚本中的 `TrainingConfig`。
5. 仅用于测试或接口占位的 fallback/default。

YAML 支持策略：

- 如果环境安装了 PyYAML，则使用 `yaml.safe_load()`。
- 如果没有 PyYAML，则使用项目内置轻量 YAML 解析器，支持当前示例配置所用的嵌套 mapping 和 JSON 风格 inline list/dict。
- 示例配置避免复杂 YAML 语法，以保持轻依赖。

### 2.1.2 当前 2D 字段

二维输入字段：

```text
kulfan_upper_weights_0..7  [-]             上表面 Kulfan/CST 权重
kulfan_lower_weights_0..7  [-]             下表面 Kulfan/CST 权重
leading_edge_weight        [-]             前缘修正权重
TE_thickness               chord_fraction  后缘厚度
alpha                      deg             攻角
Re                         [-]             雷诺数
Mach                       [-]             马赫数
n_crit                     [-]             NeuralFoil/XFoil 转捩参数
xtr_upper                  chord_fraction  上表面强制转捩位置
xtr_lower                  chord_fraction  下表面强制转捩位置
C_mu                       [-]             射流动量系数
x_jet                      chord_fraction  喷流位置
theta_jet                  deg             喷流角
```

二维输出字段：

```text
CL                         [-]  当前代码字段；二维剖面升力 cl
CD                         [-]  当前代码字段；二维剖面阻力 cd
CM                         [-]  当前代码字段；二维剖面俯仰力矩 cm
Top_Xtr                    [-]  上表面转捩位置
Bot_Xtr                    [-]  下表面转捩位置
analysis_confidence        [-]  推理可信度或适用性提示
```

二维数据集附加元数据：

```text
converged                  bool  数据源是否收敛或可用
source                     str   数据来源，例如 SU2/RANS/experiment/synthetic
confidence_label           [-]   训练用可信度标签
```

### 2.1.3 当前 3D 字段

三维分析输入：

```text
airplane                   asb.Airplane         机翼/飞机几何
op_point                   asb.OperatingPoint   飞行工况
spanwise_resolution        int                  展向离散
chordwise_resolution       int                  弦向离散，仅 VLM/CamberedVLM 使用
C_mu                       scalar/array/callable 展向 AFC 强度分布
x_jet                      scalar/array/callable 展向喷流位置分布
theta_jet                  scalar/array/callable 展向喷流角分布
model_size                 str                  2D 剖面模型尺寸
include_360_deg_effects    bool                 是否启用 NeuralFoil 后失速扩展
```

三维总量输出：

```text
CL                         [-]  总升力系数
CD                         [-]  总阻力系数
CDi                        [-]  诱导阻力
CDp                        [-]  剖面阻力
CM / Cm                    [-]  俯仰力矩系数；CM 是当前 AFC 后处理推荐字段
CY                         [-]  侧力系数，部分 3D 求解器输出
Cl                         [-]  滚转力矩系数，注意不要与二维 cl 混淆
Cn                         [-]  偏航力矩系数
```

三维展向输出：

```text
spanwise_y                 m    展向站位
local_alpha                deg  局部有效攻角
local_Re                   [-]  局部雷诺数
local_C_mu                 [-]  局部 AFC 强度
section_CL                 [-]  当前代码字段；二维截面 cl
section_CD                 [-]  当前代码字段；二维截面 cd
section_CM                 [-]  当前代码字段；二维截面 cm
analysis_confidence        [-]  截面可信度分布
root_bending_moment        N*m  翼根弯矩估计
spanwise_load              N/m  展向载荷
local_stall_indicator      [-]  局部失速风险 proxy
afc_power_proxy            [-]  integral(C_mu * chord dy) 类 proxy，不是正式 SWaP 功率
```

### 2.1.4 Training、Validation 和 Examples 字段

训练输入：

```text
data_path                  CSV/Parquet AFC 2D 数据集
model_size                 small/medium/large
baseline_model_size        原 NeuralFoil baseline 尺寸
batch_size                 训练 batch size
epochs                     最大训练轮数
learning_rate              学习率
optimizer                  adamw 或 radam
loss_weights               CL/logCD/CM/transition/confidence 损失权重
```

训练输出：

```text
checkpoint_latest.pt       最新 PyTorch checkpoint
checkpoint_best.pt         最优验证集 checkpoint
history.json               训练/验证/测试 loss 历史
exported_weights/nn-*.npz  可由 aerosandbox.numpy 推理器加载的权重
```

验证输入：

```text
external_2d_dataset        可选外部 CSV/Parquet AFC 数据集
synthetic_2d_cases         无外部数据时的轻量 synthetic 样本数
include_avl                是否尝试 AVL baseline
```

验证输出：

```text
afc_validation_report.json
afc_validation_report.md
checks[].passed
checks[].metrics
```

当前 examples 的输入输出约定：

```text
afc_neuralfoil_placeholder.py          输入单个翼型/工况/AFC；输出 2D 气动结果
afc_dataset_pipeline.py                输入 synthetic 生成参数；输出 CSV/Parquet/统计
afc_lifting_line_comparison.py         输入矩形/梯形翼和 C_mu；输出 3D 气动对比
afc_wing_distribution_optimization.py  输入设计变量和约束；输出优化前后对比和图
afc_3d_postprocessing.py               输入 3D 分析结果；输出后处理图
afc_validation_report.py               输入验证配置/外部数据路径；输出 JSON/Markdown 报告
```

## 3. 二维 AFC 代理模型原理

二维接口位于：

```text
aerosandbox/geometry/airfoil/airfoil.py
```

主要 API：

```python
airfoil.get_aero_from_afc_neuralfoil(
    alpha,
    Re,
    mach=0.0,
    C_mu=0.0,
    x_jet=0.1,
    theta_jet=30.0,
    model_size="large",
    control_surfaces=None,
    include_360_deg_effects=True,
)
```

输入含义：

- `alpha`：攻角，单位为度。
- `Re`：雷诺数。
- `mach`：马赫数。
- `C_mu`：喷流动量系数，表示 AFC 强度。
- `x_jet`：喷流位置，按弦长归一化。
- `theta_jet`：喷流角，单位为度。
- `model_size`：基线 NeuralFoil 或 AFC-NeuralFoil 模型尺寸。
- `control_surfaces`：沿用 AeroSandbox 的舵面修正。
- `include_360_deg_effects`：是否启用原 NeuralFoil 的 360 度后失速扩展。

输出字段：

```text
CL
CD
CM
Top_Xtr
Bot_Xtr
analysis_confidence
```

当前已实现的模型逻辑：

1. 先调用原始 `get_aero_from_neuralfoil()` 得到无 AFC 基线。
2. 当 `C_mu = 0` 时，`CL`、`CD`、`CM` 必须严格退化为原始 NeuralFoil 结果。
3. 当 `C_mu > 0` 时，使用平滑经验函数修正 `CL`、`CD`、`CM`。
4. 修正函数使用 `aerosandbox.numpy` 编写，因此可以进入 `asb.Opti`。
5. `analysis_confidence` 会随着 AFC placeholder 修正强度增加而降低。

需要明确区分：

- 原 NeuralFoil 是机器学习模型。
- 当前 AFC 修正函数是 **经验 placeholder**，不是训练后的真实 AFC 机器学习模型。
- 当前接口的主要价值是稳定 API、打通优化和三维耦合流程。

## 4. AFC-NeuralFoil MLP 推理模块

模块位置：

```text
aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py
```

主要函数：

```python
get_aero_from_afc_kulfan_parameters(...)
get_aero_from_afc_airfoil(...)
make_afc_neuralfoil_feature_matrix(...)
load_afc_neuralfoil_weights(...)
generate_dummy_afc_neuralfoil_weights(...)
```

特征编码：

- Kulfan 上表面权重：8 个。
- Kulfan 下表面权重：8 个。
- `leading_edge_weight`。
- `TE_thickness`。
- `alpha`：使用 `sin(alpha)`、`cos(alpha)` 和二倍角编码。
- `Re`：使用 log 编码。
- `Mach`、`n_crit`、`xtr_upper`、`xtr_lower`。
- AFC 输入：`C_mu`、`x_jet`、`sin(theta_jet)`、`cos(theta_jet)`。

推理实现：

- MLP 前向传播使用 `aerosandbox.numpy`。
- 激活函数使用 Swish。
- 权重从 `.npz` 加载，键名沿用 NeuralFoil 风格，例如 `net.0.weight`、`net.0.bias`。
- `CD` 使用 log 空间解码，避免直接预测负阻力。
- 可配置 `C_mu = 0` 时回退到原 NeuralFoil 或基线修正。
- 第一版可信度/OOD 模块位于 `aerosandbox/aerodynamics/aero_2D/afc_confidence.py`，输出 `ood_score`、`stall_risk` 和 `confidence_reason`。
- `analysis_confidence` 会叠加输入域风险、训练集距离风险和失速 proxy；这是启发式适用性提示，不是严格不确定性量化。

当前状态：

- 推理器已实现。
- dummy 权重生成脚本已实现。
- dummy 权重只用于接口测试，不代表真实 AFC 气动。

训练集距离统计文件可由训练脚本自动生成：

```text
runs/afc_neuralfoil/.../afc_confidence_training_statistics.json
```

也可以通过 `make_afc_training_statistics(...)` / `write_afc_training_statistics(...)` 从任意符合 AFC 2D schema 的训练数据生成。推理时传入 `training_statistics` 或 `training_statistics_path` 后，模型会计算归一化 feature-space 距离并合并到 `ood_score`。
- 真实模型效果需要由训练流程导出的权重决定。

## 5. 三维有限翼耦合模型原理

三维 AFC 模块位于：

```text
aerosandbox/aerodynamics/aero_3D/
```

主要类：

```python
AFCNeuralFoilLiftingLine(...)
AFCNeuralFoilNonlinearLiftingLine(...)
NeuralFoilCamberedVLM(...)
```

### 5.1 显式 LiftingLine 版本

模块：

```text
aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py
```

原理：

- 复用 AeroSandbox `LiftingLine` 的机翼几何、展向离散、诱导速度和近场力积分。
- 每个展向站位调用 `get_aero_from_afc_neuralfoil()`。
- 支持标量或展向分布形式的 `C_mu`、`x_jet`、`theta_jet`。

输出包括：

```text
CL
CD
CDi
CDp
CM
spanwise_y
local_alpha
local_Re
local_C_mu
section_CL
section_CD
section_CM
analysis_confidence
```

物理/ML 边界：

- 诱导速度、有限翼几何和积分来自低阶物理模型。
- 剖面气动来自 NeuralFoil/AFC-NeuralFoil 机器学习或 placeholder 模型。

### 5.2 非线性残差耦合版本

模块：

```text
aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py
```

核心残差：

```text
R_i = CL_from_gamma_i - CL_2D(alpha_eff_i, Re_i, Mach_i, C_mu_i)
```

原理：

- 使用 `asb.Opti` 将三维环量 `gamma` 定义为隐式变量。
- 根据 `gamma` 构建诱导速度和局部有效攻角 `alpha_eff`。
- 每个截面调用 AFC 2D 剖面模型。
- 施加 `R_i = 0`，让三维环量与二维剖面升力一致。
- `solve=True` 时内部求解；`solve=False` 时暴露残差给外部优化器。

当前状态：

- 已优先实现 `CL` 匹配。
- 可输出展向残差、局部工况和剖面气动。
- 更完整的三维粘性、分离和强非线性耦合仍需后续验证。

### 5.3 Cambered VLM 版本

模块：

```text
aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py
```

目标：

- 通过局部虚拟修正角 `delta_alpha_i` 和 `delta_camber_i`，让 VLM 面元边界条件匹配 AFC-NeuralFoil 的剖面 `CL` 和 `CM`。

实现原则：

- 不旋转整个涡格。
- 局部修改展向截面或面板的边界条件法向量。
- 支持欠松弛或隐式残差方式。
- 输出 convergence history、`residual_CL`、`residual_CM`。

当前状态：

- 已实现第一版 CL 匹配和 CM 匹配接口。
- 该方法属于低阶三维物理模型与机器学习剖面模型的耦合方法。
- 对 AFC 强非线性流动的可信度仍需要 CFD/实验标定。

## 6. 三维后处理

模块：

```text
aerosandbox/aerodynamics/aero_3D/afc_postprocessing.py
```

标准输出 schema：

```text
CL
CD
CDi
CDp
CM
root_bending_moment
spanwise_y
spanwise_width
section_chord
section_CL
section_CD
section_CM
section_lift
local_stall_indicator
analysis_confidence
afc_power_proxy
```

其中：

- `CDi`：诱导阻力，来自三维模型的尾迹/环量/近场或 Trefftz 近似。
- `CDp`：剖面阻力，沿展向积分 AFC-NeuralFoil section `CD`。
- `root_bending_moment`：翼根弯矩估计。
- `local_stall_indicator`：局部失速风险指标。
- `afc_power_proxy`：`integral(C_mu * chord dy)`，是 AFC 功耗 proxy，不是真实压缩机或执行机构功耗。

可视化函数：

```python
plot_spanwise_results(...)
```

## 7. 数据格式

二维 AFC 数据集模块：

```text
aerosandbox/aerodynamics/aero_2D/afc_dataset.py
```

支持格式：

- `.parquet`：推荐用于大规模训练数据。
- `.csv`：适合调试、小样本和跨软件交换。

必需字段：

```text
kulfan_upper_weights_0..7
kulfan_lower_weights_0..7
leading_edge_weight
TE_thickness
alpha
Re
Mach
n_crit
xtr_upper
xtr_lower
C_mu
x_jet
theta_jet
CL
CD
CM
Top_Xtr
Bot_Xtr
converged
source
confidence_label
```

字段解释：

- Kulfan 和几何字段描述翼型。
- `alpha`、`Re`、`Mach`、`n_crit`、`xtr_upper`、`xtr_lower` 描述二维流动条件。
- `C_mu`、`x_jet`、`theta_jet` 描述 AFC 激励。
- `CL`、`CD`、`CM`、`Top_Xtr`、`Bot_Xtr` 是监督学习标签。
- `converged` 标记 CFD 或实验处理是否可用。
- `source` 标记来源，例如 `SU2`、`RANS`、`XFoil_modified`、`experiment`。
- `confidence_label` 可来自收敛质量、网格质量、实验重复性或专家标注。

主要工具：

```python
read_afc_dataset(...)
write_afc_dataset(...)
validate_afc_dataframe(...)
split_afc_dataset(...)
afc_dataset_statistics(...)
make_fake_afc_dataset(...)
```

当前状态：

- schema、读写、校验、split、统计已实现。
- fake 数据生成已实现，仅用于工程链路测试。
- 真实 CFD/实验数据不应直接放入源码仓库；大文件应通过外部路径传给训练或验证脚本。

## 8. 主动学习数据闭环

模块：

```text
aerosandbox/aerodynamics/aero_2D/afc_active_learning.py
```

目标：

- 根据当前训练数据、当前模型、候选设计空间和 confidence/误差 proxy，推荐下一批 CFD 样本点。
- 输出 CSV/Parquet 任务清单。
- 不运行 SU2、RANS 或任何外部 CFD 求解器。

已实现能力：

- LHS 初始采样。
- 基于 `analysis_confidence` 的补样。
- 基于有限差分的高梯度区域补样。
- 围绕优化最优解的局部加密。
- 基于攻角、`CL` 和转捩位置的失速附近 proxy。
- 基于已有训练数据距离的 novelty score。

主要 API：

```python
latin_hypercube_sample_afc_design_space(...)
local_refinement_sample_afc_design_space(...)
evaluate_afc_active_learning_candidates(...)
estimate_afc_candidate_gradient_scores(...)
recommend_afc_cfd_samples(...)
write_afc_active_learning_tasks(...)
read_afc_active_learning_tasks(...)
```

任务清单包含：

```text
sample_id
reason
source
全部 AFC 输入字段
predicted_CL
predicted_CD
predicted_CM
predicted_Top_Xtr
predicted_Bot_Xtr
predicted_analysis_confidence
nearest_training_distance
score_uncertainty
score_gradient
score_optimization
score_stall
score_novelty
priority_score
```

`reason` 可能包括：

- `高不确定性`
- `高梯度`
- `优化关键区域`
- `失速附近`
- `覆盖稀疏区域`
- `LHS初始采样`

## 9. 训练流程

训练模块：

```text
aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/
```

主要文件：

```text
train_afc_neuralfoil.py
export_weights.py
config_example.json
README.md
```

训练流程：

1. 读取 CSV/Parquet AFC 数据集。
2. 校验 schema 和数值范围。
3. 划分 train/val/test。
4. 构建 PyTorch `Dataset` 和 `DataLoader`。
5. 使用与推理器一致的输入编码。
6. 对输入和输出做归一化。
7. 使用 Swish MLP。
8. 使用以下损失：
   - Huber(`CL`)
   - Huber(`logCD`)
   - Huber(`CM`)
   - Huber(`Top_Xtr`, `Bot_Xtr`)
   - BCE(`confidence_label`)。
9. 使用 AdamW 或 RAdam。
10. 支持 early stopping、学习率衰减和 checkpoint。
11. 导出 `.npz` 权重供 `aerosandbox.numpy` 推理器使用。

机器学习边界：

- 训练脚本本身不是物理模型。
- 它只学习数据中呈现的 AFC 气动映射。
- 数据质量、采样覆盖、CFD 收敛性和实验误差决定模型可信度。

## 10. 推理 API

二维 Airfoil API：

```python
import aerosandbox as asb

airfoil = asb.Airfoil("naca2412")
aero = airfoil.get_aero_from_afc_neuralfoil(
    alpha=5,
    Re=1e6,
    mach=0.1,
    C_mu=0.03,
    x_jet=0.1,
    theta_jet=30,
)
```

Kulfan 级 AFC-NeuralFoil API：

```python
import aerosandbox as asb

airfoil = asb.KulfanAirfoil("naca2412")
aero = asb.get_aero_from_afc_kulfan_parameters(
    kulfan_parameters=airfoil.kulfan_parameters,
    alpha=5,
    Re=1e6,
    Mach=0.1,
    C_mu=0.03,
    x_jet=0.1,
    theta_jet=30,
    model_size="small",
)
```

三维 API：

```python
analysis = asb.AFCNeuralFoilLiftingLine(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=8,
    C_mu=0.03,
)
result = analysis.run()
```

非线性隐式耦合：

```python
result = asb.AFCNeuralFoilNonlinearLiftingLine(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=8,
    C_mu=0.03,
).run(solve=True)
```

Cambered VLM：

```python
result = asb.NeuralFoilCamberedVLM(
    airplane=airplane,
    op_point=op_point,
    spanwise_resolution=8,
    chordwise_resolution=4,
    C_mu=0.03,
).run()
```

## 11. 优化示例

示例：

```text
examples/afc_wing_distribution_optimization.py
```

设计变量包括：

- 翼尖弦长。
- 半展长或展弦比。
- 线性扭转。
- 3 到 5 个展向 `C_mu` 控制段。

典型目标函数：

```text
minimize CD_total + lambda_afc * C_mu_cost
```

典型约束：

```text
CL >= 指定值
0 <= C_mu <= 0.1
翼展 <= 限制
翼根弯矩 <= 限制
min(analysis_confidence) >= 阈值
翼型厚度约束，如有几何变量
```

注意：

- 优化流程已打通。
- 当前 AFC 气动收益来自 placeholder 或 dummy 模型时，只能用于 API 和工作流验证。
- 用于真实设计前必须替换为经过验证的训练权重和验证数据。

## 12. 验证方法

验证模块：

```text
aerosandbox/aerodynamics/afc_validation.py
aerosandbox/aerodynamics/validation/
examples/afc_validation_report.py
```

验证对象：

1. `C_mu = 0` 时 AFC-NeuralFoil 与原 NeuralFoil 一致。
2. 2D surrogate 对测试数据误差可计算。
3. 3D 无 AFC 时与 AeroSandbox `LiftingLine` 和 inviscid `VortexLatticeMethod` 基线对比。
4. 有 AFC 时，`CL` 应随 `C_mu` 合理增加，`CD` 和 `CM` 连续变化。
5. `NeuralFoilCamberedVLM` 输出残差收敛历史。
6. `asb.Opti` 优化问题可收敛。

验证输出：

- JSON 报告。
- Markdown 报告。
- 3D 对比表 CSV。
- 展向载荷 / section coefficient 图。
- 残差收敛图。
- 轻量 benchmark case。
- pytest 单元测试结果。

外部数据策略：

- 大规模 CFD、RANS、风洞或专有实验数据不放入源码仓库。
- 通过路径传给 validation report 或 training script。
- 仓库内只保留小型 benchmark、schema 和接口测试。

## 13. 最小运行命令

以下命令假设已经在项目 Python 环境中安装了依赖。

运行二维 AFC placeholder：

```bash
python examples/afc_neuralfoil_placeholder.py
```

生成并检查 synthetic AFC 数据集：

```bash
python examples/afc_dataset_pipeline.py
```

生成 dummy AFC-NeuralFoil 权重：

```bash
python aerosandbox/aerodynamics/aero_2D/generate_dummy_afc_neuralfoil_weights.py
```

运行训练 skeleton：

```bash
python -m aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil \
  --config aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/config_example.json
```

运行三维 AFC LiftingLine 对比：

```bash
python examples/afc_lifting_line_comparison.py
```

运行 AFC 三维后处理示例：

```bash
python examples/afc_3d_postprocessing.py
```

运行三维 AFC 优化示例：

```bash
python examples/afc_wing_distribution_optimization.py
```

生成验证报告：

```bash
python examples/afc_validation_report.py
```

运行轻量测试：

```bash
python -m pytest \
  aerosandbox/aerodynamics/aero_2D/test_aero_2D/test_afc_neuralfoil.py \
  aerosandbox/aerodynamics/aero_2D/test_aero_2D/test_afc_dataset.py \
  aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_afc_lifting_line.py \
  aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_afc_postprocessing.py
```

## 14. 当前限制

物理和模型限制：

- 当前二维 AFC 修正是 placeholder，不是由真实 AFC CFD/实验数据训练得到。
- dummy `.npz` 权重只用于接口测试，不能用于设计判断。
- synthetic 数据只用于 schema、训练和验证管线测试。
- 3D 耦合模型是低阶有限翼模型，不能替代 RANS/LES 对强分离、复杂射流和三维非定常流动的预测。
- `afc_power_proxy = integral(C_mu * chord dy)` 只是功耗 proxy，不是真实执行机构功耗。
- `analysis_confidence` 是模型适用性提示，不等价于严格不确定性量化。

工程限制：

- 真实训练效果依赖外部 CFD/实验数据。
- 主动学习模块只生成 CFD 任务清单，不调度、不运行、不监控 SU2。
- Parquet 读写需要 `pyarrow` 或 `fastparquet`。
- 训练需要 PyTorch；推理路径尽量保持轻量和可微。

## 15. 后续计划

建议优先级：

1. 接入一批可信二维 AFC CFD 样本，覆盖 `C_mu = 0` 和小 `C_mu` 极限。
2. 校准 `confidence_label` 规则，区分 CFD 未收敛、失速强分离、超出训练域和实验不确定性。
3. 用真实数据训练 AFC-NeuralFoil，小模型先保证 `C_mu = 0` 退化和连续性。
4. 在 validation report 中加入真实测试集误差表和极限工况切片图。
5. 将主动学习任务清单接入外部 SU2/RANS 作业系统。
6. 对三维模型增加 AVL 或高保真 CFD baseline 对比。
7. 完善 AFC 功耗模型，引入质量流量、喷口面积、压比或执行机构效率。
8. 引入 ensemble、dropout 或误差模型，替代当前简单 confidence 补样策略。
9. 对 Cambered VLM 的 `CL`/`CM` 残差收敛进行更多 benchmark。
10. 扩展到多喷口、多段 AFC 和非定常控制参数。
