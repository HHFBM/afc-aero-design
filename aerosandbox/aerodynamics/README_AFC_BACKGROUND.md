# AFC 翼型/机翼代理模型项目背景文档

本文档结合《翼型计算仿真代理模型项目需求说明书》和当前 AeroSandbox + NeuralFoil + AFC 原型代码，说明本项目的背景、目标、技术路线、现有基础、当前边界和后续建设重点。

本文档面向三类读者：

- CFD 工程师：关注物理对象、数据来源、验证口径和模型适用范围。
- 算法工程师：关注输入输出、训练数据、代理模型、可信度和主动学习闭环。
- 平台工程师/项目负责人：关注模块边界、交付路线、验收风险和后续任务拆分。

## 1. 项目为什么要做

本项目面向“流动控制飞机概念设计”场景。传统流程中，翼型、三维机翼和主动流动控制系统往往分散在不同工具链中：

- 二维翼型分析依赖 XFoil、RANS 或实验数据。
- 三维机翼分析依赖 VLM、LiftingLine、AVL、RANS 等工具。
- 主动流动控制 AFC 的系统代价依赖气源、管路、压气机、作动器等系统模型。
- 设计优化需要反复调用上述模型，成本高、流程长、自动化困难。

需求说明书的核心目标不是“单独训练一个神经网络”，而是建立一条能够支持概念设计的完整技术链路：

```text
二维翼型代理模型
    ↓
三维有限翼耦合模型
    ↓
射流系统 SWaP 量化工具
    ↓
气动收益 + 系统代价联合优化
    ↓
主动学习补样 + CFD/实验验证闭环
```

这个链路的工程价值是：在高保真 CFD 或实验数据有限的情况下，用快速、可微、可优化的代理模型支撑大量方案筛选，把少量高价值方案再送回 CFD 或实验验证。

## 2. 项目分析对象

项目实际分析三个层级的对象。

### 2.1 二维翼型

二维翼型是机翼截面。它的主要输入包括：

- 翼型几何：CST/Kulfan 参数。
- 流动条件：`alpha`、`Re`、`Mach`。
- 转捩条件：`n_crit`、`xtr_upper`、`xtr_lower`。
- AFC 控制量：`C_mu`、`x_jet`、`theta_jet`。

二维输出主要是：

- `cl` 或当前代码中的 `CL`：二维截面升力系数。
- `cd` 或当前代码中的 `CD`：二维截面阻力系数。
- `cm` 或当前代码中的 `CM`：二维截面俯仰力矩系数。
- `Top_Xtr`、`Bot_Xtr`：上下表面转捩位置。
- `analysis_confidence`：模型适用性或可信度提示。

注意：当前代码为了兼容 AeroSandbox 和 NeuralFoil 的已有风格，二维输出字段仍使用 `CL/CD/CM`。从需求说明书角度，后续文档和 schema 应明确区分二维 `cl/cd/cm` 与三维 `CL/CD/CM`。

### 2.2 三维有限翼

三维有限翼考虑展向载荷、翼尖涡、下洗、诱导阻力等二维翼型中不存在的效应。它的主要输入包括：

- 机翼几何：展长、弦长、锥度、扭转、后掠、上反角、翼型分布。
- 飞行工况：速度、高度、攻角、姿态。
- 展向 AFC 分布：每个展向站位或控制段的 `C_mu`。

三维输出主要是：

- `CL`：全机翼升力系数。
- `CD`：总阻力系数。
- `CDi`：诱导阻力。
- `CDp`：剖面阻力。
- `CM`：俯仰力矩系数。
- 展向载荷、局部攻角、局部雷诺数、局部剖面气动。
- 翼根弯矩、失速风险、confidence 分布。

### 2.3 射流增升系统

主动流动控制不是免费的。为了获得某个 `C_mu`，系统需要提供质量流量、压力、功率、管路、阀门、作动器和安装空间。

需求说明书要求建立 SWaP 工具：

- Size：尺寸/体积。
- Weight：重量。
- Power：功率。

当前代码中只有 `afc_power_proxy = integral(C_mu * chord dy)` 这样的简化 proxy。它只能用于早期优化示例，不能替代正式射流系统模型。

## 3. AeroSandbox、NeuralFoil 与本项目的关系

当前项目不是从零写一套 CFD 软件，而是在 AeroSandbox 和 NeuralFoil 之上做二次开发。

### 3.1 AeroSandbox 的作用

AeroSandbox 是可微分航空设计与优化框架。它在本项目中承担：

- 几何对象：`Airfoil`、`KulfanAirfoil`、`Wing`、`Airplane`。
- 飞行工况：`OperatingPoint`。
- 低阶三维气动：`LiftingLine`、`NonlinearLiftingLine`、`VortexLatticeMethod`。
- 可微计算：`aerosandbox.numpy`。
- 优化器接口：`asb.Opti`。

在 CFD 工作流中，AeroSandbox 的定位是快速概念设计和优化前端，不替代高保真 RANS/LES。

### 3.2 NeuralFoil 的作用

NeuralFoil 是二维翼型气动代理模型。它输入翼型几何和流动条件，快速输出翼型气动系数和部分边界层信息。

在本项目中，NeuralFoil 是无 AFC 情况下的二维基线能力。AFC 模型的一个基本要求是：

```text
C_mu = 0 时，AFC-NeuralFoil 应退化到原 NeuralFoil 或与其保持一致。
```

### 3.3 AFC 扩展层的作用

当前实现的 AFC 扩展层负责：

- 在二维翼型接口中加入 `C_mu`、`x_jet`、`theta_jet`。
- 提供 AFC-NeuralFoil 风格 MLP 推理模块。
- 提供训练数据 schema 和训练脚本。
- 将二维 AFC 剖面气动耦合到三维有限翼模型。
- 提供后处理、优化示例、验证脚本和主动学习任务清单生成器。

可以把三者关系理解为：

```text
AeroSandbox = 飞机/机翼建模、低阶气动、优化框架
NeuralFoil  = 快速二维翼型气动代理模型
AFC 项目    = 在两者上增加主动流动控制建模、训练、三维耦合与优化闭环
```

## 4. 当前代码已经实现了什么

当前仓库已经实现了一个可运行的 AFC 原型链路。

### 4.1 二维接口

位置：

```text
aerosandbox/geometry/airfoil/airfoil.py
```

新增接口：

```python
airfoil.get_aero_from_afc_neuralfoil(...)
```

当前行为：

- 先调用原始 `get_aero_from_neuralfoil()`。
- `C_mu = 0` 时退化为 NeuralFoil。
- `C_mu > 0` 时使用平滑经验函数修正 `CL/CD/CM`。
- 该经验修正是 placeholder，不是最终真实 AFC 模型。

### 4.2 AFC-NeuralFoil 推理器

位置：

```text
aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py
```

能力：

- 使用 Kulfan 参数、`alpha`、`Re`、`Mach`、`C_mu`、`x_jet`、`theta_jet` 等特征。
- 使用 `aerosandbox.numpy` 前向传播，兼容自动微分。
- 从 `.npz` 加载权重。
- 支持 `small`、`medium`、`large` 模型尺寸。
- 支持 dummy 权重和训练权重。

### 4.3 数据集与训练

位置：

```text
aerosandbox/aerodynamics/aero_2D/afc_dataset.py
aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/
```

能力：

- 定义 AFC 2D 数据 schema。
- 支持 CSV/Parquet。
- 支持校验、范围检查、train/val/test split、统计报告。
- 使用 PyTorch 训练 MLP。
- 支持 Huber/BCE 损失、AdamW/RAdam、early stopping、checkpoint。
- 可导出 `.npz` 权重供推理器使用。

### 4.4 公开数据训练尝试

当前已从 Flow Control Lab 公开资源下载并转换 NACA0018 quasi-steady pressure data。

已生成：

```text
external_data/flowcontrollab/converted/fcl_naca0018_quasi_steady_afc_schema_with_baseline.csv
```

已训练：

```text
runs/afc_neuralfoil/fcl_naca0018_quasi_steady_large/exported_weights/nn-large.npz
```

该训练结果说明真实公开数据可以进入当前工程链路。但也暴露出限制：

- 数据主要是 NACA0018 单翼型。
- `x_jet`、`theta_jet` 未逐样本给出，只能先固定为假设值。
- `Top_Xtr/Bot_Xtr` 没有真实标签。
- 当前误差还未达到需求书建议验收水平。

### 4.5 三维模型

位置：

```text
aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py
aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py
aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py
```

已实现：

- 显式 AFC LiftingLine。
- 非线性残差耦合 LiftingLine。
- CamberedVLM 第一版。
- 展向 `C_mu` 分布。
- 局部剖面调用 AFC 2D 模型。
- 输出三维气动力和展向分布。

### 4.6 后处理、验证和示例

已实现：

- 3D 后处理 schema。
- 翼根弯矩。
- 局部失速指标。
- analysis confidence 分布。
- AFC 功耗 proxy。
- 优化示例。
- pytest 轻量测试。
- validation report 骨架。
- 主动学习推荐 CFD 样本点模块。

## 5. 当前实现与需求说明书的差距

需求说明书要求四个核心子系统：

1. 二维翼型代理模型子系统。
2. 三维机翼气动力子系统。
3. 射流系统量化设计子系统。
4. 集成优化与验证子系统。

当前状态可以概括为：

```text
二维代理模型：已有原型，需真实数据覆盖、精度提升、版本管理
三维气动模型：已有原型，需正式验证、收敛诊断、输出补全
射流系统工具：尚未正式实现，是最大缺口
集成优化验证：已有气动优化示例，缺少真实 SWaP 耦合和交付级报告
```

### 5.1 二维模型差距

仍需完成：

- 扩展训练数据覆盖，多翼型、多 `x_jet`、多 `theta_jet`、多 `C_mu`。
- 建立 SU2/RANS 或扩展 XFoil 数据生成管线。
- 建立 dataset manifest 和模型版本管理。
- 进一步校准 `analysis_confidence`。
- 将二维命名规范从代码兼容层面和文档层面区分清楚。

### 5.2 三维模型差距

仍需完成：

- 与 AVL、AeroSandbox 原生 VLM、RANS 或风洞数据做正式验证。
- 补齐 `CY`、滚转力矩 `Cl`、偏航力矩 `Cn` 或明确不支持。
- 强化收敛状态、失败原因、残差历史。
- 实现局部二维结果查表/插值加速。

### 5.3 SWaP 系统差距

这是当前最大缺口。

需求说明书要求：

- 节点-组件拓扑。
- 气源、管路、阀门、压气机、集气腔、作动器。
- 设计点和偏离设计点模式。
- 从目标 `C_mu` 反求质量流量、压力、功率。
- 输出系统尺寸、重量、功率和分组件代价。

当前只实现了 AFC power proxy，不能满足 SWaP 工具验收。

### 5.4 集成优化差距

当前已有：

```text
minimize CD + lambda * C_mu_cost
```

但需求目标应逐步升级为：

```text
minimize CD_total
       + lambda_power * required_power
       + lambda_weight * system_weight
       + lambda_volume * system_volume
```

也就是说，后续优化必须把射流系统代价纳入目标函数和约束。

## 6. 物理模型、机器学习模型和 placeholder 的边界

为了避免误用，需要明确三类东西。

### 6.1 物理/低阶气动模型

包括：

- AeroSandbox LiftingLine。
- AeroSandbox VLM。
- 三维诱导速度、环量、载荷积分。

这些是低阶物理模型，适合概念设计，但不能替代 RANS/LES。

### 6.2 机器学习模型

包括：

- NeuralFoil。
- AFC-NeuralFoil MLP。

它们通过数据学习二维翼型气动映射。推理快、可微、适合优化，但可信度取决于训练数据覆盖和验证质量。

### 6.3 Placeholder

当前仍属于 placeholder 的部分：

- `Airfoil.get_aero_from_afc_neuralfoil()` 中的默认平滑经验 AFC 修正。
- dummy AFC-NeuralFoil 权重。
- synthetic 数据集。
- `Top_Xtr/Bot_Xtr` 在部分公开数据转换中的占位值。
- 3D AFC power proxy。
- 还未实现的正式 SWaP 工具。

这些部分可以用于接口验证和流程打通，不能直接作为工程定型依据。

## 7. 数据路线

项目数据应分成四类。

### 7.1 基线数据

用于保证 `C_mu = 0` 时模型退化正确。

来源：

- NeuralFoil。
- XFoil。
- RANS。
- 公开无 AFC 翼型实验数据。

### 7.2 AFC 训练数据

用于训练 AFC-NeuralFoil。

来源：

- SU2/RANS 批量计算。
- 改造 XFoil 或半经验模型。
- Flow Control Lab NACA0018 等公开实验数据。
- 自有风洞实验数据。

### 7.3 三维验证数据

用于验证 3D 有限翼模型。

来源：

- AVL/VLM baseline。
- AeroSandbox 原生求解器。
- RANS/风洞数据。
- 文献 benchmark。

### 7.4 系统代价数据

用于校准 SWaP 工具。

来源：

- 压气机/阀门/管路经验公式。
- 供应商参数。
- 文献经验系数。
- 领域专家给定的设计规则。

## 8. 建议建设阶段

结合当前代码状态和需求说明书，建议后续按以下阶段推进。

### M1：接口与配置规范冻结

目标：

- 统一字段、单位、二维/三维命名。
- 建立配置文件规范。
- 明确结果 schema。

交付：

- `configs/afc/*.yaml` 示例。
- 配置读取工具。
- 字段与单位说明。

### M2：二维代理模型工程化

目标：

- 固化数据 manifest。
- 固化公开数据转换流程。
- 训练脚本输出 model card、误差图和结果摘要。

交付：

- 公开数据转换脚本。
- 训练报告。
- 模型版本说明。

### M3：三维模型验证与稳健性

目标：

- 对 AFC LiftingLine、NonlinearLiftingLine、CamberedVLM 做 benchmark。
- 输出收敛历史和失败原因。

交付：

- 3D validation report。
- 展向载荷图。
- 残差收敛图。

### M4：射流系统 SWaP 原型

目标：

- 实现节点-组件气路系统。
- 支持从目标 `C_mu` 估算质量流量、功率、重量和体积。

交付：

- `aerosandbox/afc_system/` 模块。
- 系统配置示例。
- SWaP 趋势验证。

### M5：端到端联合优化

目标：

- 将 2D AFC、3D 机翼和 SWaP 系统工具接入同一优化流程。

交付：

- 一个完整概念设计 example。
- 优化前后对比图。
- 气动收益和系统代价权衡表。

### M6：验收报告与交付整理

目标：

- 形成可复现、可运行、可检查的交付包。

交付：

- 用户手册。
- 验证报告。
- 模型权重说明。
- 数据版本清单。
- 轻量 CI 测试。

## 9. 当前最关键的工程判断

当前项目已经证明：

- AeroSandbox 可以作为三维几何、低阶气动和优化框架。
- NeuralFoil 风格的 AFC 扩展接口可以打通。
- AFC 数据 schema、训练、推理、导出、3D 调用、优化示例可以形成闭环。
- 公开实验数据可以被转换并用于训练 large 模型。

但当前项目还不能被视为需求书意义上的完整平台，因为：

- 缺少正式 SWaP 系统工具。
- 真实训练数据覆盖不足。
- 2D/3D 精度未达到验收级。
- 端到端优化缺少真实系统代价。
- 数据、模型、配置、验证版本管理还需要工程化。

因此，下一阶段最重要的不是继续堆更多示例，而是完成三件事：

1. 冻结统一配置和字段规范。
2. 建立可复现的数据/训练/验证链路。
3. 实现射流系统 SWaP 原型，并把它接入优化。

## 10. 一句话总结

本项目的本质是把高成本 CFD/实验知识压缩成可微、快速、可优化的代理模型系统，用于主动流动控制机翼的概念设计。

当前仓库已经完成了 AFC 气动原型链路，但距离需求说明书定义的完整平台，还需要补齐真实数据工程、正式验证、SWaP 系统工具和端到端联合优化交付。
