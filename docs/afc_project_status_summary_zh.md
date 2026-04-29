# AFC 项目阶段成果汇报摘要

## 1. 一句话结论

当前项目已经完成了一个 **AFC 概念设计工程原型闭环**：

- 能用 2D AFC 代理模型快速评估射流控制对翼型气动的影响
- 能把 2D 结果耦合到 3D 有限翼分析
- 能估算实现目标射流控制所需的系统功率、重量和体积代价
- 能把气动收益和系统代价放到同一个优化问题里联合权衡
- 能生成主动学习 CFD 任务清单和正式验证报告

当前定位是：

**“可运行、可验证、可扩展的概念设计平台原型”**，不是已经完成高保真外部标定的生产级最终模型。

---

## 2. 这个项目到底在解决什么问题

主动流动控制（AFC）的核心难点，不是单独算一个翼型，而是要同时回答 4 个问题：

1. **喷流开了以后，局部气动会怎么变？**
2. **把这种变化放到整片机翼上，总升阻和力矩怎么变？**
3. **为了打出这个喷流强度，需要付出多少系统代价？**
4. **从整机角度看，这件事到底值不值得？**

本项目就是把这 4 个问题接成一个统一流程，而不是分别靠 CFD、经验估算和人工反复试参数。

---

## 3. 目前已经完成的模块

### 3.1 二维 AFC 代理模型

已完成：

- 新增 AFC 2D 输入接口
- 支持输入：
  - `alpha`
  - `Re`
  - `Mach`
  - `C_mu`
  - `x_jet`
  - `theta_jet`
  - Kulfan 翼型参数
- 支持输出：
  - `CL / CD / CM`
  - `Top_Xtr / Bot_Xtr`
  - `analysis_confidence`
  - `ood_score`
  - `stall_risk`
  - `confidence_reason`

对应文件：

- [aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py)
- [aerosandbox/aerodynamics/aero_2D/afc_confidence.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/aero_2D/afc_confidence.py)

已完成第一轮训练，当前一组可用权重位于：

- [runs/afc_neuralfoil/retrain_20260423_fcl_medium/exported_weights/nn-medium.npz](/Users/jason/Documents/AeroSandbox-master/runs/afc_neuralfoil/retrain_20260423_fcl_medium/exported_weights/nn-medium.npz)

当前这轮训练的测试误差摘要：

- `CL` MAE: `0.0766`
- `CD` MAE: `0.0366`
- `CM` MAE: `0.0431`

详细文件：

- [runs/afc_neuralfoil/retrain_20260423_fcl_medium/test_error_summary.csv](/Users/jason/Documents/AeroSandbox-master/runs/afc_neuralfoil/retrain_20260423_fcl_medium/test_error_summary.csv)

---

### 3.2 三维 AFC 有限翼分析

已完成三套 3D 路径：

- `AFCNeuralFoilLiftingLine`
- `AFCNeuralFoilNonlinearLiftingLine`
- `NeuralFoilCamberedVLM`

已补齐或统一这些输出字段：

- `CL, CD, CDi, CDp, CM`
- `CY, Cl, Cn`
- `spanwise_y`
- `section_cl/cd/cm`
- `local_alpha, local_Re, local_C_mu`
- `residual_CL, residual_CM`
- `converged`
- `failure_reason`
- `convergence_history`

对应文件：

- [aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py)
- [aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py)
- [aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py)

---

### 3.3 射流系统 SWaP 原型

已完成一个概念设计级别的射流系统代价模型，能估算：

- `mass_flow`
- `required_power`
- `estimated_weight`
- `estimated_volume`
- 分组件压损 / 重量 / 体积

当前系统是节点-部件网络形式，已包含：

- inlet
- compressor
- duct
- bend
- valve
- plenum
- steady jet actuator

对应文件：

- [aerosandbox/afc_system/network.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/afc_system/network.py)
- [aerosandbox/afc_system/components.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/afc_system/components.py)
- [aerosandbox/afc_system/README.md](/Users/jason/Documents/AeroSandbox-master/aerosandbox/afc_system/README.md)

需要明确：

**这部分目前是 placeholder / trend-level 工程模型，用于趋势判断，不是经过压气机图谱或硬件试验标定后的正式系统模型。**

---

### 3.4 端到端概念设计优化

已完成端到端优化 example，打通：

- 2D AFC-NeuralFoil
- 3D 有限翼分析
- SWaP 系统代价
- `asb.Opti`

对应文件：

- [examples/afc_concept_design_optimization.py](/Users/jason/Documents/AeroSandbox-master/examples/afc_concept_design_optimization.py)

当前已生成优化前后结果：

- [examples/generated_afc_concept_design_optimization/comparison_table.csv](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/comparison_table.csv)
- [examples/generated_afc_concept_design_optimization/summary.json](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/summary.json)

当前示例中的前后对比：

- 初始方案：
  - `CL = 0.7201`
  - `CD_total = 0.03131`
  - `required_power = 3134.8 W`
- 优化方案：
  - `CL = 0.8000`
  - `CD_total = 0.02792`
  - `required_power = 109.0 W`

这说明端到端流程已经能做“几何 + 射流分布 + 系统代价”的联合权衡。

---

### 3.5 主动学习 CFD 数据闭环

已完成主动学习任务清单模块，能基于当前训练数据和模型推荐下一批 CFD 样本。

支持：

- `LHS` 初始采样
- `confidence` 补样
- `OOD` 补样
- 高梯度补样
- 优化最优解附近局部加密

输出单独的 CFD task schema，不和训练数据 schema 混用。

对应文件：

- [aerosandbox/aerodynamics/aero_2D/afc_active_learning.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/aero_2D/afc_active_learning.py)
- [examples/afc_active_learning_cfd_tasks.py](/Users/jason/Documents/AeroSandbox-master/examples/afc_active_learning_cfd_tasks.py)

当前已生成 50 个 CFD 推荐点：

- [examples/generated_afc_active_learning_tasks/afc_cfd_tasks.csv](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_active_learning_tasks/afc_cfd_tasks.csv)
- [examples/generated_afc_active_learning_tasks/afc_cfd_tasks.jsonl](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_active_learning_tasks/afc_cfd_tasks.jsonl)

这一步的意义是：

**项目已经不只是“能训练模型”，而是已经能指导下一批 SU2/RANS 该算什么点。**

---

### 3.6 正式验证报告生成器

已把原有 validation 脚本扩展成正式验收报告生成器。

支持输出：

- `report.md`
- `report.json`
- `figures/`

报告内容包含：

- 2D 模型误差
- 2D 极曲线对比
- 3D baseline 对比
- AFC 增升趋势
- SWaP 趋势验证
- 端到端优化前后对比

并记录：

- git commit
- model weights path
- dataset manifest
- config path
- run time
- pass/fail 标准

对应文件：

- [aerosandbox/aerodynamics/afc_validation.py](/Users/jason/Documents/AeroSandbox-master/aerosandbox/aerodynamics/afc_validation.py)
- [examples/afc_validation_report.py](/Users/jason/Documents/AeroSandbox-master/examples/afc_validation_report.py)
- [configs/afc/validation_config.example.yaml](/Users/jason/Documents/AeroSandbox-master/configs/afc/validation_config.example.yaml)

当前已生成一套正式报告：

- [validation_outputs/afc_acceptance_ci/report.md](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/report.md)
- [validation_outputs/afc_acceptance_ci/report.json](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/report.json)
- [validation_outputs/afc_acceptance_ci/figures](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/figures)

---

## 4. 射流问题是怎么解决的

这是汇报里最重要的一部分。

### 4.1 先把“射流控制”参数化

项目没有把“喷流”当成模糊概念，而是明确拆成了 3 个核心控制量：

- `C_mu`：喷流动量系数，表示控制强度
- `x_jet`：喷流在翼型弦向的位置
- `theta_jet`：喷流喷出角度

这样做的意义是：

- 便于训练 2D surrogate
- 便于和 CFD / 实验边界条件对接
- 便于在 3D 机翼上做展向分布设计

也就是说，本项目把“射流问题”先变成一个**可计算、可优化、可采样**的问题。

---

### 4.2 在 2D 截面上先解决“气动效果预测”

射流最直接的作用是改变边界层和局部流动，从而影响：

- 升力
- 阻力
- 力矩
- 转捩和失速风险

项目的做法是：

1. 用 `C_mu / x_jet / theta_jet` + 翼型几何 + 工况做输入  
2. 用 AFC-NeuralFoil 预测 `CL / CD / CM` 等输出  
3. 同时给出 `confidence / OOD / stall_risk`

这样解决的不是“物理机理全部展开求解”，而是先解决：

**在概念设计阶段，如何快速知道这个喷流方案值不值得继续看。**

---

### 4.3 在 3D 机翼上解决“整机效应”

二维效果不等于整片机翼效果。

项目进一步把射流控制沿展向分布化，也就是允许不同翼展位置有不同 `C_mu`。

这样就能回答：

- 喷流应该集中在翼根、翼中还是翼尖？
- 哪一段更值得给控制资源？
- 整体升阻和根弯矩怎么变化？

这一步本质上是在解决：

**“喷流在三维机翼上的分配问题”。**

---

### 4.4 再把“喷流代价”显式算出来

很多 AFC 方案只看增升，却不算系统代价，这是不够的。

本项目把射流问题进一步转成系统需求：

- 为了实现目标 `C_mu`，需要多大质量流量？
- 需要多大功率？
- 管路和压气机会带来多少重量和体积？

因此这个项目不是只回答“喷流能不能增升”，还回答：

**“喷流值不值得，为这个增升要付出多少系统代价。”**

---

### 4.5 最后用优化把“收益”和“代价”同时平衡

真正的射流问题，不是单独把 `C_mu` 调大，而是做权衡：

- 气动更好
- 功率更低
- 系统更轻
- 机翼结构约束还能满足
- 分析结果还必须可信

项目现在已经可以做这样的优化：

```text
minimize CD_total + lambda_power * required_power + lambda_weight * system_weight
```

在这个框架下，喷流问题就从“单点分析问题”变成了：

**“整机概念设计中的多学科协同优化问题”。**

---

## 5. 现在的成果价值

从工程角度看，现在最有价值的不是某一个数字，而是已经形成了下面这个闭环：

```text
射流参数
-> 2D 局部气动收益
-> 3D 机翼整体效果
-> 系统 SWaP 代价
-> 联合优化
-> CFD 主动学习补样
-> 正式验证报告
```

这意味着：

- 已经能做概念设计阶段快速方案筛选
- 已经能指导下一批 CFD 应该算什么点
- 已经能形成标准化验收报告
- 已经具备继续走向生产级的工程骨架

---

## 6. 当前还没有完成的部分

这部分汇报时必须明确，否则结论会过度承诺。

### 已完成的是

- 工程原型闭环
- 模块接口
- 训练/推理/优化/验证/主动学习流程
- placeholder 和真实外部数据接口

### 还没完成的是

- 大规模真实 AFC CFD / 风洞数据标定
- 生产级 2D 精度验证
- 生产级 3D 外部验证
- 经标定的压气机 / 管路 / 执行器系统模型
- 最终交付级用户手册与外部 benchmark 套件

也就是说：

**现在已经是一个“可用来做方案研究和流程验证”的原型平台，但还不是最终工程定型工具。**

---

## 7. 下一阶段建议

下一阶段最关键的工作不是再扩接口，而是补真实数据和外部验证：

1. 引入更多真实 AFC 2D 数据  
2. 用主动学习任务清单驱动下一批 SU2 / RANS 批量计算  
3. 用新 CFD 数据重新训练 2D AFC surrogate  
4. 用外部验证数据替换 synthetic / placeholder 验证段  
5. 标定 SWaP 子模型  
6. 形成正式验收版报告

---

## 8. 汇报时可以直接使用的总结表述

可以直接这样汇报：

> 我们已经完成了一个主动流动控制概念设计平台的第一版工程原型。  
> 它能够把二维射流气动、三维有限翼分析、射流系统功率重量代价和设计优化打通到同一个流程里。  
> 目前系统不仅能预测喷流控制可能带来的气动收益，还能评估这种收益的系统代价，并自动推荐下一批 CFD 样本、生成正式验证报告。  
> 当前版本已经适合做概念设计研究、方案对比和数据闭环建设，但还需要更多真实 CFD / 实验数据支撑，才能进入生产级验证阶段。

---

## 9. 可直接展示的现有成果文件

### 优化结果

- [examples/generated_afc_concept_design_optimization/comparison_table.csv](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/comparison_table.csv)
- [examples/generated_afc_concept_design_optimization/spanwise_c_mu.png](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/spanwise_c_mu.png)
- [examples/generated_afc_concept_design_optimization/spanwise_load.png](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/spanwise_load.png)
- [examples/generated_afc_concept_design_optimization/section_cl_cd.png](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/section_cl_cd.png)
- [examples/generated_afc_concept_design_optimization/swap_breakdown_placeholder.png](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_concept_design_optimization/swap_breakdown_placeholder.png)

### 主动学习 CFD 任务清单

- [examples/generated_afc_active_learning_tasks/afc_cfd_tasks.csv](/Users/jason/Documents/AeroSandbox-master/examples/generated_afc_active_learning_tasks/afc_cfd_tasks.csv)

### 正式验证报告

- [validation_outputs/afc_acceptance_ci/report.md](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/report.md)
- [validation_outputs/afc_acceptance_ci/figures/2d_error_summary.png](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/figures/2d_error_summary.png)
- [validation_outputs/afc_acceptance_ci/figures/2d_polar_comparison.png](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/figures/2d_polar_comparison.png)
- [validation_outputs/afc_acceptance_ci/figures/3d_baseline_comparison.png](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/figures/3d_baseline_comparison.png)
- [validation_outputs/afc_acceptance_ci/figures/3d_afc_gain_trend.png](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/figures/3d_afc_gain_trend.png)
- [validation_outputs/afc_acceptance_ci/figures/swap_trend_validation.png](/Users/jason/Documents/AeroSandbox-master/validation_outputs/afc_acceptance_ci/figures/swap_trend_validation.png)

