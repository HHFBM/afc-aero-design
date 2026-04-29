# AFC-NeuralFoil 二次开发模块说明

本文档说明当前 AeroSandbox 中新增的 AFC 相关模块、接口和开发链路。

AFC 是 **Active Flow Control，主动流动控制**。在本项目中，它用于描述通过吹气、吸气、合成射流、等离子体激励等方式主动改变翼型或机翼流动状态，从而影响升力、阻力、力矩、转捩和失速行为。

当前 AFC 控制量主要包括：

- `C_mu`：喷流动量系数，表示 AFC 强度。
- `x_jet`：喷流位置，按弦长归一化。
- `theta_jet`：喷流角度，单位为度。

当前实现的定位是 **AFC 代理模型和三维有限翼计算的工程骨架**。它已经打通了 2D 接口、数据格式、神经网络推理、训练导出、3D 有限翼模型、优化示例和后处理流程。但默认 AFC 修正和 synthetic 数据都不是经过 CFD 或实验验证的真实物理模型。

## 总体模块结构

```text
aerosandbox/
  geometry/
    airfoil/
      airfoil.py

  aerodynamics/
    README_AFC.md

    aero_2D/
      afc_neuralfoil.py
      afc_dataset.py
      generate_dummy_afc_neuralfoil_weights.py
      afc_neuralfoil_training/
        README.md
        config_example.json
        train_afc_neuralfoil.py
        export_weights.py
      test_aero_2D/
        test_afc_neuralfoil.py
        test_afc_dataset.py
        test_afc_neuralfoil_training.py

    aero_3D/
      afc_lifting_line.py
      afc_nonlinear_lifting_line.py
      afc_cambered_vlm.py
      afc_postprocessing.py
      test_aero_3D/
        test_afc_lifting_line.py
        test_afc_nonlinear_lifting_line.py
        test_afc_cambered_vlm.py
        test_afc_postprocessing.py

    afc_validation.py
    validation/
      README.md
      afc_benchmark_case.json

examples/
  afc_neuralfoil_placeholder.py
  afc_dataset_pipeline.py
  afc_lifting_line_comparison.py
  afc_wing_distribution_optimization.py
  afc_cambered_vlm_comparison.py
  afc_3d_postprocessing.py
  afc_validation_report.py
```

## 1. 二维 AFC 翼型接口

位置：

```text
aerosandbox/geometry/airfoil/airfoil.py
```

新增接口：

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

作用：

- 在原有 `Airfoil.get_aero_from_neuralfoil()` 基础上增加 AFC 输入。
- 当 `C_mu = 0` 时退化为原始 NeuralFoil 结果。
- 当 `C_mu > 0` 时，当前版本使用平滑经验函数修正 `CL`、`CD`、`CM`。
- 使用 `aerosandbox.numpy` 实现，保持对 CasADi 自动微分和 `asb.Opti` 优化的兼容性。

主要输出：

```text
CL
CD
CM
Top_Xtr
Bot_Xtr
analysis_confidence
ood_score
stall_risk
confidence_reason
```

当前状态：

- 已实现可运行接口。
- 当前默认 AFC 修正是 placeholder，不是训练后的真实 AFC 模型。
- `analysis_confidence` 已叠加第一版输入域/OOD/失速风险提示；它是工程适用性提示，不是严格不确定性量化。

## 2. AFC-NeuralFoil 推理模块

位置：

```text
aerosandbox/aerodynamics/aero_2D/afc_neuralfoil.py
```

作用：

- 提供 AFC-NeuralFoil 风格的 MLP 推理器。
- 输入使用 Kulfan 翼型参数和 AFC / 流动条件。
- 使用 `aerosandbox.numpy` 前向传播，保持可微。
- 从 `.npz` 文件加载权重。
- 支持 `small`、`medium`、`large` 等模型尺寸。
- 支持 dummy 权重生成，便于测试完整接口。

主要函数：

```python
get_aero_from_afc_kulfan_parameters(...)
get_aero_from_afc_airfoil(...)
make_afc_neuralfoil_feature_matrix(...)
load_afc_neuralfoil_weights(...)
generate_dummy_afc_neuralfoil_weights(...)
```

输入特征包括：

- Kulfan 上表面权重，8 个。
- Kulfan 下表面权重，8 个。
- `leading_edge_weight`
- `TE_thickness`
- `alpha` 的周期编码。
- `Re` 的 log 编码。
- `Mach`
- `n_crit`
- `xtr_upper`
- `xtr_lower`
- `C_mu`
- `x_jet`
- `theta_jet` 的周期编码。

输出字段：

```text
CL
CD
CM
Top_Xtr
Bot_Xtr
analysis_confidence
ood_score
stall_risk
confidence_reason
```

可信度与 OOD 风险模块：

```text
aerosandbox/aerodynamics/aero_2D/afc_confidence.py
```

第一版机制包括：

- 输入域检查：`alpha`、`Re`、`Mach`、`C_mu`、`x_jet`、`theta_jet` 和 Kulfan 权重。
- 训练集距离风险：从训练数据生成 feature mean/std，推理时计算归一化距离并输出 `ood_score`。
- 失速风险 proxy：基于 `alpha`、section `CL`、`analysis_confidence` 和 `Top_Xtr/Bot_Xtr` 占位字段。
- `confidence_reason` 给出简短原因，例如 `in_distribution`、`input_range:C_mu`、`stall_proxy_high`。

注意：`analysis_confidence`、`ood_score` 和 `stall_risk` 是启发式工程信号，不是 calibrated uncertainty，也不等价于 CFD 误差条。

权重格式：

```text
net.0.weight
net.0.bias
net.2.weight
net.2.bias
...
```

当前状态：

- 推理接口已实现。
- 权重加载逻辑已实现。
- dummy 权重生成已实现。
- 真实物理效果依赖后续训练权重。

## 3. AFC 数据集模块

位置：

```text
aerosandbox/aerodynamics/aero_2D/afc_dataset.py
```

作用：

- 定义 AFC 2D 代理模型统一数据格式。
- 支持从 CSV / Parquet 读取数据。
- 支持数据校验、范围检查、训练集划分、基础统计、manifest、重复样本检查。
- 提供 fake dataset 生成，用于测试工程链路。

主要函数：

```python
read_afc_dataset(...)
write_afc_dataset(...)
validate_afc_dataframe(...)
split_afc_dataset(...)
afc_dataset_statistics(...)
afc_dataset_grouped_statistics(...)
afc_dataset_standard_statistics_tables(...)
make_afc_dataset_manifest(...)
read_afc_dataset_manifest(...)
write_afc_dataset_manifest(...)
check_duplicate_afc_samples(...)
make_fake_afc_dataset(...)
```

数据字段：

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

支持格式：

- `.parquet`：推荐用于真实大规模训练数据。
- `.csv`：适合小数据、调试和跨工具交换。

当前状态：

- schema 已实现。
- 校验、读写、划分、统计、manifest、去重检查已实现。
- fake 数据生成已实现。
- Flow Control Lab NACA0018 公开数据转换脚本已固化，但原始大数据需要通过外部路径传入，不提交到仓库。

公开数据转换示例：

```bash
.venv-aerosandbox/bin/python scripts/afc_convert_flowcontrollab_naca0018.py \
  --input-directory /path/to/FCL_pressure_data \
  --output /path/to/fcl_naca0018_quasi_steady_afc_schema.csv \
  --manifest /path/to/fcl_naca0018_quasi_steady_manifest.json \
  --x-jet 0.10 \
  --theta-jet 30
```

该转换器会在 manifest 中记录关键假设：`x_jet`、`theta_jet`、`Top_Xtr`、`Bot_Xtr`、`n_crit`、`xtr_upper/xtr_lower` 均不是 FCL 文本表逐样本给出的实测量，而是为了进入 AFC-NeuralFoil schema 所需的显式假设。

## 4. PyTorch 训练模块

位置：

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

作用：

- 从 AFC 数据集读取训练数据。
- 构造 PyTorch `Dataset` 和 `DataLoader`。
- 训练 Swish 激活的 MLP 网络。
- 对输入和输出做归一化。
- 使用 Huber loss 拟合 `CL`、`logCD`、`CM`。
- 使用 BCE 拟合 `confidence_label`。
- 支持 AdamW / RAdam。
- 支持 early stopping、学习率衰减、checkpoint。
- 训练结束后导出 `.npz` 权重，供 `aerosandbox.numpy` 推理器使用。

训练目标：

```text
CL
logCD
CM
Top_Xtr
Bot_Xtr
confidence_label
```

当前状态：

- 训练代码骨架已实现。
- checkpoint 和 `.npz` 导出已实现。
- 训练结束会生成 `model_card.md`、`test_error_summary.csv`、`test_predictions.csv` 和 prediction-vs-truth 图。
- 已用 synthetic 数据验证训练和导出链路能跑通。
- 训练效果不代表真实 AFC 物理精度。

用公开数据训练 `large` 模型的典型流程：

```bash
.venv-aerosandbox/bin/python scripts/afc_convert_flowcontrollab_naca0018.py \
  --input-directory /path/to/FCL_pressure_data \
  --output /path/to/fcl_naca0018_quasi_steady_afc_schema.csv \
  --manifest /path/to/fcl_naca0018_quasi_steady_manifest.json

.venv-aerosandbox/bin/python -m aerosandbox.aerodynamics.aero_2D.afc_neuralfoil_training.train_afc_neuralfoil \
  --config aerosandbox/aerodynamics/aero_2D/afc_neuralfoil_training/config_example.json \
  --data /path/to/fcl_naca0018_quasi_steady_afc_schema.csv \
  --output-directory runs/afc_neuralfoil/fcl_naca0018_large \
  --model-size large \
  --epochs 300
```

训练产物中重点查看：

```text
runs/afc_neuralfoil/fcl_naca0018_large/model_card.md
runs/afc_neuralfoil/fcl_naca0018_large/test_error_summary.csv
runs/afc_neuralfoil/fcl_naca0018_large/prediction_vs_truth/
runs/afc_neuralfoil/fcl_naca0018_large/exported_weights/nn-large.npz
```

## 5. 显式 AFC Lifting-Line 模型

位置：

```text
aerosandbox/aerodynamics/aero_3D/afc_lifting_line.py
```

类名：

```python
asb.AFCNeuralFoilLiftingLine(...)
```

作用：

- 基于 AeroSandbox 现有 `LiftingLine` 能力构建三维 AFC 有限翼分析器。
- 不从零写 VLM。
- 复用几何、展向离散、诱导速度和三维力积分。
- 每个展向截面调用 `get_aero_from_afc_neuralfoil()`。
- 支持展向分布的 `C_mu`、`x_jet`、`theta_jet`。

适用场景：

- 快速评估 AFC 对有限翼升阻力的影响。
- AFC 分布优化的低成本分析器。
- 早期设计空间探索。

主要输出：

```text
CL
CD
CDi
CDp
CM
CY
Cl
Cn
spanwise_y
local_alpha
local_Re
local_C_mu
section_CL
section_CD
section_CM
section_cl
section_cd
section_cm
residual_CL
residual_CM
analysis_confidence
convergence_history
max_residual
iteration_count
converged
failure_reason
```

`AFCNeuralFoilLiftingLine` 是显式求解器，不做非线性残差迭代；其 `residual_CL/residual_CM` 为零数组，`converged=True`，`convergence_history` 中记录一条说明。`AFCNeuralFoilNonlinearLiftingLine` 输出隐式环量残差。`NeuralFoilCamberedVLM` 输出虚拟 camber/alpha 修正迭代的 `CL/CM` 残差历史。

当前状态：

- 可运行。
- 已有最小测试。
- 精度依赖 2D AFC 模型质量。

## 6. 非线性残差耦合 Lifting-Line 模型

位置：

```text
aerosandbox/aerodynamics/aero_3D/afc_nonlinear_lifting_line.py
```

类名：

```python
asb.AFCNeuralFoilNonlinearLiftingLine(...)
```

作用：

- 将三维环量 `gamma` 作为隐式变量。
- 通过局部诱导速度计算有效迎角 `alpha_eff`。
- 每个截面调用 AFC-NeuralFoil。
- 强制满足局部剖面升力匹配。

核心残差：

```text
R_i = CL_from_gamma_i - CL_2D(alpha_eff_i, Re_i, Mach_i, C_mu_i)
```

支持模式：

```python
run(solve=True)
```

内部求解残差。

```python
run(solve=False)
```

暴露 `gamma` 和 `residuals`，由外部 `asb.Opti` 统一约束。

当前状态：

- 已实现 `CL` 残差闭合。
- 已验证残差可收敛到数值精度。
- 暂未加入复杂 cambering / pitching-moment 隐式耦合。

## 7. Cambered VLM 模型

位置：

```text
aerosandbox/aerodynamics/aero_3D/afc_cambered_vlm.py
```

类名：

```python
asb.AFCNeuralFoilCamberedVLM(...)
asb.NeuralFoilCamberedVLM(...)
asb.CamberedVLM(...)
```

作用：

- 实现第一版 3D AFC cambering method。
- 不旋转或移动整个涡格。
- 只局部修正面板边界条件法向量。
- 用虚拟修正角使局部 VLM 气动接近 AFC-NeuralFoil 的剖面气动。

两个虚拟修正量：

```text
delta_alpha_i
delta_camber_i
```

含义：

- `delta_alpha_i`：展向截面常量法向修正，主要用于匹配 `CL`。
- `delta_camber_i`：沿弦向变化的法向修正，主要用于匹配 `CM`。

输出：

```text
residual_CL
residual_CM
delta_alpha
delta_camber
convergence_history
section_CL_vlm
section_CL
section_CM_vlm
section_CM
```

当前状态：

- 第一版可运行。
- CL 和 CM 残差可以通过欠松弛迭代收敛。
- CM 匹配中包含一个简化的等效 camber moment 响应，后续可替换为更严格的物理模型。

## 8. 3D AFC 后处理模块

位置：

```text
aerosandbox/aerodynamics/aero_3D/afc_postprocessing.py
```

主要接口：

```python
asb.postprocess_afc_3d_result(result)
asb.add_afc_3d_postprocessing(result)
asb.plot_spanwise_results(result)
```

结果 dataclass：

```python
asb.AFC3DPostProcessResult
```

标准输出字段：

```text
CL
CD
CDi
CDp
CM
root_bending_moment
spanwise_y
spanwise_load
local_stall_indicator
analysis_confidence
afc_power_proxy
section_CL
section_CD
section_CM
local_alpha
local_C_mu
residual_CL
residual_CM
```

后处理指标说明：

- `CL`：总升力系数。
- `CD`：总阻力系数。
- `CDi`：诱导阻力系数，来自近场无粘力 / 环量诱导力。
- `CDp`：剖面阻力系数，沿展向积分 AFC-NeuralFoil 的 `section_CD`。
- `CM`：俯仰力矩系数。
- `root_bending_moment`：翼根弯矩近似值。
- `spanwise_load`：展向载荷分布。
- `local_stall_indicator`：局部失速风险 proxy，范围为 0 到 1。
- `analysis_confidence`：AFC-NeuralFoil 分析置信度分布。
- `afc_power_proxy`：AFC 功耗 proxy，近似为 `integral(C_mu * chord dy)`。

绘图函数：

```python
asb.plot_spanwise_results(result)
```

可视化内容：

- 展向载荷和 `C_mu`
- `section_CL`、`section_CD`、`section_CM`
- `analysis_confidence`
- 局部失速指标
- `residual_CL`、`residual_CM`

## 9. Examples

位置：

```text
examples/
```

主要示例：

```text
afc_neuralfoil_placeholder.py
```

演示 2D AFC placeholder 接口。

```text
afc_dataset_pipeline.py
```

演示 AFC 数据集生成、读写、校验、划分和统计。

```text
afc_lifting_line_comparison.py
```

演示显式 AFC lifting-line 模型。

```text
afc_wing_distribution_optimization.py
```

演示三维机翼几何和展向 AFC 分布联合优化。

```text
afc_cambered_vlm_comparison.py
```

对比 standard VLM、NonlinearLiftingLine 和 CamberedVLM。

```text
afc_3d_postprocessing.py
```

演示 3D AFC 后处理和展向结果绘图。

## 10. 测试模块

二维测试：

```text
aerosandbox/aerodynamics/aero_2D/test_aero_2D/test_afc_neuralfoil.py
aerosandbox/aerodynamics/aero_2D/test_aero_2D/test_afc_dataset.py
aerosandbox/aerodynamics/aero_2D/test_aero_2D/test_afc_neuralfoil_training.py
```

三维测试：

```text
aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_afc_lifting_line.py
aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_afc_nonlinear_lifting_line.py
aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_afc_cambered_vlm.py
aerosandbox/aerodynamics/aero_3D/test_aero_3D/test_afc_postprocessing.py
```

验证与回归测试：

```text
aerosandbox/aerodynamics/test_aerodynamics/test_afc_validation.py
```

这些测试覆盖：

- `C_mu = 0` 退化到 baseline NeuralFoil。
- AFC 数据集 schema 校验。
- 训练模块导入和特征构造。
- 3D AFC lifting-line 输出。
- 非线性残差耦合。
- CamberedVLM 残差下降。
- 3D 后处理字段和绘图函数。
- 2D surrogate 对数据集的误差计算。
- 3D 无 AFC 与 AeroSandbox LiftingLine 基线对比。
- 有 AFC 时 `CL` 随 `C_mu` 合理增加，`CD` 和 `CM` 连续变化。
- 一个轻量 `asb.Opti` AFC 优化问题可收敛。

## 11. 验证与回归测试体系

位置：

```text
aerosandbox/aerodynamics/afc_validation.py
aerosandbox/aerodynamics/validation/
examples/afc_validation_report.py
```

主要接口：

```python
asb.run_2d_zero_afc_consistency_check()
asb.compute_2d_surrogate_error(df)
asb.run_3d_no_afc_baseline_check()
asb.run_3d_afc_monotonicity_check()
asb.run_opti_convergence_check()
asb.run_afc_validation_suite()
asb.write_afc_validation_report(report, output_directory)
asb.load_external_afc_validation_dataset(filepath)
```

验证对象：

1. `C_mu = 0` 时 AFC-NeuralFoil 与原 NeuralFoil 一致。
2. 2D surrogate 对测试数据误差可计算。
3. 3D 无 AFC 时与 AeroSandbox `LiftingLine` 和 inviscid `VortexLatticeMethod` 基线对比。
4. 有 AFC 时，`CL` 随 `C_mu` 合理增加，`CD` 和 `CM` 连续变化。
5. `NeuralFoilCamberedVLM` 输出残差收敛历史。
6. 一个轻量 `asb.Opti` AFC 优化问题可收敛。

benchmark case：

```text
aerosandbox/aerodynamics/validation/afc_benchmark_case.json
```

该 benchmark 是 CI 和回归测试用的小规模确定性算例，不是 CFD 验证数据。

生成正式验收报告：

```bash
python examples/afc_validation_report.py --config configs/afc/validation_config.example.yaml
```

报告会写出：

- `report.json`
- `report.md`
- `figures/`
  - `2d_error_summary.png`
  - `2d_polar_comparison.png`
  - `3d_baseline_comparison.csv/png`
  - `3d_afc_gain_trend.png`
  - `3d_residual_convergence.png`
  - `swap_trend_validation.png`
  - `end_to_end_optimization_comparison.png`

并记录：

- git commit / package version
- model weights path
- dataset manifest
- config path
- run time
- pass/fail 标准

如果有外部 CFD / RANS / 实验数据，可使用：

```bash
python examples/afc_validation_report.py \
  --config configs/afc/validation_config.example.yaml \
  --external-2d-dataset path/to/external_dataset.parquet \
  --output-directory validation_outputs/afc_external
```

外部大规模验证数据不放进仓库，只通过路径传入。数据必须符合 `afc_dataset.py` 中定义的统一 schema。若有 manifest，放在数据文件同目录并命名为 `*.manifest.json`，报告会自动记录。

CI 轻量验证：

```text
.github/workflows/afc-validation.yml
```

该 workflow 运行：

```bash
pytest aerosandbox/aerodynamics/test_aerodynamics/test_afc_validation.py
python examples/afc_validation_report.py --output-directory validation_outputs/afc_ci
```

## 12. 当前限制

当前实现仍有以下限制：

- 默认 2D AFC 修正是经验 placeholder。
- fake dataset 和 synthetic 训练结果不能代表真实流动物理。
- 真实精度依赖后续 SU2 / RANS / 实验数据训练。
- `analysis_confidence` 当前主要是工程置信度信号，不是严格不确定度量化。
- CamberedVLM 是第一版工程实现，不是最终物理模型。
- 翼根弯矩和 AFC 功耗 proxy 是设计优化阶段的近似指标。
- 3D 模型目前更适合快速设计空间探索，不应直接替代高保真 CFD。

## 13. 后续开发建议

建议后续按以下顺序推进：

1. 汇总真实 2D AFC 数据。
2. 按统一 schema 存储为 Parquet。
3. 训练 AFC-NeuralFoil MLP。
4. 导出 `.npz` 权重。
5. 验证 2D holdout 误差。
6. 将训练权重接入 3D 有限翼模型。
7. 用高保真三维 CFD 或实验数据校准 3D 修正。
8. 完善 `analysis_confidence` 和失速指标。
9. 将 AFC 功耗 proxy 替换为更真实的功率模型。
10. 用 `asb.Opti` 做机翼几何和 AFC 分布联合优化。
