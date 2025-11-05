# PSO优化器使用指南

## 核心改进：无需修改输入文件

通过MOOSE的命令行参数覆盖功能，**直接传递参数**，无需创建模板文件！

## 工作原理

```bash
# MOOSE支持命令行覆盖参数
ailuro-opt -i forward.i \
  Materials/stress_specimen/yield_stress=500 \
  Materials/stress_specimen/hardening_constant=1000 \
  Materials/stress_specimen/q=400
```

PSO优化器会自动为每个粒子生成对应的命令。

## 快速开始

### 1. 安装依赖

```bash
conda activate moose
pip install pyswarms pandas pyyaml numpy
```

### 2. 配置优化参数

编辑 `pso_config.yaml`:

```yaml
# 指定MOOSE输入文件（原始文件，无需修改）
input_file: "forward_particle_friction_plastic_voce_.i"

# 参数路径（MOOSE中的完整路径）
parameters:
  paths:
    - "Materials/stress_specimen/yield_stress"
    - "Materials/stress_specimen/hardening_constant"
    - "Materials/stress_specimen/q"
  lower_bounds: [10, 0, 0]
  upper_bounds: [1200, 2000, 800]

# PSO配置
pso:
  n_particles: 10      # 建议先用小规模测试
  max_iterations: 20
```

### 3. 测试单次评估

```bash
python test_single_eval.py
```

应该看到：
```
✓ 模拟成功完成！
  目标函数值: 1.234e-03
```

### 4. 运行PSO优化

```bash
python pso_moose_optimizer.py --config pso_config.yaml
```

## 配置文件详解

### 关键配置项

```yaml
# 原始输入文件（不需要任何占位符）
input_file: "forward_particle_friction_plastic_voce_.i"

# 参数路径 - 使用输入文件顶层定义的变量名（更简洁！）
parameters:
  names: ["p0", "p1", "p2"]
  paths:
    - "p0"  # 对应 yield_stress，在输入文件中定义为 p0 = 404.96
    - "p1"  # 对应 hardening_constant，在输入文件中定义为 p1 = 802.53
    - "p2"  # 对应 q，在输入文件中定义为 p2 = 315.89
    
# 优势：
# 1. 命令行参数更短：p0=450 vs Materials/stress_specimen/yield_stress=450
# 2. 与file_base命名一致，输出文件自动包含参数值
    - "Materials/stress_specimen/hardening_constant"
    - "Materials/stress_specimen/q"
```

### 参数路径规则

1. **完整层级路径**：从顶层Block到参数名
2. **区分大小写**：`Materials` ≠ `materials`
3. **检查输入文件**：确保路径存在

示例输入文件结构：
```moose
[Materials]
  [stress_specimen]
    type = ADIsotropicVoceLawHardeningStressUpdate
    yield_stress = 400          # ← Materials/stress_specimen/yield_stress
    hardening_constant = 800    # ← Materials/stress_specimen/hardening_constant
    q = 300                     # ← Materials/stress_specimen/q
  []
[]
```

### 输出配置

```yaml
output_csv: "forward_out.csv"           # MOOSE输出的CSV文件
objective_column: "objective_value"     # 目标函数列名
```

确保您的输入文件有：
```moose
[Outputs]
  csv = true
  file_base = forward
[]

[Reporters]
  [measure_data]
    type = OptimizationData
    objective_name = objective_value  # ← 这个名字会出现在CSV中
  []
[]
```

## 计算成本估算

```python
总评估次数 = n_particles × (max_iterations + 1)

示例配置:
  n_particles: 10
  max_iterations: 20
  → 总评估: 10 × 21 = 210 次

假设单次模拟 30秒:
  → 总时间: 210 × 30s = 105分钟 ≈ 1.75小时
```

## 调试技巧

### 1. 检查参数路径是否正确

```bash
# 手动测试命令行覆盖
ailuro-opt -i forward.i Materials/stress_specimen/yield_stress=500

# 查看是否生效（检查输出或日志）
```

### 2. 检查输出文件

```bash
ls -lh forward_out.csv
head -5 forward_out.csv  # 查看列名
```

### 3. 查看评估历史

优化完成后：
```bash
cd pso_optimization_work
cat pso_evaluation_history.csv
```

### 4. 逐步测试

```bash
# 第1步：测试单次评估
python test_single_eval.py

# 第2步：小规模测试（2粒子，2迭代）
# 修改 pso_config.yaml
pso:
  n_particles: 2
  max_iterations: 2

python pso_moose_optimizer.py --config pso_config.yaml

# 第3步：完整优化
# 恢复正常配置运行
```

## 与有限差分方法对比

| 方法 | 评估次数 | 并行 | 全局搜索 |
|------|---------|------|---------|
| **taoblmvm (FD)** | 4×50=200 | ✗ | ✗ 局部 |
| **PSO** | 10×21=210 | ✓ | ✓ 全局 |
| **PSO并行** | 10×21=210 | ✓ 10x加速 | ✓ 全局 |

PSO可以轻松并行化（修改`n_processes`参数）。

## 高级用法

### 并行评估（显著加速）

```python
# 在 pso_moose_optimizer.py 中修改
optimizer.optimize(
    objective_func,
    iters=max_iters,
    n_processes=4  # 4个并行进程
)
```

注意：确保有足够的CPU核心和内存。

### 动态惯性权重

```python
# 线性递减惯性权重（前期探索，后期收敛）
for iter in range(max_iters):
    w = 0.9 - (0.9 - 0.4) * iter / max_iters
    optimizer.options['w'] = w
```

## 故障排查

### 问题：所有评估都失败

**检查**：
1. MOOSE可执行文件路径正确？`which ailuro-opt`
2. 输入文件存在？`ls forward_particle_friction_plastic_voce_.i`
3. 辅助文件（网格等）存在？

### 问题：找不到目标函数值

**检查**：
1. 输出文件名正确？查看 `pso_optimization_work/eval_00001/`
2. 列名匹配？打开CSV查看实际列名
3. 模拟是否真的成功？检查 `.e` 文件或日志

### 问题：参数没有生效

**检查**：
1. 参数路径是否精确匹配？（大小写敏感）
2. 手动测试命令行覆盖是否工作
3. 输入文件中参数是否存在

## 总结

✅ **优点**：
- 无需修改原始输入文件
- 无需创建模板
- 参数传递清晰明确
- 易于调试和验证

✅ **适用场景**：
- 需要全局优化
- 怀疑存在多个局部最优
- 参数初值不确定
- 想要探索参数空间

📝 **建议**：
先用小规模（2粒子×2迭代）测试整个流程，确保一切正常后再进行完整优化。
