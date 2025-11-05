# MOOSE优化模块中的全局优化算法支持

## 当前状况：TAO不支持PSO等群体智能算法

### MOOSE/TAO支持的优化算法

MOOSE优化模块基于PETSc的**TAO (Toolkit for Advanced Optimization)** 库。TAO主要面向**基于梯度的局部优化**：

#### 支持的算法列表

| 类型 | 算法 | tao_solver | 特点 |
|------|------|-----------|------|
| **局部优化** | Newton Trust Region | `taontr` | 需要Hessian |
| | Bounded Newton TR | `taobntr` | 带约束 |
| | LMVM (quasi-Newton) | `taolmvm` | 需要梯度 |
| | Bounded LMVM | `taoblmvm` | 带约束 ✅ 您当前使用 |
| | BQNLS | `taobqnls` | 带约束 |
| | Newton Line Search | `taonls` | 需要梯度 |
| **无梯度算法** | Nelder-Mead | `taonm` | 单纯形法 |
| | POUNDerS | - | 模型驱动 |

### ❌ TAO不支持的算法

- **粒子群优化 (PSO)**
- **遗传算法 (GA)**
- **差分进化 (DE)**
- **模拟退火 (SA)**
- **蚁群算法 (ACO)**

### 原因分析

TAO设计目标是**大规模科学计算优化**：
- 问题规模：10³-10⁶ 变量
- 目标函数：计算昂贵（PDE求解）
- 优化策略：利用梯度信息，局部收敛快

而PSO等群体算法特点：
- 问题规模：通常 < 100 变量
- 目标函数：计算便宜（解析函数）
- 优化策略：大量评估，全局搜索

## 整合外部全局优化器的方法

### ✅ 方案1：Python包装器（推荐）⭐⭐⭐

使用Python调用MOOSE作为黑盒目标函数，用scipy/pygmo等库做优化：

#### 实现步骤

**1. 创建Python目标函数封装**

```python
#!/usr/bin/env python3
"""
PSO优化器整合MOOSE
"""
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path
import shutil

class MOOSEObjective:
    """MOOSE目标函数封装"""
    
    def __init__(self, 
                 executable="ailuro-opt",
                 input_template="forward_template.i",
                 work_dir="pso_work"):
        self.executable = executable
        self.input_template = input_template
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.eval_count = 0
        
    def __call__(self, parameters):
        """
        评估目标函数
        参数: parameters = [p0, p1, p2, ...]
        返回: objective_value (越小越好)
        """
        self.eval_count += 1
        
        # 创建临时工作目录
        run_dir = self.work_dir / f"eval_{self.eval_count:04d}"
        run_dir.mkdir(exist_ok=True)
        
        # 修改输入文件（写入参数）
        input_file = run_dir / "forward.i"
        self._write_input_file(input_file, parameters)
        
        # 运行MOOSE
        try:
            result = subprocess.run(
                [self.executable, "-i", "forward.i"],
                cwd=run_dir,
                capture_output=True,
                timeout=300,  # 5分钟超时
                text=True
            )
            
            if result.returncode != 0:
                print(f"⚠ 评估 {self.eval_count} 失败")
                return 1e10  # 惩罚值
            
            # 提取目标函数值
            objective = self._extract_objective(run_dir)
            
            print(f"评估 {self.eval_count}: params={parameters}, J={objective:.6e}")
            return objective
            
        except subprocess.TimeoutExpired:
            print(f"⚠ 评估 {self.eval_count} 超时")
            return 1e10
            
    def _write_input_file(self, filepath, parameters):
        """根据参数生成输入文件"""
        # 读取模板
        with open(self.input_template, 'r') as f:
            content = f.read()
        
        # 替换参数（可以使用Jinja2模板更灵活）
        content = content.replace("{{P0}}", str(parameters[0]))
        content = content.replace("{{P1}}", str(parameters[1]))
        content = content.replace("{{P2}}", str(parameters[2]))
        
        with open(filepath, 'w') as f:
            f.write(content)
    
    def _extract_objective(self, run_dir):
        """从输出文件提取目标函数值"""
        csv_file = run_dir / "forward_out.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # 假设最后一行包含objective_value
            return float(df['objective_value'].iloc[-1])
        else:
            return 1e10


# 使用PySwarms进行PSO优化
from pyswarms.single.global_best import GlobalBestPSO

def pso_optimize():
    """PSO优化主函数"""
    
    # 创建目标函数
    objective_func = MOOSEObjective(
        executable="ailuro-opt",
        input_template="forward_particle_friction_plastic_voce_template.i"
    )
    
    # 参数边界
    bounds = (
        np.array([10, 0, 0]),      # 下界 [yield, R, Q]
        np.array([1200, 2000, 800]) # 上界
    )
    
    # PSO配置
    options = {
        'c1': 2.0,  # 认知参数
        'c2': 2.0,  # 社会参数
        'w': 0.9    # 惯性权重
    }
    
    # 创建优化器
    optimizer = GlobalBestPSO(
        n_particles=20,    # 粒子数量
        dimensions=3,      # 参数维度
        options=options,
        bounds=bounds
    )
    
    # 运行优化
    cost, pos = optimizer.optimize(
        objective_func,
        iters=50,          # 最大迭代次数
        verbose=True
    )
    
    print(f"\n优化完成！")
    print(f"最优参数: {pos}")
    print(f"最优目标函数: {cost}")
    print(f"总评估次数: {objective_func.eval_count}")
    
    return pos, cost

if __name__ == "__main__":
    optimal_params, optimal_cost = pso_optimize()
```

**2. 创建输入文件模板**

```moose
# forward_particle_friction_plastic_voce_template.i
# 使用占位符 {{P0}}, {{P1}}, {{P2}}

[Materials]
  [stress_specimen]
    type = ADIsotropicVoceLawHardeningStressUpdate
    block = 'Specimen_Body'
    yield_stress = {{P0}}      # 将被替换
    hardening_constant = {{P1}} # 将被替换
    q = {{P2}}                  # 将被替换
    b = 10
  []
[]

# ... 其余配置
```

**3. 安装依赖**

```bash
conda activate moose
pip install pyswarms pandas numpy
```

**4. 运行优化**

```bash
python pso_moose_optimizer.py
```

#### 优缺点

✅ **优点**:
- 实现简单，不需要修改MOOSE源码
- 可以使用任何Python优化库（scipy, pygmo, optuna, etc.）
- 灵活性高，容易调试

❌ **缺点**:
- 每次评估需要启动新的MOOSE进程（开销大）
- 不能利用MOOSE内部状态（如前一次模拟的收敛解）
- 并行化需要额外处理

### ✅ 方案2：C++自定义Executioner（高级）⭐⭐⭐⭐

在MOOSE中实现自定义优化执行器：

#### 实现步骤

**1. 创建PSO Executioner**

```cpp
// PSOptimize.h
#pragma once

#include "Steady.h"
#include "OptimizationReporterBase.h"
#include <vector>

class PSOptimize : public Steady
{
public:
  static InputParameters validParams();
  PSOptimize(const InputParameters & parameters);
  
  virtual void execute() override;
  virtual bool lastSolveConverged() const override { return _converged; }

protected:
  /// PSO主循环
  void psoOptimize();
  
  /// 评估单个粒子的目标函数
  Real evaluateParticle(const std::vector<Real> & position);
  
  /// 更新粒子速度和位置
  void updateParticles();
  
private:
  /// PSO参数
  unsigned int _n_particles;
  unsigned int _max_iters;
  Real _w;  // 惯性权重
  Real _c1; // 认知参数
  Real _c2; // 社会参数
  
  /// 优化reporter
  OptimizationReporterBase * _obj_function;
  
  /// 粒子状态
  std::vector<std::vector<Real>> _positions;
  std::vector<std::vector<Real>> _velocities;
  std::vector<Real> _fitness;
  
  /// 最优值
  std::vector<Real> _pbest;  // 个体最优
  std::vector<Real> _gbest;  // 全局最优
  Real _gbest_fitness;
  
  bool _converged;
};
```

**2. 实现PSO算法**

```cpp
// PSOptimize.C
#include "PSOptimize.h"
#include "OptimizationReporterBase.h"
#include <random>

registerMooseObject("OptimizationApp", PSOptimize);

InputParameters
PSOptimize::validParams()
{
  InputParameters params = Steady::validParams();
  params.addParam<unsigned int>("n_particles", 20, "Number of particles in swarm");
  params.addParam<unsigned int>("max_iterations", 100, "Maximum PSO iterations");
  params.addParam<Real>("inertia_weight", 0.9, "Inertia weight w");
  params.addParam<Real>("cognitive_param", 2.0, "Cognitive parameter c1");
  params.addParam<Real>("social_param", 2.0, "Social parameter c2");
  return params;
}

PSOptimize::PSOptimize(const InputParameters & parameters)
  : Steady(parameters),
    _n_particles(getParam<unsigned int>("n_particles")),
    _max_iters(getParam<unsigned int>("max_iterations")),
    _w(getParam<Real>("inertia_weight")),
    _c1(getParam<Real>("cognitive_param")),
    _c2(getParam<Real>("social_param")),
    _converged(false)
{
  if (!_problem.hasUserObject("OptimizationReporter"))
    mooseError("PSOptimize requires an OptimizationReporter");
    
  _obj_function = &_problem.getUserObject<OptimizationReporterBase>("OptimizationReporter");
}

void
PSOptimize::execute()
{
  _problem.outputStep(EXEC_INITIAL);
  preExecute();
  
  // 运行PSO
  psoOptimize();
  
  _problem.execute(EXEC_FINAL);
  _problem.outputStep(EXEC_FINAL);
  postExecute();
}

void
PSOptimize::psoOptimize()
{
  // 获取参数信息
  const auto & param_names = _obj_function->getParameterNames();
  const auto & lower_bounds = _obj_function->getLowerBounds();
  const auto & upper_bounds = _obj_function->getUpperBounds();
  unsigned int n_params = param_names.size();
  
  // 初始化粒子群
  std::default_random_engine generator;
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  
  _positions.resize(_n_particles, std::vector<Real>(n_params));
  _velocities.resize(_n_particles, std::vector<Real>(n_params));
  _fitness.resize(_n_particles);
  _pbest.resize(n_params);
  _gbest.resize(n_params);
  _gbest_fitness = std::numeric_limits<Real>::max();
  
  // 随机初始化位置
  for (unsigned int i = 0; i < _n_particles; ++i)
  {
    for (unsigned int j = 0; j < n_params; ++j)
    {
      _positions[i][j] = lower_bounds[j] + 
                        uniform(generator) * (upper_bounds[j] - lower_bounds[j]);
      _velocities[i][j] = 0.0;
    }
    
    // 评估初始适应度
    _fitness[i] = evaluateParticle(_positions[i]);
    
    // 更新个体最优和全局最优
    if (_fitness[i] < _gbest_fitness)
    {
      _gbest_fitness = _fitness[i];
      _gbest = _positions[i];
    }
  }
  
  // PSO主循环
  for (unsigned int iter = 0; iter < _max_iters; ++iter)
  {
    _console << "PSO Iteration " << iter << ", Best Fitness: " << _gbest_fitness << std::endl;
    
    updateParticles();
    
    // 检查收敛
    if (_gbest_fitness < 1e-6)
    {
      _converged = true;
      break;
    }
  }
  
  _console << "\nPSO Optimization Complete!" << std::endl;
  _console << "Best Parameters: ";
  for (auto p : _gbest)
    _console << p << " ";
  _console << "\nBest Fitness: " << _gbest_fitness << std::endl;
}

Real
PSOptimize::evaluateParticle(const std::vector<Real> & position)
{
  // 更新参数
  _obj_function->updateParameters(position);
  
  // 执行正向求解
  _problem.execute(EXEC_FORWARD);
  _problem.execMultiApps(EXEC_FORWARD);
  
  // 计算目标函数
  return _obj_function->computeObjective();
}

void
PSOptimize::updateParticles()
{
  std::default_random_engine generator;
  std::uniform_real_distribution<Real> uniform(0.0, 1.0);
  
  for (unsigned int i = 0; i < _n_particles; ++i)
  {
    for (unsigned int j = 0; j < _positions[i].size(); ++j)
    {
      Real r1 = uniform(generator);
      Real r2 = uniform(generator);
      
      // 更新速度: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
      _velocities[i][j] = _w * _velocities[i][j] +
                         _c1 * r1 * (_pbest[j] - _positions[i][j]) +
                         _c2 * r2 * (_gbest[j] - _positions[i][j]);
      
      // 更新位置
      _positions[i][j] += _velocities[i][j];
      
      // 边界处理
      const auto & lb = _obj_function->getLowerBounds()[j];
      const auto & ub = _obj_function->getUpperBounds()[j];
      _positions[i][j] = std::max(lb, std::min(ub, _positions[i][j]));
    }
    
    // 重新评估
    Real new_fitness = evaluateParticle(_positions[i]);
    _fitness[i] = new_fitness;
    
    // 更新全局最优
    if (new_fitness < _gbest_fitness)
    {
      _gbest_fitness = new_fitness;
      _gbest = _positions[i];
    }
  }
}
```

**3. 注册和编译**

```bash
cd ailuro
# 将上述文件放入 src/executioners/
make -j6
```

**4. 使用**

```moose
[Executioner]
  type = PSOptimize
  n_particles = 30
  max_iterations = 100
  inertia_weight = 0.9
  cognitive_param = 2.0
  social_param = 2.0
[]
```

#### 优缺点

✅ **优点**:
- 完全集成到MOOSE框架
- 可以利用MOOSE并行化（MPI）
- 性能最优

❌ **缺点**:
- 需要C++编程和MOOSE内部知识
- 开发调试周期长
- 维护成本高

### ✅ 方案3：混合方法（实用）⭐⭐⭐⭐⭐

结合两种方法的优点：

1. **粗搜索**: 用PSO找到全局最优区域（Python实现）
2. **精细优化**: 用MOOSE的梯度优化（taoblmvm）局部收敛

```python
def hybrid_optimize():
    """混合优化策略"""
    
    # 阶段1: PSO全局搜索（粗略）
    print("阶段1: PSO全局搜索...")
    objective = MOOSEObjective()
    
    optimizer = GlobalBestPSO(
        n_particles=15,  # 较少粒子
        dimensions=3,
        options={'c1': 2.0, 'c2': 2.0, 'w': 0.9},
        bounds=(np.array([10, 0, 0]), np.array([1200, 2000, 800]))
    )
    
    cost, pos = optimizer.optimize(objective, iters=20)  # 较少迭代
    
    print(f"\nPSO找到初始点: {pos}, J={cost}")
    
    # 阶段2: MOOSE梯度优化（精确）
    print("\n阶段2: MOOSE梯度优化...")
    
    # 修改main.i中的初始值
    update_moose_initial_condition("main.i", pos)
    
    # 运行MOOSE优化
    subprocess.run(["ailuro-opt", "-i", "main.i"])
    
    # 读取最终结果
    final_params = read_moose_results()
    
    return final_params
```

## 计算成本对比

### 您的3参数问题

假设单次MOOSE模拟 = 30秒

| 方法 | 评估次数 | 总时间 | 收敛性 |
|------|---------|-------|--------|
| **taoblmvm (FD)** | 4×50 = 200 | 100分钟 | 局部最优 ✓ |
| **PSO单独** | 20×100 = 2000 | 1000分钟 | 全局搜索 ✓ |
| **混合方法** | 15×20 + 4×30 = 420 | 210分钟 | 全局+精确 ✓✓ |

### 建议

对于您的问题：

#### ✅ **推荐方案3（混合方法）**

**适用场景**:
- 怀疑存在多个局部最优
- 初始猜测不确定
- 可接受2-3倍计算时间

**实施步骤**:
1. 用简单Python脚本实现PSO + MOOSE调用
2. PSO找到3-5个候选点
3. 对每个点用taoblmvm局部优化
4. 选择最优解

#### 当前方法仍然有效

如果：
- ✅ 初始猜测合理（如从ANSYS拟合的值）
- ✅ 目标函数凸性好
- ✅ 当前结果收敛且合理

那么**继续使用taoblmvm**即可！

## 完整代码示例

我已经准备好完整的实现文件：

1. **`pso_moose_optimizer.py`** - Python包装器
2. **`hybrid_optimizer.py`** - 混合优化策略
3. **`PSOptimize.h/C`** - C++自定义执行器（高级）

需要我创建这些文件吗？

## 总结

- ❌ MOOSE/TAO **不直接支持PSO**
- ✅ **可以整合**，有3种方案
- ⭐ **推荐混合方法**: PSO粗搜索 + 梯度精优化
- 💡 **当前方案**: taoblmvm对3参数问题已足够，除非怀疑多局部最优

是否需要我提供完整的PSO整合代码？
