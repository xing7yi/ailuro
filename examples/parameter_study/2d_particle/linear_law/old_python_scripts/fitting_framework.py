#!/usr/bin/env python3
"""
通用拟合框架 - 支持多种应变因子模型的可扩展框架

特性:
- 统一的模型注册机制
- 通用的数据读取和预处理
- 批量拟合和结果管理
- 灵活的配置系统
- 易于添加新模型

用法:
    from fitting_framework import FittingFramework, Model
    
    # 创建框架实例
    framework = FittingFramework(file_base='main_lh_sampler')
    
    # 运行拟合
    framework.fit_all()
    
    # 生成报告
    framework.generate_reports()
"""

import csv
import sys
import glob
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
import re
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class FittingConfig:
    """拟合配置类"""
    initial_height: float = 1.0              # 试样初始高度 (mm)
    contact_area: float = np.pi * (1.0 ** 2) # 接触面积 (mm²)
    disp_col: str = 'disp_abs'               # 位移列名
    force_col: str = 'force'                 # 力列名
    min_displacement: float = 0.4            # 最小位移要求 (mm)
    maxfev: int = 20000                      # 最大迭代次数
    method: str = 'trf'                      # 优化方法
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'initial_height': self.initial_height,
            'contact_area': self.contact_area,
            'disp_col': self.disp_col,
            'force_col': self.force_col,
            'min_displacement': self.min_displacement
        }


# ============================================================================
# 基础数学函数
# ============================================================================

def compute_I1(lmbd: np.ndarray) -> np.ndarray:
    """计算第一不变量 I₁ = λ² + 2/λ"""
    return np.power(lmbd, 2) + 2.0 / lmbd


def polynomial_term(lmbd: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """多项式项: a + 2b(I₁-3) + 3c(I₁-3)²"""
    I1 = compute_I1(lmbd)
    I1_minus_3 = I1 - 3.0
    return a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2)


# ============================================================================
# 模型抽象基类
# ============================================================================

class Model(ABC):
    """模型抽象基类"""
    
    def __init__(self, name: str, description: str, n_params: int = 3):
        self.name = name
        self.description = description
        self.n_params = n_params
        self.key = name.lower().replace(' ', '_')
    
    @abstractmethod
    def compute_stress(self, epsilon: np.ndarray, *params) -> np.ndarray:
        """
        计算应力
        
        参数:
            epsilon: 应变数组
            *params: 模型参数
        
        返回:
            stress: 应力数组
        """
        pass
    
    @abstractmethod
    def get_strain_factor_description(self) -> str:
        """返回应变因子的数学描述"""
        pass
    
    def initial_guess(self, epsilon: np.ndarray, stress: np.ndarray) -> List[float]:
        """
        生成初始猜测值
        
        默认实现：基于线性拟合估算
        """
        n_init = min(10, len(epsilon))
        E_guess = np.polyfit(epsilon[:n_init], stress[:n_init], 1)[0]
        a_guess = max(E_guess / 2, 1.0)
        
        if self.n_params == 3:
            return [a_guess, a_guess * 0.1, a_guess * 0.01]
        elif self.n_params == 4:
            return [a_guess, a_guess * 0.1, a_guess * 0.01, a_guess * 0.001]
        else:
            return [a_guess] * self.n_params
    
    def parameter_bounds(self) -> Tuple:
        """
        返回参数边界
        
        默认实现：a > 0, 其他不限
        """
        if self.n_params == 3:
            return ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        elif self.n_params == 4:
            return ([0, -np.inf, -np.inf, -np.inf], 
                   [np.inf, np.inf, np.inf, np.inf])
        else:
            lower = [0] + [-np.inf] * (self.n_params - 1)
            upper = [np.inf] * self.n_params
            return (lower, upper)
    
    def parameter_names(self) -> List[str]:
        """返回参数名称列表"""
        if self.n_params == 3:
            return ['a', 'b', 'c']
        elif self.n_params == 4:
            return ['a', 'b', 'c', 'd']
        else:
            return [f'p{i}' for i in range(self.n_params)]


# ============================================================================
# 具体模型实现
# ============================================================================

class YeohModel(Model):
    """Yeoh 模型: σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]"""
    
    def __init__(self):
        super().__init__(
            name='Yeoh',
            description='σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
            n_params=3
        )
    
    def compute_stress(self, epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        lmbd = epsilon + 1.0
        strain_factor = lmbd - 1.0 / (lmbd**2)
        poly = polynomial_term(lmbd, a, b, c)
        return 2.0 * strain_factor * poly
    
    def get_strain_factor_description(self) -> str:
        return '(λ - λ⁻²)'


class LinearModel(Model):
    """Linear 模型: σ = 2(λ - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]"""
    
    def __init__(self):
        super().__init__(
            name='Linear',
            description='σ = 2(λ - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
            n_params=3
        )
    
    def compute_stress(self, epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        lmbd = epsilon + 1.0
        strain_factor = lmbd - 1.0
        poly = polynomial_term(lmbd, a, b, c)
        return 2.0 * strain_factor * poly
    
    def get_strain_factor_description(self) -> str:
        return '(λ - 1)'


class LogarithmicModel(Model):
    """Logarithmic 模型: σ = 2ln(λ²) × [a + 2b(I₁-3) + 3c(I₁-3)²]"""
    
    def __init__(self):
        super().__init__(
            name='Logarithmic',
            description='σ = 2ln(λ²) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
            n_params=3
        )
    
    def compute_stress(self, epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        lmbd = epsilon + 1.0
        strain_factor = np.log(lmbd**2)
        poly = polynomial_term(lmbd, a, b, c)
        return 2.0 * strain_factor * poly
    
    def get_strain_factor_description(self) -> str:
        return 'ln(λ²)'


class QuadraticModel(Model):
    """Quadratic 模型: σ = 2(λ² - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]"""
    
    def __init__(self):
        super().__init__(
            name='Quadratic',
            description='σ = 2(λ² - 1) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
            n_params=3
        )
    
    def compute_stress(self, epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        lmbd = epsilon + 1.0
        strain_factor = lmbd**2 - 1.0
        poly = polynomial_term(lmbd, a, b, c)
        return 2.0 * strain_factor * poly
    
    def get_strain_factor_description(self) -> str:
        return '(λ² - 1)'


class CubicModel(Model):
    """Cubic 模型: σ = 2(λ - 1/λ³) × [a + 2b(I₁-3) + 3c(I₁-3)²]"""
    
    def __init__(self):
        super().__init__(
            name='Cubic',
            description='σ = 2(λ - 1/λ³) × [a + 2b(I₁-3) + 3c(I₁-3)²]',
            n_params=3
        )
    
    def compute_stress(self, epsilon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        lmbd = epsilon + 1.0
        strain_factor = lmbd - 1.0 / (lmbd**3)
        poly = polynomial_term(lmbd, a, b, c)
        return 2.0 * strain_factor * poly
    
    def get_strain_factor_description(self) -> str:
        return '(λ - 1/λ³)'


class Yeoh4thOrderModel(Model):
    """Yeoh 4阶模型: σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)² + 4d(I₁-3)³]"""
    
    def __init__(self):
        super().__init__(
            name='Yeoh_4th',
            description='σ = 2(λ - λ⁻²) × [a + 2b(I₁-3) + 3c(I₁-3)² + 4d(I₁-3)³]',
            n_params=4
        )
    
    def compute_stress(self, epsilon: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        lmbd = epsilon + 1.0
        I1 = compute_I1(lmbd)
        I1_minus_3 = I1 - 3.0
        strain_factor = lmbd - 1.0 / (lmbd**2)
        poly = a + 2*b*I1_minus_3 + 3*c*(I1_minus_3**2) + 4*d*(I1_minus_3**3)
        return 2.0 * strain_factor * poly
    
    def get_strain_factor_description(self) -> str:
        return '(λ - λ⁻²)'


# ============================================================================
# 模型注册表
# ============================================================================

class ModelRegistry:
    """模型注册表 - 管理所有可用模型"""
    
    def __init__(self):
        self._models: Dict[str, Model] = {}
    
    def register(self, model: Model):
        """注册模型"""
        self._models[model.key] = model
    
    def get(self, key: str) -> Optional[Model]:
        """获取模型"""
        return self._models.get(key)
    
    def get_all(self) -> Dict[str, Model]:
        """获取所有模型"""
        return self._models.copy()
    
    def keys(self) -> List[str]:
        """获取所有模型键"""
        return list(self._models.keys())


# 创建全局注册表并注册默认模型
DEFAULT_REGISTRY = ModelRegistry()
DEFAULT_REGISTRY.register(YeohModel())
DEFAULT_REGISTRY.register(LinearModel())
DEFAULT_REGISTRY.register(LogarithmicModel())
DEFAULT_REGISTRY.register(QuadraticModel())
DEFAULT_REGISTRY.register(CubicModel())
DEFAULT_REGISTRY.register(Yeoh4thOrderModel())


# ============================================================================
# 数据管理类
# ============================================================================

@dataclass
class RunnerData:
    """单个runner的数据"""
    runner_num: int
    filename: str
    epsilon: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    n_points: int = 0
    max_displacement: float = 0.0
    max_force: float = 0.0
    max_stress: float = 0.0
    success: bool = False
    message: str = ''


@dataclass
class FitResult:
    """拟合结果"""
    runner_num: int
    filename: str
    model_key: str
    params: Dict[str, float] = field(default_factory=dict)
    r_squared: float = np.nan
    rmse: float = np.nan
    rel_rmse: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    n_points: int = 0
    max_displacement: float = 0.0
    max_force: float = 0.0
    max_stress: float = 0.0
    fit_success: bool = False
    message: str = ''
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            'runner': self.runner_num,
            'filename': self.filename,
            **self.params,
            'r_squared': self.r_squared,
            'rmse': self.rmse,
            'rel_rmse': self.rel_rmse,
            'aic': self.aic,
            'bic': self.bic,
            'n_points': self.n_points,
            'max_displacement': self.max_displacement,
            'max_force': self.max_force,
            'max_stress': self.max_stress,
            'fit_success': self.fit_success,
            'message': self.message
        }
        return result


# ============================================================================
# 文件工具函数
# ============================================================================

def extract_runner_number(filename: str) -> int:
    """从文件名提取runner编号"""
    match = re.search(r'runner(\d+)', filename)
    return int(match.group(1)) if match else -1


def find_runner_files(file_base: str = None) -> Tuple[List[str], str]:
    """查找所有runner CSV文件"""
    if file_base:
        pattern = f"{file_base}_out_runner*.csv"
    else:
        patterns = glob.glob("*_out_runner*.csv")
        if not patterns:
            return None, None
        
        first_file = patterns[0]
        match = re.match(r'(.+?)_out_runner\d+\.csv', first_file)
        if match:
            file_base = match.group(1)
            pattern = f"{file_base}_out_runner*.csv"
        else:
            return None, None
    
    files = sorted(glob.glob(pattern), key=extract_runner_number)
    
    if not files:
        return None, None
    
    return files, file_base


# ============================================================================
# 主框架类
# ============================================================================

class FittingFramework:
    """拟合框架主类"""
    
    def __init__(self, 
                 file_base: str = None,
                 config: FittingConfig = None,
                 model_registry: ModelRegistry = None):
        """
        初始化拟合框架
        
        参数:
            file_base: 文件基础名称
            config: 拟合配置
            model_registry: 模型注册表（None则使用默认）
        """
        self.config = config or FittingConfig()
        self.registry = model_registry or DEFAULT_REGISTRY
        self.file_base = file_base
        self.runner_files: List[str] = []
        self.runner_data: Dict[int, RunnerData] = {}
        self.fit_results: Dict[str, List[FitResult]] = {}
        
        # 查找文件
        if file_base:
            files, detected_base = find_runner_files(file_base)
            if files:
                self.runner_files = files
                self.file_base = file_base
        
    def read_runner_data(self, csv_file: str) -> RunnerData:
        """读取单个runner的CSV数据"""
        runner_num = extract_runner_number(csv_file)
        
        try:
            displacement_data = []
            force_data = []
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        disp = float(row.get(self.config.disp_col, 0))
                        force = float(row.get(self.config.force_col, 0))
                        
                        if abs(disp) > 1e-6 and force > 0:
                            displacement_data.append(abs(disp))
                            force_data.append(force)
                    except (ValueError, KeyError):
                        continue
            
            if len(displacement_data) < 5:
                return RunnerData(
                    runner_num=runner_num,
                    filename=csv_file,
                    n_points=len(displacement_data),
                    success=False,
                    message=f'Not enough valid data points ({len(displacement_data)} < 5)'
                )
            
            displacement = np.array(displacement_data)
            force = np.array(force_data)
            max_disp = np.max(displacement)
            
            if max_disp < self.config.min_displacement:
                return RunnerData(
                    runner_num=runner_num,
                    filename=csv_file,
                    n_points=len(displacement),
                    max_displacement=max_disp,
                    max_force=np.max(force),
                    success=False,
                    message=f'Insufficient displacement: max={max_disp:.4f} < required={self.config.min_displacement}'
                )
            
            # 计算应变和应力
            epsilon = displacement / self.config.initial_height
            stress = force / self.config.contact_area
            
            return RunnerData(
                runner_num=runner_num,
                filename=csv_file,
                epsilon=epsilon,
                stress=stress,
                n_points=len(epsilon),
                max_displacement=max_disp,
                max_force=np.max(force),
                max_stress=np.max(stress),
                success=True,
                message='Success'
            )
            
        except Exception as e:
            return RunnerData(
                runner_num=runner_num,
                filename=csv_file,
                success=False,
                message=f'Error reading file: {str(e)}'
            )
    
    def fit_model(self, data: RunnerData, model: Model) -> FitResult:
        """使用指定模型拟合数据"""
        
        # 创建基础结果对象
        result = FitResult(
            runner_num=data.runner_num,
            filename=data.filename,
            model_key=model.key,
            n_points=data.n_points,
            max_displacement=data.max_displacement,
            max_force=data.max_force,
            max_stress=data.max_stress
        )
        
        if not data.success:
            result.fit_success = False
            result.message = data.message
            return result
        
        try:
            # 初始猜测和边界
            p0 = model.initial_guess(data.epsilon, data.stress)
            bounds = model.parameter_bounds()
            
            # 拟合
            popt, pcov = curve_fit(
                model.compute_stress,
                data.epsilon,
                data.stress,
                p0=p0,
                bounds=bounds,
                maxfev=self.config.maxfev,
                method=self.config.method
            )
            
            # 保存参数
            param_names = model.parameter_names()
            result.params = {name: val for name, val in zip(param_names, popt)}
            
            # 计算拟合质量
            stress_pred = model.compute_stress(data.epsilon, *popt)
            residuals = data.stress - stress_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((data.stress - np.mean(data.stress))**2)
            
            result.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            result.rmse = np.sqrt(np.mean(residuals**2))
            result.rel_rmse = result.rmse / np.mean(data.stress) if np.mean(data.stress) > 0 else np.inf
            
            # 计算AIC和BIC
            n = len(data.epsilon)
            k = model.n_params
            log_likelihood = -n/2 * np.log(2*np.pi*ss_res/n) - n/2
            result.aic = 2*k - 2*log_likelihood
            result.bic = k*np.log(n) - 2*log_likelihood
            
            result.fit_success = True
            result.message = 'Success'
            
        except Exception as e:
            result.fit_success = False
            result.message = f'Fitting failed: {str(e)}'
        
        return result
    
    def fit_all(self, model_keys: List[str] = None, verbose: bool = True):
        """
        对所有runner和所有指定模型进行拟合
        
        参数:
            model_keys: 要使用的模型键列表（None则使用所有注册模型）
            verbose: 是否显示进度信息
        """
        if not self.runner_files:
            print("错误: 没有找到runner文件")
            return
        
        # 确定要使用的模型
        if model_keys is None:
            model_keys = self.registry.keys()
        
        models = {key: self.registry.get(key) for key in model_keys}
        models = {k: v for k, v in models.items() if v is not None}
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Multi-Model Fitting Framework")
            print(f"{'='*70}")
            print(f"File base: {self.file_base}")
            print(f"Runner files: {len(self.runner_files)}")
            print(f"Models: {', '.join([m.name for m in models.values()])}")
            print(f"{'='*70}\n")
        
        # 初始化结果存储
        self.fit_results = {key: [] for key in models.keys()}
        
        # 处理每个runner文件
        for i, csv_file in enumerate(self.runner_files):
            # 读取数据
            data = self.read_runner_data(csv_file)
            self.runner_data[data.runner_num] = data
            
            # 对每个模型进行拟合
            for model_key, model in models.items():
                result = self.fit_model(data, model)
                self.fit_results[model_key].append(result)
            
            # 进度显示
            if verbose and ((i + 1) % 10 == 0 or (i + 1) == len(self.runner_files)):
                success_counts = {key: sum(1 for r in results if r.fit_success) 
                                for key, results in self.fit_results.items()}
                print(f"  已处理 {i + 1}/{len(self.runner_files)} | " + 
                      " | ".join([f"{models[k].name}: {v}" for k, v in success_counts.items()]))
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"拟合完成!")
            print(f"{'='*70}\n")
    
    def save_results(self, output_dir: str = '.'):
        """保存拟合结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"保存拟合结果...")
        
        # 保存各模型的详细结果
        for model_key, results in self.fit_results.items():
            model = self.registry.get(model_key)
            if not model or not results:
                continue
            
            output_file = output_path / f"{self.file_base}_fit_params_{model_key}.csv"
            
            # 获取字段名
            first_result = results[0].to_dict()
            fieldnames = list(first_result.keys())
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result.to_dict())
            
            success_count = sum(1 for r in results if r.fit_success)
            print(f"  {model.name:15s}: {output_file.name:45s} "
                  f"(成功: {success_count}/{len(results)})")
        
        # 保存对比结果
        self.save_comparison_results(output_path)
    
    def save_comparison_results(self, output_path: Path):
        """保存模型对比结果"""
        if not self.fit_results:
            return
        
        comparison_file = output_path / f"{self.file_base}_fit_comparison.csv"
        
        # 构建对比数据
        comparison_data = []
        runner_nums = sorted(set(r.runner_num for results in self.fit_results.values() for r in results))
        
        for runner_num in runner_nums:
            row = {'runner': runner_num}
            
            # 收集各模型的结果
            for model_key, results in self.fit_results.items():
                result = next((r for r in results if r.runner_num == runner_num), None)
                if result:
                    row[f'{model_key}_r2'] = result.r_squared
                    row[f'{model_key}_rmse'] = result.rmse
                    row[f'{model_key}_aic'] = result.aic
                    row[f'{model_key}_bic'] = result.bic
                    row[f'{model_key}_success'] = result.fit_success
            
            # 找出最佳模型
            r2_values = {k: row.get(f'{k}_r2', np.nan) 
                        for k in self.fit_results.keys() 
                        if row.get(f'{k}_success', False) and not np.isnan(row.get(f'{k}_r2', np.nan))}
            
            if r2_values:
                best_model = max(r2_values, key=r2_values.get)
                row['best_model_r2'] = best_model
                row['best_r2_value'] = r2_values[best_model]
            else:
                row['best_model_r2'] = 'none'
                row['best_r2_value'] = np.nan
            
            comparison_data.append(row)
        
        # 写入CSV
        if comparison_data:
            fieldnames = list(comparison_data[0].keys())
            with open(comparison_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparison_data)
            
            print(f"  Comparison     : {comparison_file.name}")
    
    def generate_summary(self) -> str:
        """生成统计摘要"""
        summary_lines = []
        summary_lines.append("="*70)
        summary_lines.append("Fitting Summary")
        summary_lines.append("="*70)
        summary_lines.append("")
        
        for model_key, results in self.fit_results.items():
            model = self.registry.get(model_key)
            if not model:
                continue
            
            successful = [r for r in results if r.fit_success]
            success_rate = len(successful) / len(results) * 100 if results else 0
            
            summary_lines.append(f"{model.name} Model:")
            summary_lines.append(f"  Description: {model.description}")
            summary_lines.append(f"  Success Rate: {len(successful)}/{len(results)} ({success_rate:.1f}%)")
            
            if successful:
                r2_values = [r.r_squared for r in successful]
                summary_lines.append(f"  R² Statistics:")
                summary_lines.append(f"    Min: {np.min(r2_values):.6f}")
                summary_lines.append(f"    Max: {np.max(r2_values):.6f}")
                summary_lines.append(f"    Mean: {np.mean(r2_values):.6f}")
                summary_lines.append(f"    Std: {np.std(r2_values):.6f}")
            
            summary_lines.append("")
        
        summary_lines.append("="*70)
        
        return "\n".join(summary_lines)
    
    def print_summary(self):
        """打印统计摘要"""
        print(self.generate_summary())
    
    def save_summary(self, output_dir: str = '.'):
        """保存统计摘要到文件"""
        output_path = Path(output_dir)
        summary_file = output_path / f"{self.file_base}_fit_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(self.generate_summary())
        
        print(f"\n统计摘要已保存到: {summary_file}")


# ============================================================================
# 便捷函数
# ============================================================================

def quick_fit(file_base: str = None, 
              model_keys: List[str] = None,
              config: FittingConfig = None,
              output_dir: str = '.'):
    """
    快速拟合函数 - 一键完成所有操作
    
    参数:
        file_base: 文件基础名称
        model_keys: 要使用的模型键列表
        config: 拟合配置
        output_dir: 输出目录
    """
    framework = FittingFramework(file_base=file_base, config=config)
    framework.fit_all(model_keys=model_keys)
    framework.save_results(output_dir=output_dir)
    framework.save_summary(output_dir=output_dir)
    framework.print_summary()
    
    return framework


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal fitting framework for strain factor models'
    )
    parser.add_argument('file_base', nargs='?', default=None,
                       help='File base name (auto-detect if not provided)')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Model keys to use (all if not specified)')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--initial-height', type=float, default=1.0,
                       help='Initial height in mm (default: 1.0)')
    parser.add_argument('--contact-area', type=float, default=None,
                       help='Contact area in mm² (default: π*1.0²)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = FittingConfig(
        initial_height=args.initial_height,
        contact_area=args.contact_area or np.pi * 1.0**2
    )
    
    # 运行拟合
    quick_fit(
        file_base=args.file_base,
        model_keys=args.models,
        config=config,
        output_dir=args.output_dir
    )
