import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import subprocess
import shutil

@dataclass
class MOOSEConfig:
    """MOOSE 优化配置数据类"""

    param_names: list[str]
    param_paths: list[str]

    filebase_param: str 

    reference_csv: Path
    csv_col_x: str
    csv_col_y: str

    timeout: int = 180
    n_interp_points: int = 10
    executable: str = 'ailuro-opt'
    work_dir: Path = Path('.')
    input_file: Path = Path('input.i')
    output_dir: Path = Path('sim_results')

    def __post_init__(self):
        """后初始化以确保路径是 Path 对象"""
        if not isinstance(self.input_file, Path):
            self.input_file = Path(self.input_file)
        if not isinstance(self.reference_csv, Path):
            self.reference_csv = Path(self.reference_csv)
        if not isinstance(self.work_dir, Path):
            self.work_dir = Path(self.work_dir)
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_dict(cls, config: dict) -> "MOOSEConfig":
        """从配置字典创建 MOOSEConfig 实例"""
        params = config['parameters']
        sim = config['simulation']
        return cls(
            param_names=params['names'],
            param_paths=params.get('paths', params['names']),
            executable=sim.get('executable', 'ailuro-opt'),
            work_dir=Path(sim.get('work_dir', '.')),
            input_file=Path(sim.get('input_file', '')),
            output_dir=Path(sim.get('output_dir', 'results')),
            filebase_param=sim.get('filebase_param', 'out_name'),
            timeout=sim.get('timeout', 300),
            reference_csv=Path(sim.get('objective_csv', '')),
            csv_col_x=sim.get('result_col_x', 'disp'),
            csv_col_y=sim.get('result_col_y', 'force'),
            n_interp_points=sim.get('n_interp_points', 10)
        )

class MOOSEObjectiveFunction:
    """
    Interface to MOOSE-based simulations for optimization.
    """

    def __init__(self, config: MOOSEConfig, callback=None):
        """初始化评估器"""
        self.cfg = config
        self.cmd = [self.cfg.executable, "-i", str(self.cfg.input_file)]
        self.cfg.output_dir = self.cfg.work_dir / self.cfg.output_dir

        # create directories
        self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.cfg.output_dir)

        # check file existence
        if not self.cfg.input_file.exists():
            raise FileNotFoundError(f"MOOSE input file not found: {self.cfg.input_file}")
        if not self.cfg.reference_csv.exists():
            raise FileNotFoundError(f"Reference CSV file not found: {self.cfg.reference_csv}")
        
        self.callback = callback

        print(f"  MOOSE 程序: {self.cfg.executable}")
        print(f"  输入文件: {self.cfg.input_file}")
        print(f"  工作目录: {self.cfg.work_dir}")
        print(f"  输出目录: {self.cfg.output_dir}")
        print(f"  参考数据: {self.cfg.reference_csv}")

    def __call__(self, X: np.ndarray, eval_id:int) -> dict:

        """评估给定参数集的目标函数值"""
        # 调用 MOOSE 仿真并计算目标函数值

        # 输出文件名 run_0_p0_1000_p1_2000.csv
        out_name = f"run_{eval_id}_" + "_".join(
            [f"{name}_{X[i]:.2f}" for i, name in enumerate(self.cfg.param_names)]
        ) 

        file_base = self.cfg.output_dir / out_name
        sim_csv_file = Path(str(file_base) + '.csv')

        cmd = self.cmd + [f"{self.cfg.filebase_param}={file_base}"] + \
        [f"{path}={X[i]}" for i, path in enumerate(self.cfg.param_paths)]

        try:
            # 运行 MOOSE 仿真
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.cfg.timeout,
                text=True
            )

            if result.returncode == 0:
                objective, plot_data = self._calculate_objective(sim_csv_file)
                if self.callback is not None and plot_data is not None:
                    self.callback(self, sim_csv_file, objective, plot_data)
                status = 'success'  if np.isfinite(objective) else 'failed'
            else:
                objective = np.inf
                status = 'failed'
        except subprocess.TimeoutExpired:
            return np.inf,'timeout'

        # print(f"  运行命令: {' '.join(cmd)}")

        return objective,status
    
    def _calculate_objective(self,sim_csv: Path) -> tuple[float, dict]:
        """计算目标函数值
        
        Returns:
            tuple: (rmse, plot_data) where plot_data contains arrays for visualization
        """
        # 读取 sim_csv 并与 ref_csv 比较，计算误差
        if not sim_csv.exists():
            print(f"  ✗ 仿真结果文件未找到: {sim_csv}")
            return np.inf, None
        
        try:
            sim_data = pd.read_csv(sim_csv)
            ref_data = pd.read_csv(self.cfg.reference_csv)

            # 提取数据并排序（使用局部变量，避免多线程数据竞争）
            sim_x = sim_data[self.cfg.csv_col_x].values
            sim_y = sim_data[self.cfg.csv_col_y].values
            ref_x = ref_data[self.cfg.csv_col_x].values
            ref_y = ref_data[self.cfg.csv_col_y].values

            # 检查数据是否为空
            if len(sim_x) == 0 or len(ref_x) == 0:
                print(f"  ✗ 数据为空: sim={len(sim_x)}, ref={len(ref_x)}")
                return np.inf, None

            sim_sorted = np.argsort(sim_x)
            ref_sorted = np.argsort(ref_x)
            sim_x = sim_x[sim_sorted]
            sim_y = sim_y[sim_sorted]
            ref_x = ref_x[ref_sorted]
            ref_y = ref_y[ref_sorted]

            # 插值到统一点
            xmax = min(sim_x[-1], ref_x[-1])
            xmin = xmax / self.cfg.n_interp_points
            common_x = np.linspace(xmin, xmax, self.cfg.n_interp_points)
            sim_y_interp = np.interp(common_x, sim_x, sim_y)
            ref_y_interp = np.interp(common_x, ref_x, ref_y)
            
            # 计算均方根误差
            rmse = np.sqrt(np.mean((sim_y_interp - ref_y_interp) ** 2))
            
            # 准备绘图数据（用于回调函数）
            plot_data = {
                'sim_x': sim_x,
                'sim_y': sim_y,
                'ref_x': ref_x,
                'ref_y': ref_y,
                'common_x': common_x,
                'sim_y_interp': sim_y_interp,
                'ref_y_interp': ref_y_interp
            }
            
            return rmse, plot_data
        except Exception as e:
            print(f"  计算目标函数时出错: {e}")
            import traceback
            traceback.print_exc()
            return np.inf, None
