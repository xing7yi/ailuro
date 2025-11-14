import argparse
from pathlib import Path
import yaml
import json

# 导入新的优化器
from pso_optimizer import run_pso_optimization
from visualization import export_particles_to_vtk

def main():
    parser = argparse.ArgumentParser(description='PSO + MOOSE Optimization')
    parser.add_argument('--config', default='pso_config.yaml',
                       help='config file path (default: pso_config.yaml)')
    args = parser.parse_args()

    # Load configuration
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Error: config file does not exist: {config_file}")
        print(f"Please create the config file.")
        return

    print(f"Using config file: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Run Optimization (返回 OptimizationResultSummary 对象)
    result = run_pso_optimization(config)
    
    # 优化结果已经在优化器内部保存和打印
    print("\n✓ 优化完成！所有结果已保存。")

if __name__ == "__main__":
    main()
