import os
import yaml
import csv
from pathlib import Path
import re

def extract_pso_parameters():
    # 获取当前目录
    current_dir = Path.cwd()
    
    # 存储结果
    results = []
    
    # 遍历所有子目录
    for subdir in current_dir.iterdir():
        if subdir.is_dir():
            yaml_file = subdir / 'pso_optimal_parameters.yaml'
            
            if yaml_file.exists():
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    # 提取p0-p3参数（位于optimal_parameters下）
                    optimal_params = data.get('optimal_parameters', {})
                    row = {
                        'name': 'D' + re.search(r'D(\d+\.\d+)', subdir.name).group(1),
                        'p0': round(optimal_params.get('p0'),2),
                        'p1': round(optimal_params.get('p1'),2),
                        'p2': round(optimal_params.get('p2'),2),
                        'p3': round(optimal_params.get('p3'),2),
                    }
                    results.append(row)
                except Exception as e:
                    print(f"Error reading {yaml_file}: {e}")
    
    # 保存到CSV文件
    if results:
        csv_file = current_dir / 'pso_parameters.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'p0', 'p1', 'p2', 'p3'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to {csv_file}")
        print(f"Total directories processed: {len(results)}")
    else:
        print("No pso_optimal_parameters.yaml files found")

if __name__ == '__main__':
    extract_pso_parameters()