#!/usr/bin/env python3
"""
最简 PSO 测试脚本 - 验证基础功能
不使用配置文件，直接测试 Rastrigin 函数优化
"""

import numpy as np
from pyswarms.single.global_best import GlobalBestPSO

def rastrigin(x):
    """
    Rastrigin 函数 - 标准测试函数
    全局最小值在 x = [0, 0, 0] 处，f(x) = 0
    """
    A = 10
    x = np.asarray(x)
    n = len(x[0]) if x.ndim > 1 else len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=-1)

def main():
    print("="*60)
    print("PSO 基础功能测试 - Rastrigin 函数优化")
    print("="*60)
    
    # 问题设置
    n_dimensions = 3  # 3维问题
    bounds = (
        np.array([-5.12, -5.12, -5.12]),  # 下界
        np.array([5.12, 5.12, 5.12])      # 上界
    )
    
    # PSO 参数
    n_particles = 20      # 粒子数量（小规模测试）
    max_iters = 500        # 最大迭代次数
    
    options = {
        'c1': 1.5,        # 认知参数
        'c2': 1.5,        # 社会参数
        'w': 0.7          # 惯性权重
    }
    
    print(f"\n问题设置:")
    print(f"  维度: {n_dimensions}")
    print(f"  边界: {bounds[0]} 到 {bounds[1]}")
    print(f"  理论最优解: [0, 0, 0]")
    print(f"  理论最优值: 0")
    
    print(f"\nPSO 参数:")
    print(f"  粒子数: {n_particles}")
    print(f"  最大迭代: {max_iters}")
    print(f"  认知参数 c1: {options['c1']}")
    print(f"  社会参数 c2: {options['c2']}")
    print(f"  惯性权重 w: {options['w']}")
    
    # 创建优化器
    print(f"\n开始优化...")
    print("-"*60)
    
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_dimensions,
        options=options,
        bounds=bounds,
        ftol=1e-7,
        ftol_iter=30
    )
    
    # 执行优化
    import time
    start_time = time.time()
    
    cost, pos = optimizer.optimize(rastrigin, iters=max_iters)
    
    elapsed_time = time.time() - start_time
    
    # 输出结果
    print("-"*60)
    print(f"\n优化结果:")
    print(f"  最优目标值: {cost:.6e}")
    print(f"  最优位置: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    print(f"  计算时间: {elapsed_time:.3f} 秒")
    print(f"  总函数评估: {n_particles * (max_iters + 1)}")
    
    # 评估结果
    print(f"\n结果评估:")
    if cost < 0.1:
        print("  ✓ 优化成功！找到了接近全局最优的解")
    elif cost < 1.0:
        print("  ○ 优化较好，但可能需要更多迭代")
    else:
        print("  ✗ 优化结果不理想，建议增加粒子数或迭代次数")
    
    # 可视化收敛曲线
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(optimizer.cost_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Best Objective Value', fontsize=12)
        ax.set_title('PSO Convergence Curve - Rastrigin Function', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('pso_test_convergence.png', dpi=150)
        print(f"\n✓ 收敛曲线已保存: pso_test_convergence.png")
        
    except Exception as e:
        print(f"\n✗ 绘图失败: {e}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

if __name__ == "__main__":
    main()
