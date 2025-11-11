import numpy as np
import matplotlib.pyplot as plt


def swift_model(epsilon, k,eps_0,n):
    """Swift模型计算应力"""
    return k * (epsilon + eps_0)**n


if __name__ == "__main__":
    
    fig, ax = plt.subplots(figsize=(8,6))
    epsilon = np.linspace(0, 0.5, 100)
    k = 1000  # 示例值
    eps_0 = 0.001  # 示例值
    n = 0.5  # 示例值
    stress = swift_model(epsilon, k, eps_0, n)
    ax.plot(epsilon, stress, label='Swift Model', color='blue')
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    ax.set_title('Swift Model Stress-Strain Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig('swift_model_curve.pdf')