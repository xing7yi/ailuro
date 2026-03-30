from setuptools import setup, find_packages

# 直接定义版本号，避免导入 simuopt 包（会触发 numpy 依赖）
__version__ = '0.0.1'

setup(
    name='simu-opt',
    python_requires='>=3.5',
    version=__version__,
    description='Particle swarm optimization in Python',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
)