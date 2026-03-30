#!/usr/bin/env python3
"""
Setup script for Ailuro Python utilities
"""
from setuptools import setup, find_packages

setup(
    name='ailuro-utils',
    version='1.0.0',
    description='Ailuro Python utility scripts',
    author='Ailuro Team',
    packages=find_packages(),
    scripts=[
        'realtime_plotter.py',
        'analyse_params.py',
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'scipy',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'ailuro-plot=realtime_plotter:main',
            'responsefit=responsefit.cli:main',
            'responsefit-analyse=analyse_params:main',
        ],
    },
    python_requires='>=3.6',
)
