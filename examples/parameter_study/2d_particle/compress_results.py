#!/usr/bin/env python3
"""
CSV/PNG 文件压缩和解压工具

功能:
- 自动查找目录中的 runner*.csv 文件
- 查找对应的 PNG 图像文件
- 将所有文件压缩成单个 zip 文件
- 解压 zip 文件到指定目录

用法:
    # 压缩当前目录的文件
    python compress_csv_files.py --mode zip
    
    # 压缩指定目录的文件
    python compress_csv_files.py --mode zip --dir ./voce_law --output voce_results.zip
    
    # 解压文件
    python compress_csv_files.py --mode unzip --input results.zip --output ./extracted
    
    # 压缩时包含特定模式的文件
    python compress_csv_files.py --mode zip --pattern "*_out_runner*.csv"
"""

import os
import re
import glob
import zipfile
import argparse
from pathlib import Path
from datetime import datetime


def find_runner_files(directory, pattern="*runner*.csv"):
    """
    查找所有匹配的 runner CSV 文件
    
    参数:
        directory: 搜索目录
        pattern: 文件匹配模式
    
    返回:
        list: 匹配的文件路径列表
    """
    search_path = Path(directory)
    
    # 支持多种常见的 runner 文件命名模式
    patterns = [
        pattern,
        "*_out_runner*.csv",
        "runner*.csv",
        "*runner*.csv"
    ]
    
    all_files = []
    for p in patterns:
        files = list(search_path.glob(p))
        all_files.extend(files)
    
    # 去重
    all_files = list(set(all_files))
    
    # 排序
    all_files.sort()
    
    return all_files


def find_png_files(directory, patterns=None):
    """
    查找所有 PNG 文件
    
    参数:
        directory: 搜索目录
        patterns: PNG 文件匹配模式列表
    
    返回:
        list: 匹配的 PNG 文件路径列表
    """
    search_path = Path(directory)
    
    if patterns is None:
        patterns = ["*.png", "**/*.png"]
    
    all_files = []
    for p in patterns:
        files = list(search_path.glob(p))
        all_files.extend(files)
    
    # 去重
    all_files = list(set(all_files))
    
    # 排序
    all_files.sort()
    
    return all_files


def find_related_files(directory, include_png=True, include_fit_results=True, 
                       include_plots=True, csv_pattern="*runner*.csv"):
    """
    查找所有相关文件
    
    参数:
        directory: 搜索目录
        include_png: 是否包含 PNG 文件
        include_fit_results: 是否包含拟合结果 CSV
        include_plots: 是否包含 fit_plots 目录
        csv_pattern: CSV 文件匹配模式
    
    返回:
        dict: 文件分类字典
    """
    files = {
        'csv': [],
        'png': [],
        'fit_results': [],
        'plots': []
    }
    
    search_path = Path(directory)
    
    # 查找 runner CSV 文件
    files['csv'] = find_runner_files(directory, csv_pattern)
    
    # 查找 PNG 文件
    if include_png:
        files['png'] = find_png_files(directory)
    
    # 查找拟合结果 CSV
    if include_fit_results:
        fit_results = list(search_path.glob("*_fit_params_*.csv"))
        fit_results.extend(search_path.glob("complete_results.csv"))
        files['fit_results'] = list(set(fit_results))
    
    # 查找 fit_plots 目录中的文件
    if include_plots:
        plots_dir = search_path / "fit_plots"
        if plots_dir.exists():
            files['plots'] = list(plots_dir.glob("**/*"))
            # 只保留文件，不包含目录
            files['plots'] = [f for f in files['plots'] if f.is_file()]
    
    return files


def compress_files(directory, output_file, include_png=True, include_fit_results=True,
                   include_plots=True, csv_pattern="*runner*.csv", verbose=True):
    """
    压缩文件到 zip 文件
    
    参数:
        directory: 源目录
        output_file: 输出的 zip 文件名
        include_png: 是否包含 PNG 文件
        include_fit_results: 是否包含拟合结果
        include_plots: 是否包含绘图结果
        csv_pattern: CSV 文件匹配模式
        verbose: 是否显示详细信息
    """
    directory = Path(directory)
    output_file = Path(output_file)
    
    # 查找所有相关文件
    files = find_related_files(directory, include_png, include_fit_results, 
                               include_plots, csv_pattern)
    
    # 统计文件数量
    total_files = sum(len(f) for f in files.values())
    
    if total_files == 0:
        print(f"错误: 在 {directory} 中没有找到任何文件")
        return False
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"压缩文件")
        print(f"{'='*70}")
        print(f"源目录: {directory}")
        print(f"输出文件: {output_file}")
        print(f"\n文件统计:")
        print(f"  Runner CSV 文件: {len(files['csv'])}")
        if include_png:
            print(f"  PNG 图像文件: {len(files['png'])}")
        if include_fit_results:
            print(f"  拟合结果文件: {len(files['fit_results'])}")
        if include_plots:
            print(f"  绘图文件: {len(files['plots'])}")
        print(f"  总计: {total_files} 个文件")
        print(f"\n开始压缩...")
    
    # 创建 zip 文件
    compressed_count = 0
    total_size = 0
    compressed_size = 0
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # 添加 CSV 文件
        for file_path in files['csv']:
            arcname = file_path.relative_to(directory)
            zipf.write(file_path, arcname)
            total_size += file_path.stat().st_size
            compressed_count += 1
            
            if verbose and compressed_count % 50 == 0:
                print(f"  已压缩 {compressed_count}/{total_files} 个文件...")
        
        # 添加 PNG 文件
        if include_png:
            for file_path in files['png']:
                arcname = file_path.relative_to(directory)
                zipf.write(file_path, arcname)
                total_size += file_path.stat().st_size
                compressed_count += 1
                
                if verbose and compressed_count % 50 == 0:
                    print(f"  已压缩 {compressed_count}/{total_files} 个文件...")
        
        # 添加拟合结果文件
        if include_fit_results:
            for file_path in files['fit_results']:
                arcname = file_path.relative_to(directory)
                zipf.write(file_path, arcname)
                total_size += file_path.stat().st_size
                compressed_count += 1
        
        # 添加绘图文件
        if include_plots:
            for file_path in files['plots']:
                arcname = file_path.relative_to(directory)
                zipf.write(file_path, arcname)
                total_size += file_path.stat().st_size
                compressed_count += 1
                
                if verbose and compressed_count % 50 == 0:
                    print(f"  已压缩 {compressed_count}/{total_files} 个文件...")
    
    # 获取压缩文件大小
    compressed_size = output_file.stat().st_size
    compression_ratio = (1 - compressed_size / total_size) * 100 if total_size > 0 else 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"压缩完成!")
        print(f"{'='*70}")
        print(f"  压缩文件: {output_file}")
        print(f"  文件数量: {compressed_count}")
        print(f"  原始大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"  压缩大小: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  压缩率: {compression_ratio:.1f}%")
        print(f"{'='*70}\n")
    
    return True


def decompress_files(input_file, output_dir, verbose=True):
    """
    解压 zip 文件
    
    参数:
        input_file: 输入的 zip 文件
        output_dir: 输出目录
        verbose: 是否显示详细信息
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    if not input_file.exists():
        print(f"错误: 文件 {input_file} 不存在")
        return False
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"解压文件")
        print(f"{'='*70}")
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
    
    # 解压文件
    with zipfile.ZipFile(input_file, 'r') as zipf:
        file_list = zipf.namelist()
        total_files = len(file_list)
        
        if verbose:
            print(f"  文件数量: {total_files}")
            print(f"\n开始解压...")
        
        extracted_count = 0
        for file_name in file_list:
            zipf.extract(file_name, output_dir)
            extracted_count += 1
            
            if verbose and extracted_count % 50 == 0:
                print(f"  已解压 {extracted_count}/{total_files} 个文件...")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"解压完成!")
        print(f"{'='*70}")
        print(f"  输出目录: {output_dir}")
        print(f"  文件数量: {extracted_count}")
        print(f"{'='*70}\n")
    
    return True


def list_archive_contents(input_file, verbose=True):
    """
    列出压缩文件内容
    
    参数:
        input_file: 输入的 zip 文件
        verbose: 是否显示详细信息
    """
    input_file = Path(input_file)
    
    if not input_file.exists():
        print(f"错误: 文件 {input_file} 不存在")
        return False
    
    with zipfile.ZipFile(input_file, 'r') as zipf:
        file_list = zipf.namelist()
        info_list = zipf.infolist()
        
        # 统计文件类型
        csv_files = [f for f in file_list if f.endswith('.csv')]
        png_files = [f for f in file_list if f.endswith('.png')]
        pdf_files = [f for f in file_list if f.endswith('.pdf')]
        other_files = [f for f in file_list if not any(f.endswith(ext) for ext in ['.csv', '.png', '.pdf'])]
        
        total_size = sum(info.file_size for info in info_list)
        compressed_size = sum(info.compress_size for info in info_list)
        
        print(f"\n{'='*70}")
        print(f"压缩文件内容: {input_file}")
        print(f"{'='*70}")
        print(f"\n文件统计:")
        print(f"  CSV 文件: {len(csv_files)}")
        print(f"  PNG 文件: {len(png_files)}")
        print(f"  PDF 文件: {len(pdf_files)}")
        print(f"  其他文件: {len(other_files)}")
        print(f"  总计: {len(file_list)} 个文件")
        print(f"\n大小信息:")
        print(f"  原始大小: {total_size / 1024 / 1024:.2f} MB")
        print(f"  压缩大小: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  压缩率: {(1 - compressed_size / total_size) * 100:.1f}%")
        
        if verbose:
            print(f"\n前 20 个文件:")
            for i, file_name in enumerate(file_list[:20]):
                info = info_list[i]
                print(f"  {file_name} ({info.file_size / 1024:.1f} KB)")
            
            if len(file_list) > 20:
                print(f"  ... 还有 {len(file_list) - 20} 个文件")
        
        print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='CSV/PNG 文件压缩和解压工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 压缩当前目录的所有 runner CSV 文件和 PNG 文件
  python compress_csv_files.py --mode zip
  
  # 压缩指定目录
  python compress_csv_files.py --mode zip --dir ./voce_law
  
  # 压缩指定目录并指定输出文件名
  python compress_csv_files.py --mode zip --dir ./voce_law --output voce_results.zip
  
  # 压缩但不包含 PNG 文件
  python compress_csv_files.py --mode zip --no-png
  
  # 压缩特定模式的 CSV 文件
  python compress_csv_files.py --mode zip --csv-pattern "*_out_runner*.csv"
  
  # 解压文件到当前目录
  python compress_csv_files.py --mode unzip --input results.zip
  
  # 解压文件到指定目录
  python compress_csv_files.py --mode unzip --input results.zip --dir ./extracted
  
  # 解压文件到指定目录（使用 --output，与 --dir 等效）
  python compress_csv_files.py --mode unzip --input results.zip --output ./extracted
  
  # 列出压缩文件内容
  python compress_csv_files.py --mode list --input results.zip
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['zip', 'unzip', 'list'],
                       help='操作模式: zip (压缩), unzip (解压), list (列出内容)')
    
    parser.add_argument('--dir', type=str, default=None,
                       help='工作目录 (压缩模式: 源目录; 解压模式: 输出目录; 默认: 当前目录)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件/目录 (压缩: .zip 文件名; 解压: 目标目录; 优先于 --dir)')
    
    parser.add_argument('--input', type=str, default=None,
                       help='输入文件 (用于解压/列出模式)')
    
    parser.add_argument('--csv-pattern', type=str, default='*runner*.csv',
                       help='CSV 文件匹配模式 (默认: *runner*.csv)')
    
    parser.add_argument('--no-png', action='store_true',
                       help='不包含 PNG 文件')
    
    parser.add_argument('--no-fit-results', action='store_true',
                       help='不包含拟合结果 CSV 文件')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='不包含 fit_plots 目录')
    
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式，只显示错误信息')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # 确定工作目录（默认为当前目录）
    work_dir = args.dir if args.dir is not None else '.'
    
    # 压缩模式
    if args.mode == 'zip':
        # 确定输出文件名
        if args.output is None:
            dir_name = Path(work_dir).resolve().name if work_dir != '.' else 'results'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{dir_name}_{timestamp}.zip"
        else:
            output_file = args.output
        
        # 确保输出文件有 .zip 扩展名
        if not output_file.endswith('.zip'):
            output_file += '.zip'
        
        if verbose:
            print(f"工作目录: {Path(work_dir).resolve()}")
        
        success = compress_files(
            directory=work_dir,
            output_file=output_file,
            include_png=not args.no_png,
            include_fit_results=not args.no_fit_results,
            include_plots=not args.no_plots,
            csv_pattern=args.csv_pattern,
            verbose=verbose
        )
        
        return 0 if success else 1
    
    # 解压模式
    elif args.mode == 'unzip':
        if args.input is None:
            print("错误: 解压模式需要指定 --input 参数")
            return 1
        
        # 确定输出目录 (优先级: --output > --dir > 自动生成)
        if args.output is not None:
            output_dir = args.output
        elif args.dir is not None:
            output_dir = work_dir
        else:
            # 使用压缩文件名作为目录名
            output_dir = Path(args.input).stem + '_extracted'
        
        if verbose:
            print(f"输出目录: {Path(output_dir).resolve()}")
        
        success = decompress_files(
            input_file=args.input,
            output_dir=output_dir,
            verbose=verbose
        )
        
        return 0 if success else 1
    
    # 列出内容模式
    elif args.mode == 'list':
        if args.input is None:
            print("错误: list 模式需要指定 --input 参数")
            return 1
        
        success = list_archive_contents(
            input_file=args.input,
            verbose=verbose
        )
        
        return 0 if success else 1


if __name__ == '__main__':
    exit(main())
