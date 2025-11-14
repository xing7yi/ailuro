"""
仿真缓存模块

提供基于 SQLite 和 KDTree 的智能缓存系统，用于加速重复和邻近参数的评估。
"""

import sqlite3
import json
import hashlib
import time
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree


class SimulationCache:
    """
    仿真结果缓存管理器
    
    特性:
    1. 参数归一化到 [0,1] 空间
    2. 精确匹配（基于哈希）
    3. 最近邻查找和插值（基于 KDTree）
    4. SQLite 持久化存储
    5. 多进程安全（通过 SQLite 文件锁）
    """
    
    def __init__(self, cache_file, lower_bounds, upper_bounds, 
                 exact_decimals=4, near_tol=0.01, interp_method='nearest',
                 max_neighbors=5, enabled=True, version='1.0'):
        """
        Args:
            cache_file: SQLite 数据库文件路径
            lower_bounds: 参数下界数组
            upper_bounds: 参数上界数组
            exact_decimals: 精确匹配的小数位数（用于哈希）
            near_tol: 最近邻匹配的容差（归一化空间中的欧氏距离）
            interp_method: 插值方法 ('nearest', 'weighted', 'none')
            max_neighbors: 用于插值的最大邻居数
            enabled: 是否启用缓存
            version: 缓存版本号（用于兼容性检查）
        """
        self.enabled = enabled
        if not enabled:
            return
            
        self.cache_file = Path(cache_file)
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.exact_decimals = exact_decimals
        self.near_tol = near_tol
        self.interp_method = interp_method
        self.max_neighbors = max_neighbors
        self.version = version
        
        # 内存缓存：精确匹配哈希表
        # 注意：在多进程环境下，每个进程有独立的内存缓存副本
        # SQLite 数据库本身提供了进程间的同步机制
        self.exact_cache = {}  # {hash: result}
        
        # KDTree 用于最近邻查找（延迟构建）
        self.kdtree = None
        self.kdtree_params = []  # 归一化参数列表
        self.kdtree_results = []  # 对应的结果列表
        
        # 统计信息
        self.stats = {
            'exact_hits': 0,
            'near_hits': 0,
            'misses': 0,
            'interpolations': 0
        }
        
        # 初始化数据库
        self._init_database()
        
        # 从数据库加载到内存
        self._load_from_database()
    
    def _init_database(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(str(self.cache_file))
        cursor = conn.cursor()
        
        # 创建缓存表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                param_hash TEXT UNIQUE NOT NULL,
                params_norm TEXT NOT NULL,
                params_orig TEXT NOT NULL,
                objective REAL NOT NULL,
                result_data TEXT,
                metadata TEXT,
                timestamp REAL NOT NULL,
                version TEXT NOT NULL
            )
        ''')
        
        # 创建索引加速查询
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON cache(param_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_version ON cache(version)')
        
        conn.commit()
        conn.close()
    
    def _load_from_database(self):
        """从数据库加载所有缓存到内存"""
        conn = sqlite3.connect(str(self.cache_file))
        cursor = conn.cursor()
        
        # 只加载当前版本的缓存
        cursor.execute('SELECT param_hash, params_norm, objective, result_data FROM cache WHERE version = ?',
                      (self.version,))
        
        rows = cursor.fetchall()
        conn.close()
        
        for param_hash, params_norm_str, objective, result_data in rows:
            params_norm = np.array(json.loads(params_norm_str))
            result = {
                'objective': objective,
                'params_norm': params_norm,
                'result_data': json.loads(result_data) if result_data else None
            }
            
            # 加载到精确匹配缓存
            self.exact_cache[param_hash] = result
            
            # 加载到 KDTree 数据
            self.kdtree_params.append(params_norm)
            self.kdtree_results.append(result)
        
        # 构建 KDTree
        if self.kdtree_params:
            self.kdtree = KDTree(np.array(self.kdtree_params))
            print(f"  缓存已加载: {len(self.kdtree_params)} 条记录")
    
    def _normalize_params(self, params):
        """将参数归一化到 [0, 1]"""
        return (params - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
    
    def _compute_hash(self, params_norm):
        """计算归一化参数的哈希（精确到指定小数位）"""
        # 四舍五入到指定小数位
        params_rounded = np.round(params_norm, self.exact_decimals)
        # 转换为字符串并哈希
        params_str = ','.join([f'{p:.{self.exact_decimals}f}' for p in params_rounded])
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def query(self, params):
        """
        查询缓存
        
        Returns:
            tuple: (found, result, source)
                found: bool, 是否找到
                result: dict, 结果（包含 objective, result_data 等）
                source: str, 来源 ('exact', 'nearest', 'interpolated', None)
        """
        if not self.enabled:
            return False, None, None
        
        # 归一化参数
        params_norm = self._normalize_params(params)
        
        # 1. 精确匹配
        param_hash = self._compute_hash(params_norm)
        if param_hash in self.exact_cache:
            self.stats['exact_hits'] += 1
            return True, self.exact_cache[param_hash].copy(), 'exact'
        
        # 2. 最近邻查找
        if self.kdtree is not None and len(self.kdtree_params) > 0:
            # 查找最近的 k 个邻居
            k = min(self.max_neighbors, len(self.kdtree_params))
            distances, indices = self.kdtree.query(params_norm, k=k)
            
            # 如果只有一个邻居
            if k == 1:
                distances = [distances]
                indices = [indices]
            
            # 检查最近邻是否在容差范围内
            min_dist = distances[0]
            if min_dist < self.near_tol:
                self.stats['near_hits'] += 1
                
                if self.interp_method == 'nearest':
                    # 直接返回最近邻
                    return True, self.kdtree_results[indices[0]].copy(), 'nearest'
                
                elif self.interp_method == 'weighted':
                    # 距离加权插值
                    valid_neighbors = [(d, i) for d, i in zip(distances, indices) if d < self.near_tol]
                    
                    if not valid_neighbors:
                        self.stats['misses'] += 1
                        return False, None, None
                    
                    # 计算权重（距离倒数）
                    weights = np.array([1.0 / (d + 1e-10) for d, _ in valid_neighbors])
                    weights /= weights.sum()
                    
                    # 加权平均目标值
                    objective_interp = sum(w * self.kdtree_results[i]['objective'] 
                                          for w, (_, i) in zip(weights, valid_neighbors))
                    
                    self.stats['interpolations'] += 1
                    result = {
                        'objective': objective_interp,
                        'params_norm': params_norm,
                        'result_data': None  # 插值结果没有详细数据
                    }
                    return True, result, 'interpolated'
        
        # 3. 未找到
        self.stats['misses'] += 1
        return False, None, None
    
    def store(self, params, objective, result_data=None, metadata=None):
        """
        存储评估结果到缓存
        
        Args:
            params: 原始参数
            objective: 目标函数值
            result_data: 详细结果数据（可选，如力-位移曲线）
            metadata: 元数据（可选，如 solver 设置）
        """
        if not self.enabled:
            return
        
        # 归一化参数
        params_norm = self._normalize_params(params)
        param_hash = self._compute_hash(params_norm)
        
        result = {
            'objective': objective,
            'params_norm': params_norm,
            'result_data': result_data
        }
        
        # 存储到内存缓存
        # 注意：在多进程环境下，每个进程的内存缓存是独立的
        # 但所有进程共享同一个 SQLite 数据库
        if param_hash not in self.exact_cache:
            self.exact_cache[param_hash] = result
            self.kdtree_params.append(params_norm)
            self.kdtree_results.append(result)
            
            # 重建 KDTree
            self.kdtree = KDTree(np.array(self.kdtree_params))
        
        # 存储到数据库（所有进程共享）
        self._store_to_database(param_hash, params, params_norm, objective, result_data, metadata)
    
    def _store_to_database(self, param_hash, params, params_norm, objective, result_data, metadata):
        """存储到 SQLite 数据库"""
        try:
            conn = sqlite3.connect(str(self.cache_file), timeout=10.0)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache 
                (param_hash, params_norm, params_orig, objective, result_data, metadata, timestamp, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                param_hash,
                json.dumps(params_norm.tolist()),
                json.dumps(params.tolist()),
                objective,
                json.dumps(result_data) if result_data else None,
                json.dumps(metadata) if metadata else None,
                time.time(),
                self.version
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  警告: 缓存写入数据库失败: {e}")
    
    def get_stats(self):
        """获取缓存统计信息（从数据库读取实际大小）"""
        if not self.enabled:
            return {
                'enabled': False,
                'exact_hits': 0,
                'near_hits': 0,
                'misses': 0,
                'interpolations': 0,
                'total_queries': 0,
                'hit_rate': 0.0,
                'cache_size': 0
            }
        
        # 从数据库获取实际缓存大小
        try:
            conn = sqlite3.connect(str(self.cache_file))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM cache WHERE version = ?', (self.version,))
            db_cache_size = cursor.fetchone()[0]
            conn.close()
        except:
            db_cache_size = len(self.exact_cache)
        
        total_queries = self.stats['exact_hits'] + self.stats['near_hits'] + self.stats['misses']
        if total_queries == 0:
            hit_rate = 0.0
        else:
            hit_rate = (self.stats['exact_hits'] + self.stats['near_hits']) / total_queries * 100
        
        return {
            **self.stats,
            'enabled': True,
            'total_queries': total_queries,
            'hit_rate': hit_rate,
            'cache_size': db_cache_size  # 使用数据库中的实际大小
        }
    
    def print_stats(self, eval_history):
        """
        打印缓存统计信息
        
        Args:
            eval_history: 评估历史记录列表（可选）
                如果提供，将从历史记录中统计缓存使用情况
        """
        if not self.enabled:
            print("\n" + "="*60)
            print("缓存统计信息")
            print("="*60)
            print("  缓存系统未启用")
            print("="*60)
            return
        
        if eval_history:
            # 从评估历史中统计缓存命中情况
            cache_hits = sum(1 for record in eval_history if record['status'] in ['C', 'N', 'I'])
            exact_hits = sum(1 for record in eval_history if record['status'] == 'C')
            near_hits = sum(1 for record in eval_history if record['status'] == 'N')
            interp_hits = sum(1 for record in eval_history if record['status'] == 'I')
            total = len(eval_history)
            hit_rate = (cache_hits / total * 100) if total > 0 else 0.0
            
            # 从数据库获取缓存大小
            try:
                conn = sqlite3.connect(str(self.cache_file))
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM cache WHERE version = ?', (self.version,))
                cache_size = cursor.fetchone()[0]
                conn.close()
            except Exception as e:
                cache_size = 0
                print(f"  警告: 读取缓存数据库失败: {e}")
            
            print("\n" + "="*60)
            print("缓存统计信息（从评估历史）")
            print("="*60)
            print(f"  数据库缓存:   {cache_size} 条")
            print(f"  总评估次数:   {total}")
            print(f"  精确命中:     {exact_hits} (C)")
            print(f"  邻近命中:     {near_hits} (N)")
            print(f"  插值命中:     {interp_hits} (I)")
            print(f"  新评估:       {total - cache_hits}")
            print(f"  命中率:       {hit_rate:.2f}%")
            print("="*60)
        else:
            # print error
            print("No evaluation history provided for cache statistics.")
