import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.stats import qmc

def get_lhs_matrix(n_samples, n_dims, lb, ub):
    """生成拉丁超立方采样矩阵
    """
    sampler = qmc.LatinHypercube(d=n_dims,scramble=False) # 不使用扰动
    sample = sampler.random(n=n_samples)
    scaled_sample = qmc.scale(sample, lb, ub)
    return scaled_sample

def func_transformer(func, n_processes=0):
    """Return a wrapper that adapts `func` to the PSO expected signature.

    The returned function accepts an iterable/array ``X`` where each element
    is a parameter vector for a particle (shape: (n_particles, n_dims)). It
    returns ``(costs, statuses)`` where ``costs`` is a 1D numpy array and
    ``statuses`` is a list of status strings.

    The wrapping behaviour is selected from ``mode = getattr(func, 'mode', 'others')``.

    Supported modes:
      - 'common' / 'others': simple Python loop, call ``func(x)`` for each
        particle and collect results.
      - 'vectorization': assume ``func`` accepts the whole ``X`` and returns
        either ``costs`` or ``(costs, statuses)``.
      - 'multithreading': use ThreadPoolExecutor to evaluate particles in
        parallel threads.
      - 'multiprocessing': use ProcessPoolExecutor to evaluate in separate
        processes (requires picklable ``func`` and args).
      - 'joblib': use joblib.Parallel if available.
      - 'cached': treated like 'common' but is handy to mark functions that
        internally cache evaluations.

    Parameters
    ----------
    func : callable
        The original evaluator function. It may return either a single
        numeric cost (per particle) or a tuple ``(cost, status)``.
        If the function requires additional arguments (like eval_id),
        inspect the function signature to determine the number of arguments.
    n_processes : int
        Number of workers to use for parallel modes. 0 means auto-select.

    Notes
    -----
    - The wrapper will accept both list-of-vectors and a 2D numpy array.
    - If a per-particle call returns only a cost, the status will be set to
      the string 'success' by default.
    - If func accepts 2 arguments, the second argument is assumed to be eval_id.
    """

    mode = getattr(func, 'mode', 'others')
    
    # 检查函数签名以确定是否需要 eval_id
    import inspect
    sig = inspect.signature(func)
    n_func_params = len(sig.parameters)
    needs_eval_id = n_func_params >= 2

    mode = getattr(func, 'mode', 'others')

    def _normalize_result(r):
        # Normalize a single-call result to (cost, status)
        if isinstance(r, tuple) or isinstance(r, list) and len(r) == 2:
            return r[0], r[1]
        return r, 'success'

    def _handle_sequence_results(results):
        costs = []
        statuses = []
        for r in results:
            c, s = _normalize_result(r)
            costs.append(c)
            statuses.append(s)
        return np.asarray(costs), statuses

    def wrapper_common(X, eval_ids=None):
        results = []
        for i, x in enumerate(X):
            if needs_eval_id and eval_ids is not None:
                results.append(func(x, eval_ids[i]))
            else:
                results.append(func(x))
        return _handle_sequence_results(results)

    def wrapper_vectorized(X, eval_ids=None):
        # func is expected to accept the whole X
        if needs_eval_id and eval_ids is not None:
            res = func(X, eval_ids)
        else:
            res = func(X)
        if isinstance(res, tuple) or isinstance(res, list) and len(res) == 2:
            costs, statuses = res
            return np.asarray(costs), list(statuses)
        return np.asarray(res), ['success'] * (len(res) if hasattr(res, '__len__') else 1)

    def wrapper_thread(X, eval_ids=None):
        workers = None
        if n_processes and n_processes > 0:
            workers = n_processes
        else:
            workers = min(32, (len(X) or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            if needs_eval_id and eval_ids is not None:
                results = list(ex.map(func, X, eval_ids))
            else:
                results = list(ex.map(func, X))
        return _handle_sequence_results(results)

    def wrapper_process(X, eval_ids=None):
        workers = None
        if n_processes and n_processes > 0:
            workers = n_processes
        else:
            workers = max(1, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            if needs_eval_id and eval_ids is not None:
                results = list(ex.map(func, X, eval_ids))
            else:
                results = list(ex.map(func, X))
        return _handle_sequence_results(results)

    # Select wrapper based on mode and determine actual workers
    actual_workers = 1
    if mode == 'vectorization':
        func_transformed = wrapper_vectorized
        actual_workers = 1  # 向量化不使用多进程
    elif mode == 'multithreading':
        func_transformed = wrapper_thread
        actual_workers = n_processes if n_processes > 0 else min(32, os.cpu_count() or 1)
    elif mode == 'multiprocessing':
        func_transformed = wrapper_process
        actual_workers = n_processes if n_processes > 0 else (os.cpu_count() or 1)
    else:
        # default/common/others
        func_transformed = wrapper_common
        actual_workers = 1
    
    # 在返回的函数上附加并行信息（供 PSO 输出使用）
    func_transformed._parallel_info = {
        'mode': mode,
        'n_processes': actual_workers,
        'total_cores': os.cpu_count() or 1,
        'needs_eval_id': needs_eval_id
    }

    return func_transformed
