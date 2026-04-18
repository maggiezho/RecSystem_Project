"""
缓存工具模块
提供在不同环境下都能正常工作的缓存功能
"""

import pandas as pd
import functools
import os

# 检测是否在Streamlit环境下运行
def is_in_streamlit():
    """检测是否在Streamlit环境中运行"""
    # 方法1: 检查环境变量
    if os.environ.get('STREAMLIT_RUNNING') == 'false':
        return False
    
    # 方法2: 尝试导入streamlit并检查上下文
    try:
        import streamlit as st
        # 尝试访问streamlit的运行时
        if hasattr(st, 'runtime') and st.runtime:
            return True
        # 检查是否有script运行上下文
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except (ImportError, Exception):
        pass
    
    return False

IN_STREAMLIT = is_in_streamlit()

# 简单的内存缓存字典
_memory_cache = {}

def simple_cache(maxsize=128):
    """
    简单的内存缓存装饰器（用于非Streamlit环境）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
            
            # 检查缓存
            if cache_key in _memory_cache:
                return _memory_cache[cache_key]
            
            # 调用函数
            result = func(*args, **kwargs)
            
            # 存入缓存（如果缓存未满）
            if len(_memory_cache) < maxsize:
                _memory_cache[cache_key] = result
            
            return result
        return wrapper
    return decorator


def smart_cache(func):
    """
    智能缓存：在Streamlit环境下使用st.cache_data，否则使用简单内存缓存
    """
    if IN_STREAMLIT:
        # 在Streamlit环境下，使用st.cache_data
        try:
            import streamlit as st
            return st.cache_data(func)
        except (ImportError, Exception):
            return simple_cache()(func)
    else:
        # 在非Streamlit环境下，使用简单缓存
        return simple_cache()(func)


def clear_all_cache():
    """清除所有缓存"""
    global _memory_cache
    _memory_cache.clear()
    if IN_STREAMLIT:
        try:
            import streamlit as st
            st.cache_data.clear()
        except (ImportError, Exception):
            pass