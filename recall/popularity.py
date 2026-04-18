# recall/popularity.py - 修复版
import pandas as pd
import numpy as np
import os

class Popularity:
    """热门电影召回 - 修复版"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Popularity, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._ratings_cache = None
        self._popular_movies_cache = None
        self._load_popular()
        self._initialized = True
    
    def _load_popular(self):
        """加载热门电影"""
        print("加载热门电影...")
        ratings = self._get_ratings()
        
        # 计算每部电影的平均分和评分人数
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'rating_mean']
        
        # 热门分数 = 评分人数 * 平均分
        movie_stats['popularity_score'] = movie_stats['rating_count'] * movie_stats['rating_mean']
        
        # 排序
        movie_stats = movie_stats.sort_values('popularity_score', ascending=False)
        
        self._popular_movies_cache = movie_stats.index.tolist()
        print(f"✅ 加载了 {len(self._popular_movies_cache)} 部热门电影")
    
    def recall(self, user_id=None, top_n=100):
        """返回热门电影"""
        return self._popular_movies_cache[:top_n]
    
    def _get_ratings(self):
        """获取评分数据（带简单缓存）"""
        if self._ratings_cache is None:
            print("加载评分数据...")
            self._ratings_cache = pd.read_parquet('processed/ratings.parquet')
        return self._ratings_cache