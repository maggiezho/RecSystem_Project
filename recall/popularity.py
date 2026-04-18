import pandas as pd

class Popularity:
    """热门电影召回（用于冷启动）"""
    
    def __init__(self):
        self.popular_movies = None
        self._load_popular()
    
    def _load_popular(self):
        """加载热门电影（基于评分数量和平均分）"""
        ratings = pd.read_parquet('processed/ratings.parquet')
        
        # 计算热度分数 = 评分人数 * 平均分
        stats = ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        stats.columns = ['rating_count', 'rating_mean']
        
        # 热度分：需要至少50个评分
        stats['popularity_score'] = stats['rating_count'] * stats['rating_mean']
        stats = stats[stats['rating_count'] >= 50]
        
        # 按热度排序
        stats = stats.sort_values('popularity_score', ascending=False)
        self.popular_movies = stats.index.tolist()
        print(f"✅ 热门电影加载完成，共 {len(self.popular_movies)} 部")
    
    def recall(self, user_id=None, top_n=100):
        """
        返回热门电影
        user_id参数保留是为了接口统一，实际不用
        """
        return self.popular_movies[:top_n]