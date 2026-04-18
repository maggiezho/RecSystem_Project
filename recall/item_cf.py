import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_utils import smart_cache


class ItemCF:
    """基于物品的协同过滤召回"""
    
    _instance = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ItemCF, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path='models/item_similarity.pkl'):
        if hasattr(self, '_initialized'):
            return
        
        self.model_path = model_path
        self.sim_matrix = None
        self.item_map = None
        self.reverse_item_map = None
        self._load_model()
        self._initialized = True
    
    def _load_model(self):
        """加载预训练的相似度矩阵"""
        if not os.path.exists(self.model_path):
            if not ItemCF._loaded:
                print(f"⚠️ ItemCF模型不存在: {self.model_path}")
                print("   请先运行 build_item_cf() 训练模型")
            self.sim_matrix = None
            return
        
        # 只在第一次加载时打印
        if not ItemCF._loaded:
            print(f"正在加载 ItemCF 模型...")
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.sim_matrix = data['matrix']
            self.item_map = data['map']
            self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        if not ItemCF._loaded:
            print(f"✅ ItemCF模型加载成功，覆盖 {len(self.item_map)} 部电影")
            ItemCF._loaded = True
    
    def recall(self, user_id, top_n=100, user_history=None):
        """为用户生成召回候选集"""
        if self.sim_matrix is None:
            return []
        
        ratings = self._get_ratings()
        
        # 找到该用户评分最高的前5部电影
        if user_history is not None:
            user_movie_ids = list(user_history)[:5]
        else:
            user_ratings = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False).head(5)
            user_movie_ids = user_ratings['movieId'].tolist()
        
        if len(user_movie_ids) == 0:
            return []
        
        # 基于相似度矩阵召回
        recommendations = {}
        
        for m_id in user_movie_ids:
            if m_id not in self.reverse_item_map:
                continue
                
            idx = self.reverse_item_map[m_id]
            sim_scores = self.sim_matrix[idx].toarray().flatten()
            similar_indices = sim_scores.argsort()[-(top_n+5):-1][::-1]
            
            for i in similar_indices:
                movie_id = self.item_map[i]
                if movie_id not in user_movie_ids:
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + sim_scores[i]
        
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [x[0] for x in sorted_recs]
    
    @smart_cache
    def _get_ratings(self):
        """获取评分数据（带缓存）"""
        return pd.read_parquet('processed/ratings.parquet')


def build_item_cf():
    """训练ItemCF模型"""
    print("加载处理后的数据...")
    ratings = pd.read_parquet('processed/ratings.parquet')
    
    popular_movies = ratings['movieId'].value_counts().head(10000).index
    ratings_tiny = ratings[ratings['movieId'].isin(popular_movies)]
    
    print("正在构建稀疏矩阵...")
    user_u_col = ratings_tiny['userId'].astype('category').cat.codes
    item_i_col = ratings_tiny['movieId'].astype('category').cat.codes
    
    item_map = dict(enumerate(ratings_tiny['movieId'].astype('category').cat.categories))
    
    row = item_i_col.values
    col = user_u_col.values
    data = ratings_tiny['rating'].values
    
    sparse_item_user = csr_matrix((data, (row, col)))
    
    print("正在计算电影相似度矩阵（余弦相似度）...")
    item_similarity = cosine_similarity(sparse_item_user, dense_output=False)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/item_similarity.pkl', 'wb') as f:
        pickle.dump({'matrix': item_similarity, 'map': item_map}, f)
    
    print("✅ 召回层模型已保存至 'models/item_similarity.pkl'")


if __name__ == "__main__":
    build_item_cf()