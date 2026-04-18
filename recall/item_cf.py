import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    """基于物品的协同过滤召回"""
    
    def __init__(self, model_path='models/item_similarity.pkl'):
        self.model_path = model_path
        self.sim_matrix = None
        self.item_map = None
        self.reverse_item_map = None
        self._load_model()
        
    def _load_model(self):
        """加载预训练的相似度矩阵"""
        if not os.path.exists(self.model_path):
            print(f"⚠️ ItemCF模型不存在: {self.model_path}")
            print("   请先运行 build_item_cf() 训练模型")
            self.sim_matrix = None
            return
            
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.sim_matrix = data['matrix']
            self.item_map = data['map']  # idx -> movieId
            self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        print(f"✅ ItemCF模型加载成功，覆盖 {len(self.item_map)} 部电影")
    
    def recall(self, user_id, top_n=100):
        """
        为用户生成召回候选集
        返回: List[movieId]
        """
        if self.sim_matrix is None:
            return []
            
        # 加载数据
        ratings = pd.read_parquet('processed/ratings.parquet')
        
        # 1. 找到该用户评分最高的前5部电影
        user_ratings = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False).head(5)
        user_movie_ids = user_ratings['movieId'].tolist()
        
        if len(user_movie_ids) == 0:
            return []  # 新用户，无历史
        
        # 2. 基于相似度矩阵召回
        recommendations = {}
        
        for m_id in user_movie_ids:
            if m_id not in self.reverse_item_map:
                continue  # 该电影不在相似度矩阵中
                
            idx = self.reverse_item_map[m_id]
            sim_scores = self.sim_matrix[idx].toarray().flatten()
            # 获取相似度最高的前 top_n+5 个
            similar_indices = sim_scores.argsort()[-(top_n+5):-1][::-1]
            
            for i in similar_indices:
                movie_id = self.item_map[i]
                if movie_id not in user_movie_ids:
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + sim_scores[i]
        
        # 排序返回
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [x[0] for x in sorted_recs]