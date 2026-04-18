import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class UserCF:
    """基于用户的协同过滤召回"""
    
    def __init__(self, model_path='models/user_similarity.pkl'):
        self.model_path = model_path
        self.sim_matrix = None
        self.user_map = None
        self._check_and_train()
        
    def _check_and_train(self):
        """检查模型是否存在，不存在则训练"""
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            print("⚠️ UserCF模型不存在，开始训练...")
            self._train()
            self._load_model()
    
    def _train(self):
        """训练UserCF模型"""
        print("开始训练UserCF模型...")
        ratings = pd.read_parquet('processed/ratings.parquet')
        
        # 取前5000名活跃用户（节省内存）
        active_users = ratings['userId'].value_counts().head(5000).index
        ratings_tiny = ratings[ratings['userId'].isin(active_users)]
        
        # 只保留热门电影（被评分次数>50）
        popular_movies = ratings_tiny['movieId'].value_counts()
        popular_movies = popular_movies[popular_movies > 50].index
        ratings_tiny = ratings_tiny[ratings_tiny['movieId'].isin(popular_movies)]
        
        print(f"   用户数: {ratings_tiny['userId'].nunique()}, 电影数: {ratings_tiny['movieId'].nunique()}")
        
        # 构建用户-物品矩阵
        user_codes = ratings_tiny['userId'].astype('category').cat.codes
        item_codes = ratings_tiny['movieId'].astype('category').cat.codes
        
        self.user_map = dict(enumerate(ratings_tiny['userId'].astype('category').cat.categories))
        reverse_user_map = {v: k for k, v in self.user_map.items()}
        
        # 构建稀疏矩阵 (用户 x 物品)
        row = user_codes.values
        col = item_codes.values
        data = ratings_tiny['rating'].values
        sparse_user_item = csr_matrix((data, (row, col)))
        
        # 计算用户相似度
        print("   计算用户相似度矩阵...")
        user_similarity = cosine_similarity(sparse_user_item, dense_output=False)
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'matrix': user_similarity,
                'map': self.user_map,
                'reverse_map': reverse_user_map
            }, f)
        print(f"✅ UserCF模型已保存至 {self.model_path}")
    
    def _load_model(self):
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.sim_matrix = data['matrix']
            self.user_map = data['map']  # idx -> userId
            self.reverse_user_map = data['reverse_map']  # userId -> idx
        print(f"✅ UserCF模型加载成功，覆盖 {len(self.user_map)} 个用户")
    
    def recall(self, user_id, top_n=100):
        """基于相似用户召回电影"""
        if self.sim_matrix is None:
            return []
        
        # 检查用户是否在模型中
        if user_id not in self.reverse_user_map:
            return []
        
        user_idx = self.reverse_user_map[user_id]
        
        # 找到最相似的K个用户
        sim_scores = self.sim_matrix[user_idx].toarray().flatten()
        similar_users = sim_scores.argsort()[-(20+1):-1][::-1]  # 取前20个相似用户
        
        # 加载评分数据
        ratings = pd.read_parquet('processed/ratings.parquet')
        
        # 收集相似用户喜欢但当前用户没看过的电影
        recommendations = {}
        user_watched = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
        
        for sim_user_idx in similar_users:
            sim_user_id = self.user_map[sim_user_idx]
            sim_score = sim_scores[sim_user_idx]
            
            # 获取该用户评分高的电影
            user_movies = ratings[ratings['userId'] == sim_user_id]
            liked_movies = user_movies[user_movies['rating'] >= 4]['movieId'].tolist()
            
            for movie_id in liked_movies:
                if movie_id not in user_watched:
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + sim_score
        
        # 排序返回
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [x[0] for x in sorted_recs]