# recall/user_cf.py
"""
基于用户的协同过滤召回 - 课设优化版
特点：
1. 内存优化：限制用户和电影数量
2. 智能采样：选择最有代表性的用户
3. 降级策略：保证推荐覆盖率
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class UserCF:
    """基于用户的协同过滤召回 - 课设优化版"""
    
    _instance = None
    _loaded = False
    
    # 类级别的缓存
    _ratings_cache = None
    _movies_cache = None
    _rating_stats_cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserCF, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_path='models/user_similarity.pkl'):
        if hasattr(self, '_initialized'):
            return
        
        self.model_path = model_path
        self.sim_matrix = None
        self.user_indices = None
        self.user_map = None
        self.reverse_user_map = None
        self.user_means = None
        self._check_and_train()
        self._initialized = True
    
    def _check_and_train(self):
        """检查模型是否存在，不存在则训练"""
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            if not UserCF._loaded:
                print("⚠️ UserCF模型不存在，开始训练...")
            self._train()
            self._load_model()
    
    def _train(self):
        """训练UserCF模型 - 课设优化版（内存友好）"""
        if not UserCF._loaded:
            print("=" * 50)
            print("开始训练UserCF模型（课设优化版）")
            print("=" * 50)
        
        ratings = self._get_ratings()
        
        # ========== 1. 智能用户选择策略 ==========
        # 计算用户统计信息
        user_stats = ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        user_stats.columns = ['rating_count', 'rating_mean', 'rating_std']
        
        # 筛选条件：评分数量在50-500之间（活跃但不过度）
        user_stats = user_stats[
            (user_stats['rating_count'] >= 50) & 
            (user_stats['rating_count'] <= 500)
        ]
        
        if not UserCF._loaded:
            print(f"📊 用户统计:")
            print(f"   候选用户数: {len(user_stats)}")
            print(f"   平均评分数量: {user_stats['rating_count'].mean():.1f}")
            print(f"   平均评分标准差: {user_stats['rating_std'].mean():.3f}")
        
        # 选择最有代表性的用户（评分方差大 × 评分数量适中）
        # 这样的用户能提供更多样化的兴趣信息
        user_stats['representative_score'] = user_stats['rating_std'] * np.log1p(user_stats['rating_count'])
        
        # 限制最大用户数（课设友好：2000个用户）
        MAX_USERS = 2000
        if len(user_stats) > MAX_USERS:
            selected_users = user_stats.nlargest(MAX_USERS, 'representative_score').index.tolist()
            if not UserCF._loaded:
                print(f"🎯 选择最具代表性的 {MAX_USERS} 个用户")
        else:
            selected_users = user_stats.index.tolist()
        
        # 过滤数据
        ratings_filtered = ratings[ratings['userId'].isin(selected_users)]
        
        # ========== 2. 智能电影选择策略 ==========
        movie_stats = ratings_filtered.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).round(2)
        movie_stats.columns = ['rating_count', 'rating_mean']
        
        # 选择热门电影（至少被10个用户评分）
        movie_stats = movie_stats[movie_stats['rating_count'] >= 10]
        
        # 限制最大电影数（课设友好：3000部电影）
        MAX_MOVIES = 3000
        if len(movie_stats) > MAX_MOVIES:
            # 选择评分次数最多的电影
            selected_movies = movie_stats.nlargest(MAX_MOVIES, 'rating_count').index.tolist()
            if not UserCF._loaded:
                print(f"🎬 选择最热门的 {MAX_MOVIES} 部电影")
        else:
            selected_movies = movie_stats.index.tolist()
        
        ratings_filtered = ratings_filtered[ratings_filtered['movieId'].isin(selected_movies)]
        
        if not UserCF._loaded:
            print(f"\n📊 训练数据统计:")
            print(f"   用户数: {len(selected_users)}")
            print(f"   电影数: {len(selected_movies)}")
            print(f"   评分记录: {len(ratings_filtered)}")
            print(f"   稀疏度: {len(ratings_filtered) / (len(selected_users) * len(selected_movies)):.2%}")
        
        # ========== 3. 构建用户-电影评分矩阵 ==========
        # 创建用户ID映射
        unique_users = ratings_filtered['userId'].unique()
        self.user_map = {idx: user_id for idx, user_id in enumerate(unique_users)}
        self.reverse_user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        
        # 创建电影ID映射
        unique_movies = ratings_filtered['movieId'].unique()
        movie_map = {idx: movie_id for idx, movie_id in enumerate(unique_movies)}
        reverse_movie_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        
        # 构建稀疏矩阵
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        rows = []
        cols = []
        data = []
        
        for _, row in ratings_filtered.iterrows():
            user_idx = self.reverse_user_map[row['userId']]
            movie_idx = reverse_movie_map[row['movieId']]
            rows.append(user_idx)
            cols.append(movie_idx)
            data.append(row['rating'])
        
        user_movie_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))
        
        if not UserCF._loaded:
            print(f"\n📐 矩阵形状: {user_movie_matrix.shape}")
            print(f"   非零元素: {user_movie_matrix.nnz:,}")
        
        # ========== 4. 评分归一化（减去用户平均分）==========
        user_means = np.array(user_movie_matrix.mean(axis=1)).flatten()
        self.user_means = {self.user_map[i]: user_means[i] for i in range(n_users) if user_means[i] > 0}
        
        # 中心化矩阵
        user_movie_centered = user_movie_matrix.copy()
        for i in range(n_users):
            if user_means[i] > 0:
                row_start = user_movie_centered.indptr[i]
                row_end = user_movie_centered.indptr[i+1]
                user_movie_centered.data[row_start:row_end] -= user_means[i]
        
        # ========== 5. 分批计算用户相似度（内存优化）==========
        if not UserCF._loaded:
            print("\n🔄 计算用户相似度矩阵（分批处理）...")

        BATCH_SIZE = 200  # 每批处理200个用户
        sim_matrix_full = np.zeros((n_users, n_users))  # 初始化完整矩阵

        for start_idx in range(0, n_users, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, n_users)
            batch_indices = list(range(start_idx, end_idx))
            
            # 计算当前批次用户的相似度
            batch_sim = cosine_similarity(user_movie_centered[batch_indices])
            
            # 🔧 修复：将批次结果放入完整矩阵的正确位置
            batch_size_actual = len(batch_indices)
            sim_matrix_full[start_idx:end_idx, start_idx:end_idx] = batch_sim
            
            if not UserCF._loaded and (start_idx % 1000 == 0):
                progress = (end_idx / n_users) * 100
                print(f"   进度: {end_idx}/{n_users} ({progress:.1f}%)")

        sim_matrix = sim_matrix_full

        if not UserCF._loaded:
            print(f"✅ 相似度矩阵计算完成，形状: {sim_matrix.shape}")
        
        # 保存所有用户索引
        user_indices = np.arange(n_users)
        
        # ========== 6. 保存模型 ==========
        os.makedirs('models', exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'sim_matrix': sim_matrix,
                'user_indices': user_indices,
                'user_map': self.user_map,
                'reverse_user_map': self.reverse_user_map,
                'user_means': self.user_means,
                'training_stats': {
                    'n_users': n_users,
                    'n_movies': n_movies,
                    'sparsity': len(ratings_filtered) / (n_users * n_movies)
                }
            }, f)
        
        if not UserCF._loaded:
            print(f"\n💾 UserCF模型已保存至 {self.model_path}")
            print(f"   覆盖用户数: {len(user_indices)}")
            print(f"   覆盖电影数: {n_movies}")
            UserCF._loaded = True
            print("=" * 50)
            print("✅ UserCF训练完成！")
            print("=" * 50)
    
    def _load_model(self):
        """加载模型"""
        if not UserCF._loaded:
            print("正在加载 UserCF 模型...")
        
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.sim_matrix = data['sim_matrix']
            self.user_indices = data['user_indices']
            self.user_map = data['user_map']
            self.reverse_user_map = data['reverse_user_map']
            self.user_means = data.get('user_means', {})
        
        if not UserCF._loaded:
            print(f"✅ UserCF模型加载成功")
            if 'training_stats' in data:
                stats = data['training_stats']
                print(f"   覆盖 {stats['n_users']} 个用户, {stats['n_movies']} 部电影")
                print(f"   稀疏度: {stats['sparsity']:.2%}")
            UserCF._loaded = True
    
    def recall(self, user_id, top_n=100, user_history=None):
        """
        为用户生成召回候选集
        
        Args:
            user_id: 用户ID
            top_n: 召回数量
            user_history: 用户历史（评估模式使用）
        
        Returns:
            List[int]: 推荐的电影ID列表
        """
        
        # 快速检查用户是否在模型中
        if user_id not in self.reverse_user_map:
            # 用户不在训练集中，使用降级策略
            return self._fallback_recall(user_id, top_n, user_history)
        
        user_idx = self.reverse_user_map[user_id]
        
        # 检查用户索引是否在相似度矩阵中
        if user_idx not in self.user_indices:
            return self._fallback_recall(user_id, top_n, user_history)
        
        # 找到用户在相似度矩阵中的位置
        pos = np.where(self.user_indices == user_idx)[0]
        if len(pos) == 0:
            return self._fallback_recall(user_id, top_n, user_history)
        
        pos = pos[0]
        
        # 获取相似度分数
        sim_scores = self.sim_matrix[pos]
        
        # 找到最相似的K个用户（排除自己）
        K_SIMILAR = 30
        if len(sim_scores) > K_SIMILAR + 1:
            similar_indices = np.argsort(sim_scores)[::-1][1:K_SIMILAR+1]
        else:
            similar_indices = np.argsort(sim_scores)[::-1][1:]
        
        # 过滤低相似度（相似度<0.1的忽略）
        similar_indices = [idx for idx in similar_indices if sim_scores[idx] > 0.1]
        
        if len(similar_indices) == 0:
            return self._fallback_recall(user_id, top_n, user_history)
        
        # 获取相似用户的原始索引
        similar_users = [self.user_indices[idx] for idx in similar_indices]
        
        # 获取用户已看过的电影
        ratings = self._get_ratings()
        
        if user_history is not None:
            user_watched = set(user_history)
        else:
            user_watched = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
        
        # 收集相似用户喜欢的电影（加权投票）
        recommendations = {}
        
        for sim_user_idx in similar_users[:20]:  # 最多使用20个相似用户
            sim_user_id = self.user_map[sim_user_idx]
            
            # 获取相似度分数
            sim_pos = np.where(self.user_indices == sim_user_idx)[0]
            if len(sim_pos) == 0:
                continue
            sim_score = sim_scores[sim_pos[0]]
            
            # 获取该用户评分高的电影（>=4分）
            user_movies = ratings[ratings['userId'] == sim_user_id]
            liked_movies = user_movies[user_movies['rating'] >= 4.0]['movieId'].tolist()
            
            # 为每个喜欢的电影投票（相似度作为权重）
            for movie_id in liked_movies[:20]:  # 每个用户最多贡献20部电影
                if movie_id not in user_watched:
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + sim_score
        
        # 如果推荐不足，补充热门电影
        if len(recommendations) < top_n:
            # 只在非评估模式下补充（评估模式下不补充，保持公平）
            if user_history is None:
                from .popularity import Popularity
                pop = Popularity()
                hot_movies = pop.recall(None, top_n * 2)
                for movie_id in hot_movies:
                    if movie_id not in user_watched and movie_id not in recommendations:
                        recommendations[movie_id] = recommendations.get(movie_id, 0) + 0.01
                        if len(recommendations) >= top_n:
                            break
        
        # 如果没有推荐结果，使用降级策略
        if not recommendations:
            return self._fallback_recall(user_id, top_n, user_history)
        
        # 按分数排序返回
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [x[0] for x in sorted_recs]
    
    def _fallback_recall(self, user_id, top_n=100, user_history=None):
        """
        降级召回策略：基于用户观看历史的体裁推荐
        保证所有用户都能获得推荐
        """
        ratings = self._get_ratings()
        movies = self._get_movies()
        
        # 获取用户已看过的电影
        if user_history is not None:
            user_watched = set(user_history)
        else:
            user_watched = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
        
        # 如果用户没有观看历史，直接返回热门电影
        if len(user_watched) == 0:
            from .popularity import Popularity
            pop = Popularity()
            return pop.recall(None, top_n)
        
        # 提取用户观看过的电影信息
        user_movies = movies[movies['movieId'].isin(user_watched)]
        
        # 提取用户喜欢的体裁
        all_genres = []
        for genres in user_movies['genres'].dropna():
            if genres and genres != '(no genres listed)':
                all_genres.extend(genres.split('|'))
        
        if len(all_genres) == 0:
            from .popularity import Popularity
            pop = Popularity()
            return pop.recall(None, top_n)
        
        # 统计体裁分布
        genre_counts = Counter(all_genres)
        
        # 取前3个最喜欢的体裁
        top_genres = [g for g, _ in genre_counts.most_common(3)]
        
        # 推荐这些体裁的热门电影
        genre_movies = set()
        for genre in top_genres:
            genre_filter = movies[movies['genres'].str.contains(genre, na=False)]
            genre_movies.update(genre_filter['movieId'].tolist())
        
        # 排除已看过的电影
        genre_movies = genre_movies - user_watched
        
        if len(genre_movies) == 0:
            from .popularity import Popularity
            pop = Popularity()
            return pop.recall(None, top_n)
        
        # 按热度排序
        ratings_stats = self._get_rating_stats()
        
        movie_list = list(genre_movies)
        popularity_scores = [ratings_stats.get(mid, 0) for mid in movie_list]
        
        sorted_movies = [mid for _, mid in sorted(zip(popularity_scores, movie_list), reverse=True)]
        return sorted_movies[:top_n]
    
    def _get_ratings(self):
        """获取评分数据（带缓存）"""
        if UserCF._ratings_cache is None:
            UserCF._ratings_cache = pd.read_parquet('processed/ratings.parquet')
        return UserCF._ratings_cache
    
    def _get_movies(self):
        """获取电影数据（带缓存）"""
        if UserCF._movies_cache is None:
            UserCF._movies_cache = pd.read_parquet('processed/movies.parquet')
        return UserCF._movies_cache
    
    def _get_rating_stats(self):
        """获取评分统计（带缓存）"""
        if UserCF._rating_stats_cache is None:
            ratings = self._get_ratings()
            stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
            stats['popularity'] = stats['count'] * stats['mean']
            UserCF._rating_stats_cache = stats['popularity'].to_dict()
        return UserCF._rating_stats_cache


# ========== 独立的训练函数（可选）==========
def build_user_cf():
    """独立的训练函数，方便单独调用"""
    print("开始训练UserCF模型...")
    model = UserCF()
    print("训练完成！")


if __name__ == "__main__":
    # 快速测试
    print("=" * 50)
    print("UserCF 快速测试")
    print("=" * 50)
    
    usercf = UserCF()
    
    # 测试几个用户
    test_users = [1, 100, 1000, 5000]
    for uid in test_users:
        recs = usercf.recall(uid, 10)
        print(f"User {uid}: 推荐 {len(recs)} 部电影")
        if recs:
            print(f"  示例: {recs[:5]}")
    
    print("\n✅ 测试完成！")