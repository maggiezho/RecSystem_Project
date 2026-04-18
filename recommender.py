import pandas as pd
import pickle
import os
from recall import HybridRecall

# 全局初始化混合召回（只加载一次）
_hybrid_recall = None

def get_hybrid_recall():
    global _hybrid_recall
    if _hybrid_recall is None:
        _hybrid_recall = HybridRecall()
    return _hybrid_recall

def get_final_recommendation(user_id, top_k=10, recall_top_n=100):
    """
    双层推荐：召回 + 精排
    """
    # 1. 召回层：混合召回获取候选集
    recaller = get_hybrid_recall()
    candidate_ids = recaller.recall(user_id, recall_top_n)
    
    if len(candidate_ids) == 0:
        # 冷启动：新用户直接返回热门电影
        from recall.popularity import Popularity
        pop = Popularity()
        candidate_ids = pop.recall(user_id, top_k)
        movies = pd.read_parquet('processed/movies.parquet')
        return movies[movies['movieId'].isin(candidate_ids)][['title', 'genres', 'year']]
    
    # 2. 精排层：加载排序模型
    ranking_model_path = 'models/ranking_model.pkl'
    
    if not os.path.exists(ranking_model_path):
        # 如果排序模型不存在，直接返回召回结果
        movies = pd.read_parquet('processed/movies.parquet')
        result = movies[movies['movieId'].isin(candidate_ids[:top_k])]
        return result[['title', 'genres', 'year']]
    
    with open(ranking_model_path, 'rb') as f:
        ranker = pickle.load(f)
    
    # 准备特征
    user_feat = pd.read_parquet('processed/user_features.parquet')
    movie_feat = pd.read_parquet('processed/movie_features.parquet')
    
    test_data = pd.DataFrame({
        'userId': [user_id] * len(candidate_ids),
        'movieId': candidate_ids
    })
    test_data = pd.merge(test_data, user_feat, on='userId', how='left')
    test_data = pd.merge(test_data, movie_feat, on='movieId', how='left')
    
    feature_cols = [
        'user_avg_rating', 'user_rating_std', 'user_rating_count',
        'movie_avg_rating', 'movie_rating_std', 'movie_rating_count', 'year'
    ]
    
    # 填充缺失值
    for col in feature_cols:
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(test_data[col].median())
    
    # 预测
    probs = ranker.predict_proba(test_data[feature_cols])[:, 1]
    test_data['score'] = probs
    
    # 排序取top
    final_ids = test_data.sort_values('score', ascending=False).head(top_k)['movieId'].tolist()
    
    movies = pd.read_parquet('processed/movies.parquet')
    result = movies[movies['movieId'].isin(final_ids)]
    return result[['title', 'genres', 'year']]