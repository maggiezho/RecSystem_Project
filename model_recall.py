import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def build_item_cf():
    print("加载处理后的数据...")
    ratings = pd.read_parquet('processed/ratings.parquet')
    
    # --- 1. 构建评分矩阵 ---
    # 为了节省内存，我们只取前 10000 名活跃用户和热门电影做相似度计算，否则矩阵太大
    # 这一步是性能优化的关键
    popular_movies = ratings['movieId'].value_counts().head(10000).index
    ratings_tiny = ratings[ratings['movieId'].isin(popular_movies)]
    
    print("正在构建稀疏矩阵...")
    # 行是 movieId, 列是 userId, 值是评分
    # 使用 CSR 格式存储，只记录有分的部分，极大地节省空间
    user_u_col = ratings_tiny['userId'].astype('category').cat.codes
    item_i_col = ratings_tiny['movieId'].astype('category').cat.codes
    
    # 记录分类编码的映射关系，方便后续找回原始 ID
    item_map = dict(enumerate(ratings_tiny['movieId'].astype('category').cat.categories))
    
    row = item_i_col.values
    col = user_u_col.values
    data = ratings_tiny['rating'].values
    
    sparse_item_user = csr_matrix((data, (row, col)))

    # --- 2. 计算余弦相似度 ---
    print("正在计算电影相似度矩阵（余弦相似度）...")
    # cosine_similarity 会返回一个 [10000, 10000] 的对称矩阵
    item_similarity = cosine_similarity(sparse_item_user, dense_output=False)
    
    # --- 3. 结果持久化 ---
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open('models/item_similarity.pkl', 'wb') as f:
        pickle.dump({'matrix': item_similarity, 'map': item_map}, f)
    
    print("✅ 召回层模型已保存至 'models/item_similarity.pkl'")

if __name__ == "__main__":
    build_item_cf()

def get_recommendations(user_id, top_n=10):
    # 加载模型和原始数据
    with open('models/item_similarity.pkl', 'rb') as f:
        data = pickle.load(f)
        sim_matrix = data['matrix']
        item_map = data['map']
    
    ratings = pd.read_parquet('processed/ratings.parquet')
    movies = pd.read_parquet('processed/movies.parquet')
    
    # 1. 找到该用户评价最高的前 5 部电影
    user_ratings = ratings[ratings['userId'] == user_id].sort_values('rating', ascending=False).head(5)
    user_movie_ids = user_ratings['movieId'].tolist()
    
    # 2. 在相似度矩阵中找到与这些电影最像的候选
    recommendations = {}
    reverse_item_map = {v: k for k, v in item_map.items()}
    
    for m_id in user_movie_ids:
        if m_id in reverse_item_map:
            idx = reverse_item_map[m_id]
            # 获取相似度最高的前 N 个索引
            sim_scores = sim_matrix[idx].toarray().flatten()
            similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
            
            for i in similar_indices:
                movie_id = item_map[i]
                if movie_id not in user_movie_ids: # 排除用户看过的
                    recommendations[movie_id] = recommendations.get(movie_id, 0) + sim_scores[i]
    
    # 3. 排序并返回标题
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    rec_ids = [x[0] for x in sorted_recs]
    
    result = movies[movies['movieId'].isin(rec_ids)][['title', 'genres']]
    return result

# 测试一下
# print(get_recommendations(user_id=1))