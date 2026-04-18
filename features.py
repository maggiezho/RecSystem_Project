import pandas as pd
import numpy as np

def extract_features():
    print("开始特征工程...")
    ratings = pd.read_parquet('processed/ratings.parquet')
    movies = pd.read_parquet('processed/movies.parquet')
    
    # --- 1. 用户特征 (User Features) ---
    user_features = ratings.groupby('userId').agg({
        'rating': ['mean', 'std', 'count'],
        'timestamp': ['max', 'min']
    })
    user_features.columns = ['user_avg_rating', 'user_rating_std', 'user_rating_count', 'user_last_ts', 'user_first_ts']
    user_features = user_features.reset_index()

    # --- 2. 电影特征 (Item Features) ---
    movie_features = ratings.groupby('movieId').agg({
        'rating': ['mean', 'std', 'count']
    })
    movie_features.columns = ['movie_avg_rating', 'movie_rating_std', 'movie_rating_count']
    movie_features = movie_features.reset_index()
    
    # 合并电影年份
    movie_features = pd.merge(movie_features, movies[['movieId', 'year']], on='movieId')

    # 保存特征库
    user_features.to_parquet('processed/user_features.parquet', index=False)
    movie_features.to_parquet('processed/movie_features.parquet', index=False)
    print("✅ 特征提取完成")

if __name__ == "__main__":
    extract_features()