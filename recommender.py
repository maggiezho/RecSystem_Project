import pandas as pd
import pickle

def get_final_recommendation(user_id, top_k=10):
    # 1. 加载所有模型和特征
    with open('models/ranking_model.pkl', 'rb') as f:
        ranker = pickle.load(f)
    
    # 假设你已经写好了前一步的召回函数 get_recall_results
    # 这里我们模拟召回了 100 个候选 movieId
    from model_recall import get_recommendations 
    recall_df = get_recommendations(user_id, top_n=100) 
    candidate_ids = recall_df.index.tolist() # 或者是你召回结果里的 ID 列表

    # 2. 准备排序特征
    user_feat = pd.read_parquet('processed/user_features.parquet')
    movie_feat = pd.read_parquet('processed/movie_features.parquet')
    
    test_data = pd.DataFrame({'userId': [user_id]*len(candidate_ids), 'movieId': candidate_ids})
    test_data = pd.merge(test_data, user_feat, on='userId', how='left')
    test_data = pd.merge(test_data, movie_feat, on='movieId', how='left')
    
    feature_cols = ['user_avg_rating', 'user_rating_std', 'user_rating_count',
                    'movie_avg_rating', 'movie_rating_std', 'movie_rating_count', 'year']
    
    # 3. 模型预测概率
    probs = ranker.predict_proba(test_data[feature_cols])[:, 1]
    test_data['score'] = probs
    
    # 4. 按概率排序，取前 Top K
    final_ids = test_data.sort_values('score', ascending=False).head(top_k)['movieId'].tolist()
    
    movies = pd.read_parquet('processed/movies.parquet')
    return movies[movies['movieId'].isin(final_ids)]