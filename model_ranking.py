import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle

def train_ranking_model():
    print("构建训练数据集...")
    ratings = pd.read_parquet('processed/ratings.parquet')
    user_feat = pd.read_parquet('processed/user_features.parquet')
    movie_feat = pd.read_parquet('processed/movie_features.parquet')

    # 1. 构造正负样本
    # 我们把评分 >= 4 的看作 1 (用户喜欢), < 4 的看作 0 (不喜欢)
    data = ratings.copy()
    data['label'] = (data['rating'] >= 4).astype(int)
    
    # 2. 合并特征
    data = pd.merge(data, user_feat, on='userId', how='left')
    data = pd.merge(data, movie_feat, on='movieId', how='left')
    
    # 3. 选择特征列
    feature_cols = [
        'user_avg_rating', 'user_rating_std', 'user_rating_count',
        'movie_avg_rating', 'movie_rating_std', 'movie_rating_count', 'year'
    ]
    X = data[feature_cols]
    y = data['label']

    # 4. 划分训练集和验证集 (用最后 20% 的数据验证)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("开始训练 LightGBM 排序模型...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        objective='binary',
        importance_type='gain'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=10)])

    # 5. 保存模型
    with open('models/ranking_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 排序模型已保存")

if __name__ == "__main__":
    train_ranking_model()