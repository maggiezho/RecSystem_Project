"""
推荐系统评估模块
实现常用的离线评估指标
"""

import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from typing import List, Dict, Tuple
import time


class RecEvaluator:
    """推荐系统评估器"""
    
    def __init__(self, ratings_path='processed/ratings.parquet'):
        """
        初始化评估器
        """
        self.ratings = pd.read_parquet(ratings_path)
        # 定义正样本：评分 >= 4 的电影
        self.positive_threshold = 4.0
        
    def get_user_positive_items(self, user_id: int) -> List[int]:
        """
        获取用户喜欢的电影列表（正样本）
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        positive = user_ratings[user_ratings['rating'] >= self.positive_threshold]['movieId'].tolist()
        return positive
    
    def get_user_test_items(self, user_id: int, train_ratio: float = 0.8, use_time_split: bool = False) -> Tuple[List[int], List[int]]:
        """
        将用户的历史数据分为训练集和测试集
        返回: (train_items, test_items)
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if use_time_split:
            # 按时间分割
            user_ratings = user_ratings.sort_values('timestamp')
            n = len(user_ratings)
            split_idx = int(n * train_ratio)
            train_items = user_ratings.iloc[:split_idx]['movieId'].tolist()
            test_items = user_ratings.iloc[split_idx:]['movieId'].tolist()
        else:
            # 随机分割
            user_items = user_ratings['movieId'].tolist()
            n = len(user_items)
            n_train = int(n * train_ratio)
            indices = np.random.permutation(n)
            train_items = [user_items[i] for i in indices[:n_train]]
            test_items = [user_items[i] for i in indices[n_train:]]
        
        return train_items, test_items
    
    # ========== 新增：留一法评估方法 ==========
    def get_user_test_items_loo(self, user_id: int, train_ratio: float = 0.8) -> Tuple[List[int], List[int]]:
        """
        留一法评估：将用户评分高的电影分成训练集和测试集
        测试集只包含用户喜欢但还没推荐过的电影
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        # 只使用高评分电影（>=4分）
        positive_items = user_ratings[user_ratings['rating'] >= self.positive_threshold]['movieId'].tolist()
        
        if len(positive_items) < 2:
            # 如果高评分电影太少，使用所有评分
            positive_items = user_ratings['movieId'].tolist()
        
        if len(positive_items) < 2:
            return [], []
        
        # 随机打乱
        np.random.shuffle(positive_items)
        
        # 80%作为训练，20%作为测试（但至少留1个）
        n_train = max(1, int(len(positive_items) * train_ratio))
        n_test = len(positive_items) - n_train
        
        if n_test == 0:
            n_train = len(positive_items) - 1
            n_test = 1
        
        train_items = positive_items[:n_train]
        test_items = positive_items[n_train:]
        
        return train_items, test_items
    
    def precision_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """
        计算 Precision@K
        公式: |推荐列表中相关的物品| / K
        """
        if k <= 0:
            return 0.0
        
        rec_k = recommendations[:k]
        relevant = set(rec_k) & set(actual)
        return len(relevant) / k
    
    def recall_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """
        计算 Recall@K
        公式: |推荐列表中相关的物品| / |所有相关物品|
        """
        if len(actual) == 0:
            return 0.0
        
        rec_k = recommendations[:k]
        relevant = set(rec_k) & set(actual)
        return len(relevant) / len(actual)
    
    def f1_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """
        计算 F1@K
        公式: 2 * (Precision * Recall) / (Precision + Recall)
        """
        p = self.precision_at_k(recommendations, actual, k)
        r = self.recall_at_k(recommendations, actual, k)
        
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def ndcg_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """
        计算 NDCG@K (Normalized Discounted Cumulative Gain)
        考虑排序位置的评估指标
        """
        if k <= 0 or len(actual) == 0:
            return 0.0
        
        # 构建相关度字典
        relevance_dict = {movie_id: 1 for movie_id in actual}
        
        # 计算 DCG
        dcg = 0.0
        for i, movie_id in enumerate(recommendations[:k]):
            relevance = relevance_dict.get(movie_id, 0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 1)
        
        # 计算 IDCG (理想情况)
        ideal_recommendations = actual[:k]
        idcg = 0.0
        for i, movie_id in enumerate(ideal_recommendations):
            if i == 0:
                idcg += 1.0
            else:
                idcg += 1.0 / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """
        计算 Hit Rate@K
        是否至少有一个相关物品在推荐列表中
        """
        if len(actual) == 0:
            return 0.0
        
        rec_k = set(recommendations[:k])
        hits = rec_k & set(actual)
        return 1.0 if len(hits) > 0 else 0.0
    
    def mrr_at_k(self, recommendations: List[int], actual: List[int], k: int) -> float:
        """
        计算 MRR@K (Mean Reciprocal Rank)
        第一个相关物品的倒数排名
        """
        if len(actual) == 0:
            return 0.0
        
        actual_set = set(actual)
        for i, movie_id in enumerate(recommendations[:k]):
            if movie_id in actual_set:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_recommendations(self, 
                                  recommend_func, 
                                  test_users: List[int],
                                  k_values: List[int] = [5, 10, 20],
                                  train_ratio: float = 0.8,
                                  verbose: bool = True) -> Dict:
        """
        评估推荐函数
        """
        results = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': [], 'hit_rate': [], 'mrr': []} 
                   for k in k_values}
        
        total_time = 0
        
        for i, user_id in enumerate(test_users):
            if verbose and (i + 1) % 50 == 0:
                print(f"评估进度: {i+1}/{len(test_users)}")
            
            # 获取训练集和测试集
            train_items, test_items = self.get_user_test_items(user_id, train_ratio)
            
            # 如果测试集为空，跳过
            if len(test_items) == 0:
                continue
            
            # 调用推荐函数（基于训练集的历史）
            start_time = time.time()
            try:
                recommendations = recommend_func(user_id, max(k_values), train_items)
                total_time += time.time() - start_time
            except Exception as e:
                print(f"用户 {user_id} 推荐失败: {e}")
                continue
            
            # 计算各项指标
            for k in k_values:
                results[k]['precision'].append(self.precision_at_k(recommendations, test_items, k))
                results[k]['recall'].append(self.recall_at_k(recommendations, test_items, k))
                results[k]['f1'].append(self.f1_at_k(recommendations, test_items, k))
                results[k]['ndcg'].append(self.ndcg_at_k(recommendations, test_items, k))
                results[k]['hit_rate'].append(self.hit_rate_at_k(recommendations, test_items, k))
                results[k]['mrr'].append(self.mrr_at_k(recommendations, test_items, k))
        
        # 计算平均值
        avg_results = {}
        for k in k_values:
            avg_results[k] = {}
            for metric in results[k]:
                values = results[k][metric]
                avg_results[k][metric] = np.mean(values) if values else 0.0
        
        if verbose:
            print(f"\n✅ 评估完成！平均耗时: {total_time/len(test_users):.3f} 秒/用户")
        
        return avg_results
    
    # ========== 新增：留一法评估 ==========
    def evaluate_recommendations_loo(self, 
                                      recommend_func, 
                                      test_users: List[int],
                                      k_values: List[int] = [5, 10, 20],
                                      train_ratio: float = 0.8,
                                      verbose: bool = True) -> Dict:
        """
        使用留一法评估推荐函数
        """
        results = {k: {'precision': [], 'recall': [], 'f1': [], 'ndcg': [], 'hit_rate': [], 'mrr': []} 
                   for k in k_values}
        
        for i, user_id in enumerate(test_users):
            if verbose and (i + 1) % 50 == 0:
                print(f"评估进度: {i+1}/{len(test_users)}")
            
            # 获取训练集和测试集（留一法）
            train_items, test_items = self.get_user_test_items_loo(user_id, train_ratio)
            
            if len(test_items) == 0:
                continue
            
            try:
                recommendations = recommend_func(user_id, max(k_values), train_items)
            except Exception as e:
                print(f"用户 {user_id} 推荐失败: {e}")
                continue
            
            for k in k_values:
                # 计算各项指标
                results[k]['precision'].append(self.precision_at_k(recommendations, test_items, k))
                results[k]['recall'].append(self.recall_at_k(recommendations, test_items, k))
                results[k]['f1'].append(self.f1_at_k(recommendations, test_items, k))
                results[k]['ndcg'].append(self.ndcg_at_k(recommendations, test_items, k))
                results[k]['hit_rate'].append(self.hit_rate_at_k(recommendations, test_items, k))
                results[k]['mrr'].append(self.mrr_at_k(recommendations, test_items, k))
        
        # 计算平均值
        avg_results = {}
        for k in k_values:
            avg_results[k] = {}
            for metric in results[k]:
                values = results[k][metric]
                avg_results[k][metric] = np.mean(values) if values else 0.0
        
        return avg_results
    
    def compare_strategies(self, 
                           strategy_funcs: Dict[str, callable],
                           test_users: List[int],
                           k_values: List[int] = [5, 10, 20],
                           train_ratio: float = 0.8) -> pd.DataFrame:
        """
        对比多种推荐策略（原有方法）
        """
        comparison = []
        
        for strategy_name, recommend_func in strategy_funcs.items():
            print(f"\n📊 评估策略: {strategy_name}")
            results = self.evaluate_recommendations(
                recommend_func, test_users, k_values, train_ratio, verbose=False
            )
            
            for k in k_values:
                comparison.append({
                    '策略': strategy_name,
                    'K': k,
                    'Precision@K': f"{results[k]['precision']:.4f}",
                    'Recall@K': f"{results[k]['recall']:.4f}",
                    'F1@K': f"{results[k]['f1']:.4f}",
                    'NDCG@K': f"{results[k]['ndcg']:.4f}",
                    'Hit Rate@K': f"{results[k]['hit_rate']:.4f}",
                    'MRR@K': f"{results[k]['mrr']:.4f}"
                })
        
        return pd.DataFrame(comparison)
    
    # ========== 新增：留一法对比策略 ==========
    def compare_strategies_loo(self, 
                               strategy_funcs: Dict[str, callable],
                               test_users: List[int],
                               k_values: List[int] = [5, 10, 20],
                               train_ratio: float = 0.8) -> pd.DataFrame:
        """
        使用留一法对比多种推荐策略
        """
        comparison = []
        
        for strategy_name, recommend_func in strategy_funcs.items():
            print(f"\n📊 评估策略: {strategy_name}")
            results = self.evaluate_recommendations_loo(
                recommend_func, test_users, k_values, train_ratio, verbose=False
            )
            
            for k in k_values:
                comparison.append({
                    '策略': strategy_name,
                    'K': k,
                    'Precision@K': f"{results[k]['precision']:.4f}",
                    'Recall@K': f"{results[k]['recall']:.4f}",
                    'F1@K': f"{results[k]['f1']:.4f}",
                    'NDCG@K': f"{results[k]['ndcg']:.4f}",
                    'Hit Rate@K': f"{results[k]['hit_rate']:.4f}",
                    'MRR@K': f"{results[k]['mrr']:.4f}"
                })
        
        return pd.DataFrame(comparison)

class OnlineEvaluator:
    """在线评估（模拟A/B测试）"""
    
    def __init__(self):
        self.feedback_log = []  # 存储用户反馈
        
    def log_feedback(self, user_id: int, movie_id: int, liked: bool, strategy_name: str = None):
        """
        记录用户反馈
        """
        self.feedback_log.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'liked': liked,
            'strategy': strategy_name,  # 确保记录策略名称
            'timestamp': time.time()
        })
    
    def get_ctr(self, strategy_name: str = None) -> float:
        """
        计算点击通过率 (Click Through Rate)
        """
        if strategy_name:
            logs = [l for l in self.feedback_log if l.get('strategy') == strategy_name]
        else:
            logs = self.feedback_log
        
        if len(logs) == 0:
            return 0.0
        
        likes = sum(1 for l in logs if l['liked'])
        return likes / len(logs)
    
    def get_feedback_summary(self) -> pd.DataFrame:
        """
        获取反馈摘要
        """
        if len(self.feedback_log) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.feedback_log)
        summary = df.groupby('strategy').agg({
            'liked': ['count', 'sum', 'mean']
        }).round(4)
        summary.columns = ['总推荐数', '喜欢数', '喜欢率']
        return summary