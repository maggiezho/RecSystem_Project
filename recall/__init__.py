# recall/__init__.py
"""
召回模块
提供各种召回策略的统一接口
"""

from .user_cf import UserCF
from .item_cf import ItemCF
from .popularity import Popularity

# 全局实例（单例模式）
_usercf_instance = None
_itemcf_instance = None
_popularity_instance = None


def get_recommendation_func(strategy, params=None):
    """
    获取推荐函数
    
    参数:
        strategy: 召回策略名称 ('usercf', 'itemcf', 'popularity', 'hybrid')
        params: 策略参数（用于混合召回）
    
    返回:
        recommend_func: 推荐函数，签名为 func(user_id, top_n) -> List[int]
    """
    
    if strategy == 'usercf':
        global _usercf_instance
        if _usercf_instance is None:
            _usercf_instance = UserCF()
        
        def recommend(user_id, top_n=100, user_history=None):
            return _usercf_instance.recall(user_id, top_n, user_history)
        return recommend
    
    elif strategy == 'itemcf':
        global _itemcf_instance
        if _itemcf_instance is None:
            _itemcf_instance = ItemCF()
        
        def recommend(user_id, top_n=100, user_history=None):
            return _itemcf_instance.recall(user_id, top_n, user_history)
        return recommend
    
    elif strategy == 'popularity':
        global _popularity_instance
        if _popularity_instance is None:
            _popularity_instance = Popularity()
        
        def recommend(user_id, top_n=100, user_history=None):
            return _popularity_instance.recall(user_id, top_n)
        return recommend
    
    elif strategy == 'hybrid':
        # 混合召回
        itemcf_func = get_recommendation_func('itemcf')
        usercf_func = get_recommendation_func('usercf')
        pop_func = get_recommendation_func('popularity')
        
        weights = params or {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}
        
        def recommend(user_id, top_n=100, user_history=None):
            # 获取各策略的推荐（保持有序列表形式）
            itemcf_recs = itemcf_func(user_id, top_n * 2, user_history)
            usercf_recs = usercf_func(user_id, top_n * 2, user_history)
            pop_recs = pop_func(user_id, top_n, user_history)
            
            all_recs = {}
            k_smooth = 60
            
            # 1. ItemCF 倒数顺位加权
            for rank, rec in enumerate(itemcf_recs):
                rrf_score = 1.0 / (k_smooth + rank + 1)
                all_recs[rec] = all_recs.get(rec, 0) + weights['itemcf'] * rrf_score
                
            # 2. UserCF 倒数顺位加权
            for rank, rec in enumerate(usercf_recs):
                rrf_score = 1.0 / (k_smooth + rank + 1)
                all_recs[rec] = all_recs.get(rec, 0) + weights['usercf'] * rrf_score
                
            # 3. Popularity 倒数顺位加权
            for rank, rec in enumerate(pop_recs):
                rrf_score = 1.0 / (k_smooth + rank + 1)
                all_recs[rec] = all_recs.get(rec, 0) + weights['pop'] * rrf_score
            
            # 按计算后的混合 RRF 分数重新从大到小排序
            sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
            return [rec for rec, _ in sorted_recs[:top_n]]
        
        return recommend
    
    else:
        raise ValueError(f"未知的召回策略: {strategy}")


# 为了方便，也可以直接导出各个类
__all__ = ['UserCF', 'ItemCF', 'Popularity', 'get_recommendation_func']

class HybridRecall:
    """混合召回类，封装了混合召回逻辑"""
    
    def __init__(self, weights=None):
        """
        初始化混合召回
        
        参数:
            weights: 各策略权重，例如 {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}
        """
        self.weights = weights or {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}
        self.itemcf_func = get_recommendation_func('itemcf')
        self.usercf_func = get_recommendation_func('usercf')
        self.pop_func = get_recommendation_func('popularity')
    
    def recommend(self, user_id, top_n=100, user_history=None):
        """
        混合召回推荐
        
        参数:
            user_id: 用户ID
            top_n: 推荐数量
            user_history: 用户历史（可选）
        
        返回:
            List[int]: 推荐的item ID列表
        """
        # 获取各策略的推荐
        itemcf_recs = self.itemcf_func(user_id, top_n * 2, user_history)
        usercf_recs = self.usercf_func(user_id, top_n * 2, user_history)
        pop_recs = self.pop_func(user_id, top_n, user_history)
        
        # 合并并去重，计算加权分数
        all_recs = {}
        for rec in itemcf_recs:
            all_recs[rec] = all_recs.get(rec, 0) + self.weights['itemcf']
        for rec in usercf_recs:
            all_recs[rec] = all_recs.get(rec, 0) + self.weights['usercf']
        for rec in pop_recs:
            all_recs[rec] = all_recs.get(rec, 0) + self.weights['pop']
        
        # 按分数排序
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in sorted_recs[:top_n]]
    
    def recall(self, user_id, top_n=100, user_history=None):
        """别名，与UserCF/ItemCF接口保持一致"""
        return self.recommend(user_id, top_n, user_history)


# 更新 __all__
__all__ = ['UserCF', 'ItemCF', 'Popularity', 'HybridRecall', 'get_recommendation_func']