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
            # 获取各策略的推荐
            itemcf_recs = itemcf_func(user_id, top_n * 2, user_history)
            usercf_recs = usercf_func(user_id, top_n * 2, user_history)
            pop_recs = pop_func(user_id, top_n, user_history)
            
            # 合并并去重，计算加权分数
            all_recs = {}
            for rec in itemcf_recs:
                all_recs[rec] = all_recs.get(rec, 0) + weights['itemcf']
            for rec in usercf_recs:
                all_recs[rec] = all_recs.get(rec, 0) + weights['usercf']
            for rec in pop_recs:
                all_recs[rec] = all_recs.get(rec, 0) + weights['pop']
            
            # 按分数排序
            sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
            return [rec for rec, _ in sorted_recs[:top_n]]
        
        return recommend
    
    else:
        raise ValueError(f"未知的召回策略: {strategy}")


# 为了方便，也可以直接导出各个类
__all__ = ['UserCF', 'ItemCF', 'Popularity', 'get_recommendation_func']