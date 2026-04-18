from .item_cf import ItemCF
from .user_cf import UserCF
from .popularity import Popularity

class HybridRecall:
    """混合召回：加权融合多种召回策略"""
    
    def __init__(self, weights=None):
        """
        weights: dict, 各召回策略的权重
        例如: {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}
        """
        self.itemcf = ItemCF()
        self.usercf = UserCF()
        self.popularity = Popularity()
        
        # 默认权重
        self.weights = weights or {
            'itemcf': 0.5,
            'usercf': 0.3, 
            'pop': 0.2
        }
        
        # 各策略的召回数量（每个策略召回 top_n * 倍数）
        self.recall_ratios = {
            'itemcf': 2,
            'usercf': 1.5,
            'pop': 0.5
        }
    
    def recall(self, user_id, top_n=100):
        """
        混合召回，返回融合后的候选电影列表
        """
        # 动态计算每个策略需要召回的数量
        recall_counts = {}
        for strategy, ratio in self.recall_ratios.items():
            recall_counts[strategy] = int(top_n * ratio)
        
        # 执行各策略召回
        candidates = {}
        
        # ItemCF
        if self.weights.get('itemcf', 0) > 0:
            itemcf_recs = self.itemcf.recall(user_id, recall_counts['itemcf'])
            self._add_candidates(candidates, itemcf_recs, self.weights['itemcf'])
        
        # UserCF
        if self.weights.get('usercf', 0) > 0:
            usercf_recs = self.usercf.recall(user_id, recall_counts['usercf'])
            self._add_candidates(candidates, usercf_recs, self.weights['usercf'])
        
        # Popularity（冷启动兜底，尤其是新用户）
        if self.weights.get('pop', 0) > 0:
            pop_recs = self.popularity.recall(user_id, recall_counts['pop'])
            self._add_candidates(candidates, pop_recs, self.weights['pop'])
        
        # 如果候选不足top_n，补充热门电影
        if len(candidates) < top_n:
            extra = self.popularity.recall(user_id, top_n * 2)
            for movie_id in extra:
                if movie_id not in candidates:
                    candidates[movie_id] = candidates.get(movie_id, 0) + 0.01
        
        # 排序返回
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_candidates[:top_n]]
    
    def _add_candidates(self, candidates, rec_list, weight):
        """将召回结果加入候选池，带权重"""
        for rank, movie_id in enumerate(rec_list):
            # 位置越靠前分数越高
            position_score = 1.0 / (rank + 1)
            candidates[movie_id] = candidates.get(movie_id, 0) + weight * position_score