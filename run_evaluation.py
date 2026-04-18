# run_evaluation.py - 优化版，直观展示命中率

import os
import sys
import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")
os.environ['STREAMLIT_RUNNING'] = 'false'

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from evaluate import RecEvaluator
from recall import get_recommendation_func


def run_intuitive_evaluation():
    """运行直观的离线评估（用命中率百分比展示）"""
    
    # 初始化评估器
    print("\n📂 加载数据...")
    evaluator = RecEvaluator('processed/ratings.parquet')
    
    # 获取测试用户
    ratings = pd.read_parquet('processed/ratings.parquet')
    user_rating_counts = ratings.groupby('userId').size()
    
    # 选择至少有50个评分的用户
    test_users = user_rating_counts[user_rating_counts >= 50].index.tolist()
    
    # 限制测试用户数量
    max_test_users = 100
    if len(test_users) > max_test_users:
        np.random.seed(42)
        test_users = np.random.choice(test_users, max_test_users, replace=False).tolist()
    
    print(f"📊 测试用户数量: {len(test_users)}")
    print(f"📊 评估方法: 留一法（80%历史数据训练，20%测试）")
    print(f"📊 K值: 5, 10, 20")
    
    # 定义要评估的策略
    strategies = {
        'ItemCF': get_recommendation_func('itemcf'),
        'UserCF': get_recommendation_func('usercf'),
        '热门电影': get_recommendation_func('popularity'),
        '混合召回': get_recommendation_func('hybrid', {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}),
    }
    
    print("\n" + "=" * 70)
    print("开始评估...")
    print("=" * 70)
    
    results = []
    
    for strategy_name, recommend_func in strategies.items():
        print(f"\n📊 评估策略: {strategy_name}")
        
        # 存储每个用户的命中率
        user_hit_rates = {5: [], 10: [], 20: []}
        
        for user_id in test_users:
            # 获取训练集和测试集
            train_items, test_items = evaluator.get_user_test_items_loo(user_id, 0.8)
            
            if len(test_items) == 0:
                continue
            
            try:
                # 获取推荐
                recommendations = recommend_func(user_id, 20, train_items)
                
                if not recommendations:
                    continue
                
                # 计算各K值的命中率
                for k in [5, 10, 20]:
                    rec_k = set(recommendations[:k])
                    test_set = set(test_items)
                    hits = len(rec_k & test_set)
                    hit_rate = hits / k  # 命中率 = 命中数 / 推荐数
                    user_hit_rates[k].append(hit_rate)
                    
            except Exception as e:
                continue
        
        # 计算平均命中率
        avg_hit_rates = {}
        for k in [5, 10, 20]:
            if user_hit_rates[k]:
                avg_hit_rates[k] = np.mean(user_hit_rates[k]) * 100  # 转为百分比
            else:
                avg_hit_rates[k] = 0
        
        results.append({
            '策略': strategy_name,
            '有效用户数': len(user_hit_rates[5]),
            '命中率@5': f"{avg_hit_rates[5]:.1f}%",
            '命中率@10': f"{avg_hit_rates[10]:.1f}%",
            '命中率@20': f"{avg_hit_rates[20]:.1f}%",
        })
        
        print(f"  有效用户: {len(user_hit_rates[5])}/{len(test_users)}")
        print(f"  命中率@5: {avg_hit_rates[5]:.1f}%  ← 推荐5部，平均有几部在测试集中")
        print(f"  命中率@10: {avg_hit_rates[10]:.1f}%")
        print(f"  命中率@20: {avg_hit_rates[20]:.1f}%")
    
    # 创建结果表格
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 评估结果（命中率百分比）")
    print("=" * 70)
    print("\n说明：命中率 = 推荐列表中出现在测试集中的比例")
    print("测试集 = 用户20%的历史评分（用户真实喜欢的电影）\n")
    
    print(results_df.to_string(index=False))
    
    # 找出最佳策略
    best_idx = results_df['命中率@5'].str.rstrip('%').astype(float).argmax()
    best_strategy = results_df.iloc[best_idx]['策略']
    best_hitrate = results_df.iloc[best_idx]['命中率@5']
    
    print("\n" + "=" * 70)
    print(f"🏆 最佳策略: {best_strategy}")
    print(f"   命中率@5 = {best_hitrate}")
    print("=" * 70)
    
    # 保存结果
    results_df.to_csv('evaluation_results_intuitive.csv', index=False)
    print("\n✅ 结果已保存至 evaluation_results_intuitive.csv")
    
    return results_df


if __name__ == "__main__":
    print("=" * 70)
    print("推荐系统评估工具（直观版）")
    print("=" * 70)
    print("\n评估方法：留一法")
    print("- 训练集：用户80%的历史评分")
    print("- 测试集：用户20%的历史评分（真实喜欢的电影）")
    print("- 命中率：推荐结果在测试集中的比例\n")
    
    results = run_intuitive_evaluation()
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)