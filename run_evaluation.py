# run_evaluation.py - 优化版，直观展示准确率（已集成 tqdm 进度条与多指标）

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
from tqdm import tqdm 
from rich.console import Console
from rich.table import Table

def run_intuitive_evaluation():
    """运行直观的离线评估（用准确率百分比展示）"""
    
    # 初始化评估器
    print("\n📂 加载数据...")
    evaluator = RecEvaluator('processed/ratings.parquet')
    
    # 获取测试用户
    ratings = pd.read_parquet('processed/ratings.parquet')
    user_rating_counts = ratings.groupby('userId').size()
    
    # 选择至少有50个评分的用户
    test_users = user_rating_counts[user_rating_counts >= 50].index.tolist()
    
    max_test_users = 2000
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
        
        # 存储每个用户的各项指标
        user_precisions = {5: [], 10: [], 20: []}
        user_recalls = {5: [], 10: [], 20: []}
        user_hit_flags = {5: [], 10: [], 20: []}
        user_f1s = {5: [], 10: [], 20: []}
        user_ndcgs = {5: [], 10: [], 20: []}
        
        for user_id in tqdm(test_users, desc=f"   计算 {strategy_name} 进度", unit="user"):
            # 获取训练集和测试集
            train_items, test_items = evaluator.get_user_test_items_loo(user_id, 0.8)
            
            if len(test_items) == 0:
                continue
            
            try:
                # 获取推荐
                recommendations = recommend_func(user_id, 20, train_items)
                
                if not recommendations:
                    continue
                
                test_set = set(test_items)
                
                # 计算各K值的多指标
                for k in [5, 10, 20]:
                    rec_k = set(recommendations[:k])
                    hits = len(rec_k & test_set)
                    
                    # 1. Precision (准确率)
                    user_precisions[k].append(hits / k)
                    
                    # 2. Recall (召回率)
                    user_recalls[k].append(hits / len(test_set))
                    
                    # 3. HitRate (命中率)
                    user_hit_flags[k].append(1.0 if hits > 0 else 0.0)

                    # 4. 计算 F1-Score (防止分母为0)
                    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    user_f1s[k].append(f1)
                    
                    # 5. 计算 NDCG (留一法下的 DCG 计算)
                    dcg = 0.0
                    for rank, item in enumerate(rec_k):
                        if item in test_set:
                            dcg += 1.0 / np.log2(rank + 2) # rank从0开始，所以是 rank+2
                    user_ndcgs[k].append(dcg) # IDCG 为 1.0，所以 NDCG = DCG
                    
            except Exception as e:
                continue
        
        # 计算平均多指标
        strategy_res = {
            'Strategy': strategy_name, 
            'Valid Users': len(user_precisions[5])
        }
        for k in [5, 10, 20]:
            avg_prec = np.mean(user_precisions[k]) * 100 if user_precisions[k] else 0.0
            avg_recall = np.mean(user_recalls[k]) * 100 if user_recalls[k] else 0.0
            avg_hr = np.mean(user_hit_flags[k]) * 100 if user_hit_flags[k] else 0.0
            avg_f1 = np.mean(user_f1s[k]) * 100 if user_f1s[k] else 0.0
            avg_ndcg = np.mean(user_ndcgs[k]) * 100 if user_ndcgs[k] else 0.0
            
            strategy_res[f'Precision@{k}'] = f"{avg_prec:.1f}%"
            strategy_res[f'Recall@{k}'] = f"{avg_recall:.1f}%"
            strategy_res[f'HitRate@{k}'] = f"{avg_hr:.1f}%"
            strategy_res[f'F1@{k}'] = f"{avg_f1:.1f}%"
            strategy_res[f'NDCG@{k}'] = f"{avg_ndcg:.1f}%"
            
        results.append(strategy_res)
        
        # 打印当前策略的阶段性总结
        print(f"   📊 {strategy_name} 评估完成！有效用户数: {strategy_res['Valid Users']}")
    
    # 创建结果表格
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    print("说明：")
    print("- Precision@K : 推荐列表中属于测试集的比例")
    print("- Recall@K    : 测试集中被成功推荐出来的比例")
    print("- HitRate@K   : 推荐列表中至少包含1部测试集电影的用户比例")
    print("- F1@K        : 准确率和召回率的调和平均指标")
    print("- NDCG@K      : 考虑了推荐顺序相关性的归一化折损累计增益")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)  # 将模糊宽度字符视为双倍宽度
    pd.set_option('display.unicode.east_asian_width', True)  # 将东亚宽度字符视为双倍宽度
    #print(results_df.to_string(index=False))
    console = Console()
    table = Table(title="推荐系统多维度指标评测表", show_header=True, header_style="white", border_style="white")
    for column in results_df.columns:
        table.add_column(column, justify="center")
    for _, row in results_df.iterrows():
        table.add_row(*[str(val) for val in row.values])
    print("\n")
    console.print(table)


    # 找出最佳策略（基于 Precision@5）
    best_idx = results_df['Precision@5'].str.rstrip('%').astype(float).argmax()
    best_strategy = results_df.iloc[best_idx]['Strategy']
    best_hitrate = results_df.iloc[best_idx]['Precision@5']
    
    print("\n" + "=" * 70)
    print(f"🏆 最佳策略 (基于 Precision@5): {best_strategy}")
    print(f"   Precision@5 = {best_hitrate}")
    print("=" * 70)
    
    # 保存结果
    results_df.to_csv('evaluation_results_comprehensive.csv', index=False)
    print("\n✅ 结果已保存至 evaluation_results_comprehensive.csv")
    
    return results_df


if __name__ == "__main__":
    print("=" * 70)
    print("推荐系统评估工具（多指标 & 实时进度版）")
    print("=" * 70)
    print("\n评估方法：留一法")
    print("- 训练集：用户80%的历史评分")
    print("- 测试集：用户20%的历史评分")
    
    results = run_intuitive_evaluation()
    
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)