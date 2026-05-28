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
    
    max_test_users = 50
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
        
        # 初始化存储器
        user_precisions = {5: [], 10: [], 20: []}
        user_recalls = {5: [], 10: [], 20: []}
        user_hit_flags = {5: [], 10: [], 20: []}
        user_f1s = {5: [], 10: [], 20: []}
        user_ndcgs = {5: [], 10: [], 20: []}
        user_mrrs = {5: [], 10: [], 20: []}  # ✨ 新增 MRR 存储
        
        for user_id in tqdm(test_users, desc=f"   计算 {strategy_name} 进度", unit="user"):
            train_items, test_items = evaluator.get_user_test_items_loo(user_id, 0.8)
            
            if len(test_items) == 0:
                continue
            
            try:
                # 产生 100 个候选
                recommendations = recommend_func(user_id, 100, train_items)
                if not recommendations:
                    continue
                test_set = set(test_items)
                
                # 计算各 K 值的多指标
                for k in [5, 10, 20]:
                    rec_k_list = recommendations[:k]  
                    rec_k_set = set(rec_k_list)
                    hits = len(rec_k_set & test_set)
                    
                    # 1. Precision & Recall
                    prec = hits / k
                    rec = hits / len(test_set)
                    
                    user_precisions[k].append(prec)
                    user_recalls[k].append(rec)
                    
                    # 2. HitRate
                    user_hit_flags[k].append(1.0 if hits > 0 else 0.0)

                    # 3. F1-Score (修复了未定义变量 Bug)
                    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    user_f1s[k].append(f1)
                    
                    # 4. NDCG (修复了无序 Bug，并进行了标准归一化)
                    dcg = 0.0
                    for rank, item in enumerate(rec_k_list):
                        if item in test_set:
                            dcg += 1.0 / np.log2(rank + 2)
                    
                    # 计算当前 K 下的理想 IDCG
                    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(test_set)))])
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    user_ndcgs[k].append(ndcg)
                    
                    # 5. MRR (✨ 完美新增)
                    mrr = 0.0
                    for rank, item in enumerate(rec_k_list):
                        if item in test_set:
                            mrr = 1.0 / (rank + 1)
                            break  # 找到第一个命中的就停止
                    user_mrrs[k].append(mrr)
                    
            except Exception as e:
                # print(f"Error: {e}") # 调试时可打开
                continue
        
        # 计算平均多指标并格式化
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
            avg_mrr = np.mean(user_mrrs[k]) * 100 if user_mrrs[k] else 0.0
            
            strategy_res[f'Prec@{k}'] = f"{avg_prec:.1f}%"
            strategy_res[f'Rec@{k}'] = f"{avg_recall:.1f}%"
            strategy_res[f'HR@{k}'] = f"{avg_hr:.1f}%"
            strategy_res[f'F1@{k}'] = f"{avg_f1:.1f}%"
            strategy_res[f'NDCG@{k}'] = f"{avg_ndcg:.1f}%"
            strategy_res[f'MRR@{k}'] = f"{avg_mrr:.1f}%"
            
        results.append(strategy_res)
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
    table = Table(title="🏆 推荐系统核心指标评测表 (K=10)", show_header=True, header_style="bold magenta", border_style="dim")
    
    # 定义紧凑的表头
    table.add_column("Strategy", justify="left", style="cyan")
    table.add_column("Users", justify="center")
    table.add_column("Prec@10", justify="center")
    table.add_column("Rec@10", justify="center")
    table.add_column("HR@10", justify="center")
    table.add_column("F1@10", justify="center")
    table.add_column("NDCG@10 (排序)", justify="center", style="green")
    table.add_column("MRR@10 (顺位)", justify="center", style="green")

    # 提取 K=10 的数据填入行
    for res in results:
        table.add_row(
            res['Strategy'],
            str(res['Valid Users']),
            res['Prec@10'],
            res['Rec@10'],
            res['HR@10'],
            res['F1@10'],
            res['NDCG@10'],
            res['MRR@10']
        )
        
    print("\n")
    console.print(table)
    
    # 找出最佳策略（基于 NDCG@10 更有说服力）
    best_idx = results_df['NDCG@10'].str.rstrip('%').astype(float).argmax()
    best_strategy = results_df.iloc[best_idx]['Strategy']
    best_ndcg = results_df.iloc[best_idx]['NDCG@10']
    
    print("\n" + "=" * 60)
    print(f"🥇 综合最优策略 (基于 NDCG@10):\n {best_strategy} (NDCG@10 = {best_ndcg})")
    print("=" * 60)
    
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