"""
评估报告页面
需要在项目根目录创建 pages 文件夹
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="评估报告", page_icon="📊", layout="wide")

st.title("📊 推荐系统评估报告")

# 尝试加载已有的评估结果
if os.path.exists('evaluation_results.csv'):
    results_df = pd.read_csv('evaluation_results.csv')
    
    st.subheader("📋 评估结果汇总表")
    st.dataframe(results_df, use_container_width=True)
    
    # 可视化
    st.subheader("📈 指标对比可视化")
    
    metrics_to_plot = ['Precision@K', 'Recall@K', 'F1@K', 'NDCG@K']
    
    for metric in metrics_to_plot:
        col1, col2 = st.columns([1, 1])
        with col1:
            # 按策略分组
            pivot_df = results_df.pivot(index='K', columns='策略', values=metric)
            fig = px.line(
                pivot_df, 
                markers=True,
                title=f'{metric} 对比',
                labels={'value': metric, 'K': 'K值'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 柱状图（K=10）
            k10_data = results_df[results_df['K'] == 10]
            fig2 = px.bar(
                k10_data,
                x='策略',
                y=metric,
                title=f'{metric} (K=10)',
                color='策略',
                text=metric
            )
            fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
    
    # 最佳策略分析
    st.subheader("🏆 最佳策略分析")
    
    k10_results = results_df[results_df['K'] == 10]
    best_f1 = k10_results.loc[k10_results['F1@K'].astype(float).idxmax()]
    
    st.success(f"""
    **综合表现最好的策略**: {best_f1['策略']}
    
    - F1@10: {best_f1['F1@K']}
    - Precision@10: {best_f1['Precision@K']}
    - Recall@10: {best_f1['Recall@K']}
    - NDCG@10: {best_f1['NDCG@K']}
    """)
    
else:
    st.warning("未找到评估结果文件 'evaluation_results.csv'")
    st.info("请先运行 `python run_evaluation.py` 生成评估结果")
    
    # 提供快速运行按钮
    if st.button("🚀 立即运行评估"):
        with st.spinner("正在运行评估，这可能需要几分钟..."):
            import subprocess
            result = subprocess.run(['python', 'run_evaluation.py'], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("评估完成！请刷新页面")
                st.rerun()
            else:
                st.error(f"评估失败: {result.stderr}")