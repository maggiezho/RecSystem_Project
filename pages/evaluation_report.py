# pages/evaluation_report.py - 极致精简与优雅多维表达版
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="评估报告", page_icon="📊", layout="wide")

st.title("📊 推荐系统多维度评估报告")
st.markdown("---")

# 期望读取的新版综合评估结果文件
TARGET_FILE = 'evaluation_results_comprehensive.csv'

if os.path.exists(TARGET_FILE):
    # 1. 读取原始数据
    df_raw = pd.read_csv(TARGET_FILE)
    
    # 使用 Tabs 标签页与 Metric 卡片精简表达
    st.subheader("📋 评估数据多维透视")
    
    tab_k10, tab_k5, tab_k20, tab_raw = st.tabs([
        "🎯 核心推荐 (K=10)", 
        "⚡ 快速浏览 (K=5)", 
        "🔍 深度沉浸 (K=20)", 
        "💾 原始宽表数据"
    ])
    
    with tab_k10:
        st.markdown("##### 💡 黄金标准 K=10 下的策略表现")
        # 仅抽取 K=10 相关的核心指标
        cols_k10 = ['Strategy', 'Valid Users', 'Prec@10', 'Rec@10', 'HR@10', 'F1@10', 'NDCG@10', 'MRR@10']
        df_k10_clean = df_raw[cols_k10].rename(columns={
            'Strategy': '策略', 'Valid Users': '有效测试用户', 'Prec@10': '准确率@10',
            'Rec@10': '召回率@10', 'HR@10': '命中率@10', 'F1@10': 'F1值@10',
            'NDCG@10': 'NDCG@10 (关键排序)', 'MRR@10': 'MRR@10 (首中顺位)'
        })
        st.dataframe(df_k10_clean, use_container_width=True, hide_index=True)
        
        # 用大字号卡片高亮全场综合表现最强的算法
        st.markdown("##### 🏆 K=10 策略高光审计")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        # 基于排序黄金指标 NDCG@10 寻找最优解
        best_idx_k10 = df_raw['NDCG@10'].str.rstrip('%').astype(float).argmax()
        strat_best = df_raw.iloc[best_idx_k10]
        
        m_col1.metric("🥇 综合首选策略", strat_best['Strategy'])
        m_col2.metric("排序能力 (NDCG@10)", strat_best['NDCG@10'])
        m_col3.metric("首中速度 (MRR@10)", strat_best['MRR@10'])
        m_col4.metric("综合平衡 (F1@10)", strat_best['F1@10'])

    with tab_k5:
        st.markdown("##### ⚡ 当推荐列表较短 (K=5) 时的策略表现")
        cols_k5 = ['Strategy', 'Prec@5', 'Rec@5', 'HR@5', 'F1@5', 'NDCG@5', 'MRR@5']
        st.dataframe(df_raw[cols_k5].rename(columns={'Strategy': '策略'}), use_container_width=True, hide_index=True)

    with tab_k20:
        st.markdown("##### 🔍 当推荐列表较长 (K=20) 时的策略表现")
        cols_k20 = ['Strategy', 'Prec@20', 'Rec@20', 'HR@20', 'F1@20', 'NDCG@20', 'MRR@20']
        st.dataframe(df_raw[cols_k20].rename(columns={'Strategy': '策略'}), use_container_width=True, hide_index=True)
        
    with tab_raw:
        st.markdown("##### 💾 后台生成的完整原始宽表")
        st.dataframe(df_raw, use_container_width=True)
    
    # 2. 数据清洗与重构 (将宽表平铺转为长表，用于 Plotly 动态画图)
    parsed_data = []
    for _, row in df_raw.iterrows():
        strategy = row['Strategy']
        for col in df_raw.columns:
            if '@' in col:
                metric_name, k_val = col.split('@')
                val_str = str(row[col]).rstrip('%')
                try:
                    val_float = float(val_str)
                except ValueError:
                    val_float = 0.0
                
                parsed_data.append({
                    '策略': strategy,
                    '指标': metric_name,
                    'K值': int(k_val),
                    '数值(%)': val_float
                })
    df_plot = pd.DataFrame(parsed_data)
    
    # 3. 动态指标可视化
    st.markdown("---")
    st.subheader("📈 指标对比可视化")
    
    all_metrics = df_plot['指标'].unique().tolist()
    selected_metric = st.selectbox(
        "切换查看的评估核心指标：", 
        all_metrics, 
        index=all_metrics.index('NDCG') if 'NDCG' in all_metrics else 0,
        help="Prec: 准确率 | Rec: 召回率 | HR: 命中率 | F1: 综合平衡指标 | NDCG: 考虑顺序的归一化增益 | MRR: 首个正确结果的平均倒数排名"
    )
    
    df_metric = df_plot[df_plot['指标'] == selected_metric]
    col1, col2 = st.columns([1.1, 0.9])
    
    with col1:
        fig_line = px.line(
            df_metric, x='K值', y='数值(%)', color='策略', markers=True,
            title=f'各策略 {selected_metric}@K 随 K 值变化趋势',
            labels={'数值(%)': f'{selected_metric} (%)', 'K值': '推荐列表长度 (K)'},
            template="plotly_white"
        )
        fig_line.update_layout(xaxis=dict(tickmode='array', tickvals=[5, 10, 20]))
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col2:
        df_k10_plot = df_metric[df_metric['K值'] == 10]
        if not df_k10_plot.empty:
            fig_bar = px.bar(
                df_k10_plot, x='策略', y='数值(%)', color='策略',
                title=f'K=10 时各策略的 {selected_metric} 直观对比',
                labels={'数值(%)': f'{selected_metric} (%)'}, text='数值(%)',
                template="plotly_white"
            )
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

    # 4. 雷达图多维能力画像
    st.markdown("---")
    st.subheader("🕸️ 策略综合能力画像 (K=10 归一化视图)")
    st.caption("注：由于各指标绝对值差距过大，本图采用最大值归一化，展示各策略在不同维度的相对优劣。")
    
    df_radar_base = df_plot[df_plot['K值'] == 10].copy()
    if not df_radar_base.empty:
        # 对每个指标独立进行归一化
        df_radar_base['展示数值'] = df_radar_base.groupby('指标')['数值(%)'].transform(
            lambda x: (x / x.max() * 100) if x.max() > 0 else 0
        )
        
        # 固定的轴顺序
        desired_order = ['Prec', 'Rec', 'F1', 'NDCG', 'MRR', 'HR']
        strategy_scores = df_radar_base.groupby('策略')['展示数值'].sum().reset_index()
        # 按照总分从大到小排序。总分大(面积大)的先画，放在最底层；总分小(面积小)的后画，浮在最顶层
        sorted_strategies = strategy_scores.sort_values(by='展示数值', ascending=False)['策略'].tolist()
        
        fig_radar = go.Figure()
        
        # 按照排序后的顺序循环添加图层
        for strategy in sorted_strategies:
            df_strat = df_radar_base[df_radar_base['策略'] == strategy]
            df_strat = df_strat.set_index('指标').reindex(desired_order).reset_index()
            df_strat = df_strat.dropna(subset=['展示数值'])
            
            r_vals = df_strat['展示数值'].tolist()
            theta_vals = df_strat['指标'].tolist()
            if r_vals:
                r_vals.append(r_vals[0])
                theta_vals.append(theta_vals[0])
                
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill='toself',
                name=strategy,
                # 提示文字格式优化：清晰展示相对比例和具体指标名字
                hovertemplate=f"<b>{strategy}</b><br>相对水平: %{{r}}:.1f%<br><extra></extra>"
            ))
            
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(128, 128, 128, 0.04)",  
                radialaxis=dict(
                    visible=True, 
                    range=[0, 110],
                    showticklabels=False,          
                    showline=False,                
                    tickvals=[25, 50, 75, 100],
                    gridcolor="rgba(128, 128, 128, 0.25)", 
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, weight="bold"), 
                    gridcolor="rgba(128, 128, 128, 0.25)"  
                )
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=13, weight="bold"),
                bgcolor="rgba(0,0,0,0)"
            ),
            title=dict(
                text="K=10时 各策略多维能力均衡度对比 (智能透视交互版)",
                font=dict(size=16, weight="bold")
            ),
            # ✨【突破核心 2/2】：改变悬浮卡片的触发模式
            # hovermode="closest" 配合刚才的面积排序可以完美选中任何小图形的顶点
            hovermode="closest" 
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
else:
    st.warning(f"❌ 未找到新版评估结果文件 '{TARGET_FILE}'")
    st.info("请先前往终端或项目根目录下运行 `python run_evaluation.py` 生成全新的综合评估数据。")
    
    if st.button("🚀 立即在后台拉起多指标评估"):
        with st.spinner("正在基于 MovieLens 25M 数据集运行深度留一法评估，请耐心等待..."):
            import subprocess
            result = subprocess.run(['python', 'run_evaluation.py'], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("🎉 离线多指标评估成功完成！正在刷新交互报告...")
                st.rerun()
            else:
                st.error(f"评估脚本运行失败，错误日志:\n{result.stderr}")