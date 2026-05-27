import streamlit as st
import pandas as pd
import pickle
import time
import os
import sys
import warnings
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 抑制警告
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ['STREAMLIT_RUNNING'] = 'true'

# 设置页面配置
st.set_page_config(
    page_title="MovieLens 智能推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 导入召回模块
from recall import HybridRecall, ItemCF, UserCF, Popularity


# ==================== 缓存加载函数 ====================
@st.cache_data
def load_base_data():
    """加载基础数据"""
    movies = pd.read_parquet('processed/movies.parquet')
    ratings = pd.read_parquet('processed/ratings.parquet')
    return movies, ratings

@st.cache_data
def load_features():
    """加载特征数据"""
    user_features = pd.read_parquet('processed/user_features.parquet')
    movie_features = pd.read_parquet('processed/movie_features.parquet')
    return user_features, movie_features

@st.cache_resource
def load_ranking_model():
    """加载排序模型"""
    model_path = 'models/ranking_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_recallers():
    """预加载召回器"""
    from recall.item_cf import ItemCF
    from recall.user_cf import UserCF
    from recall.popularity import Popularity
    
    itemcf = ItemCF()
    usercf = UserCF()
    popularity = Popularity()
    
    return {
        'itemcf': itemcf,
        'usercf': usercf,
        'popularity': popularity
    }


# ==================== 辅助函数 ====================
def get_user_history(user_id, ratings_df, movies_df, n=10):
    """获取用户观看历史"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp', ascending=False)
    user_history = pd.merge(user_ratings, movies_df, on='movieId')
    return user_history.head(n)

def get_user_stats(user_id, ratings_df):
    """获取用户统计信息"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if len(user_ratings) == 0:
        return None
    return {
        'total_ratings': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'rating_std': user_ratings['rating'].std(),
        'min_rating': user_ratings['rating'].min(),
        'max_rating': user_ratings['rating'].max()
    }

def plot_rating_distribution(user_id, ratings_df):
    """绘制用户评分分布图"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if len(user_ratings) == 0:
        return None
    
    fig = px.histogram(
        user_ratings, x='rating', 
        title=f'用户 {user_id} 的评分分布',
        labels={'rating': '评分', 'count': '数量'},
        color_discrete_sequence=['#FF4B4B'],
        nbins=10
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_genre_distribution(recommendations_df):
    """绘制推荐结果的体裁分布"""
    if recommendations_df.empty:
        return None
    
    all_genres = []
    for genres in recommendations_df['genres'].dropna():
        all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.values, y=genre_counts.index,
        orientation='h',
        title='推荐电影体裁分布 (Top 10)',
        labels={'x': '数量', 'y': '体裁'},
        color_discrete_sequence=['#FF4B4B']
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    # 让长条图按数量从大到小从上往下排
    fig.update_yaxes(autorange="reversed") 
    return fig


# ==================== 主界面 ====================
def main():
    # 加载数据
    with st.spinner("正在加载数据..."):
        movies_df, ratings_df = load_base_data()
        user_features_df, movie_features_df = load_features()
        ranking_model = load_ranking_model()
        recallers = load_recallers()
    
    # ==================== 【极其严格】的状态机初始化 ====================
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = 1
    if 'rec_results' not in st.session_state:
        st.session_state.rec_results = None
        
    # ==================== 侧边栏 ====================
    st.sidebar.title("🎬 推荐控制台")
    st.sidebar.info(
        "这是一个基于 MovieLens 25M 数据集的双层推荐系统。\n\n"
        "- **系统架构**: 召回层 + 精排层\n"
        "- **召回算法**: ItemCF + UserCF + 热门电影\n"
        "- **精排模型**: LightGBM 排序模型"
    )
    
    # 用户输入
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 用户设置")
    user_id = st.sidebar.number_input("用户 ID", min_value=1, max_value=162541, value=1, step=1)
    top_k = st.sidebar.slider("推荐数量", 5, 30, 10)
    
    # 【核心修复 1】只要用户换了 ID 或者调整了 K 值，清空缓存，要求重新点击生成
    if st.session_state.current_user != user_id:
        st.session_state.show_recommendations = False
        st.session_state.rec_results = None
        st.session_state.current_user = user_id
    
    # 召回策略选择
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 召回策略")
    recall_strategy = st.sidebar.selectbox(
        "选择召回算法",
        ["混合召回 (ItemCF + UserCF + 热门)", "仅 ItemCF", "仅 UserCF", "仅热门电影"],
        help="混合召回综合多种策略，推荐效果最佳"
    )
    
    # 混合召回权重调节
    weights = {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}
    if recall_strategy == "混合召回 (ItemCF + UserCF + 热门)":
        st.sidebar.markdown("**权重调节**")
        col1, col2, col3 = st.sidebar.columns(3)
        weights['itemcf'] = col1.slider("ItemCF", 0.0, 1.0, 0.5, 0.05)
        weights['usercf'] = col2.slider("UserCF", 0.0, 1.0, 0.3, 0.05)
        weights['pop'] = col3.slider("热门", 0.0, 1.0, 0.2, 0.05)
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
    
    # 高级选项
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔧 高级选项"):
        use_ranking = st.checkbox("启用精排模型", value=(ranking_model is not None))
        recall_top_n = st.slider("召回候选数量", 50, 300, 100)
        filter_watched = st.checkbox("过滤已看过的电影", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 评估面板")
    show_evaluation = st.sidebar.checkbox("显示离线评估结果", value=False)
    if show_evaluation:
        eval_k = st.sidebar.selectbox("评估K值", [5, 10, 20], index=1)
        eval_users_sample = st.sidebar.slider("测试用户数", 50, 500, 200, step=50)
    
    show_monitor = st.sidebar.checkbox("📊 数据洞察面板")
    
    # ==================== 主内容区 ====================
    st.title("🎬 MovieLens 智能推荐系统")
    st.markdown(f"### 当前模拟用户: **User {user_id}**")
    
    user_stats = get_user_stats(user_id, ratings_df)
    
    # 用户信息卡片
    col_info1, col_info2, col_info3, col_info4, col_info5 = st.columns(5)
    if user_stats:
        col_info1.metric("📊 评价数量", user_stats['total_ratings'])
        col_info2.metric("⭐ 平均评分", f"{user_stats['avg_rating']:.2f}")
        col_info3.metric("📈 评分标准差", f"{user_stats['rating_std']:.2f}")
        col_info4.metric("🔝 最高评分", user_stats['max_rating'])
        col_info5.metric("📉 最低评分", user_stats['min_rating'])
    else:
        col_info1.metric("📊 评价数量", 0)
        col_info2.metric("⭐ 平均评分", "N/A")
        col_info3.metric("👤 用户类型", "新用户")
        col_info4.metric("💡 推荐策略", "热门电影")
        col_info5.metric("🔄 冷启动", "已启用")
        st.info("💡 这是一个新用户，系统将使用热门电影进行冷启动推荐")
        
    # ==================== 【核心修复 2】后台静默计算逻辑 ====================
    # 我们把触发动作和计算抽离出来，不放在 columns 块里，避免渲染刷新冲突
    
    # 定义点击按钮的回调函数，极其干净
    def trigger_recommendation():
        st.session_state.show_recommendations = not st.session_state.show_recommendations
        if st.session_state.show_recommendations:
            st.session_state.rec_results = None

    # 如果状态为 True 且还没有计算结果，立刻在后台计算并存入 session_state
    if st.session_state.show_recommendations and st.session_state.rec_results is None:
        with st.spinner("🚀 正在后台运行召回与精排算法..."):
            start_time = time.time()
            try:
                # 1. 召回阶段
                if recall_strategy == "仅 ItemCF":
                    recaller = recallers['itemcf']
                    candidate_ids = recaller.recall(user_id, recall_top_n) if recaller.sim_matrix is not None else []
                elif recall_strategy == "仅 UserCF":
                    recaller = recallers['usercf']
                    candidate_ids = recaller.recall(user_id, recall_top_n) if recaller.sim_matrix is not None else []
                elif recall_strategy == "仅热门电影":
                    recaller = recallers['popularity']
                    candidate_ids = recaller.recall(user_id, recall_top_n)
                else:
                    hybrid = HybridRecall(weights=weights)
                    candidate_ids = hybrid.recall(user_id, recall_top_n)
                
                # 过滤已看
                if filter_watched and user_stats and user_stats['total_ratings'] > 0:
                    watched_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())
                    candidate_ids = [mid for mid in candidate_ids if mid not in watched_movies]
                
                # 2. 精排阶段
                ranking_scores = {}
                use_ranking_flag = use_ranking and ranking_model is not None and len(candidate_ids) > 0
                
                if use_ranking_flag:
                    test_data = pd.DataFrame({
                        'userId': [user_id] * len(candidate_ids),
                        'movieId': candidate_ids
                    })
                    test_data = pd.merge(test_data, user_features_df, on='userId', how='left')
                    test_data = pd.merge(test_data, movie_features_df, on='movieId', how='left')
                    
                    feature_cols = [
                        'user_avg_rating', 'user_rating_std', 'user_rating_count',
                        'movie_avg_rating', 'movie_rating_std', 'movie_rating_count', 'year'
                    ]
                    
                    for col in feature_cols:
                        if col in test_data.columns:
                            test_data[col] = test_data[col].fillna(test_data[col].median())
                    
                    try:
                        probs = ranking_model.predict_proba(test_data[feature_cols])
                        if hasattr(probs, 'shape') and len(probs.shape) > 1:
                            probs = probs[:, 1]
                        
                        probs_list = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
                        test_data['score'] = probs_list
                        ranking_scores = dict(zip(candidate_ids, probs_list))
                        
                        final_ids = test_data.sort_values('score', ascending=False).head(top_k)['movieId'].tolist()
                    except Exception as e:
                        final_ids = candidate_ids[:top_k]
                else:
                    final_ids = candidate_ids[:top_k]
                
                # 3. 获取详情
                recommendations = movies_df[movies_df['movieId'].isin(final_ids)].copy()
                recommendations['rank'] = recommendations['movieId'].apply(lambda x: final_ids.index(x) if x in final_ids else 999)
                recommendations = recommendations.sort_values('rank').drop('rank', axis=1)
                duration = time.time() - start_time
                
                # 存入独享状态槽
                st.session_state.rec_results = {
                    'recommendations': recommendations,
                    'duration': duration,
                    'candidate_count': len(candidate_ids),
                    'ranking_scores': ranking_scores,
                    'use_ranking_flag': use_ranking_flag
                }
            except Exception as e:
                st.error(f"推荐算法运行失败: {str(e)}")

    # ==================== 两栏页面渲染布局 ====================
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📜 观看历史")
        user_history = get_user_history(user_id, ratings_df, movies_df, 10)
        if not user_history.empty:
            display_history = user_history[['title', 'genres', 'rating', 'timestamp']].copy()
            display_history['timestamp'] = pd.to_datetime(display_history['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            display_history.columns = ['电影名称', '体裁', '评分', '观看时间']
            st.dataframe(display_history, use_container_width=True, height=250)
            
            rating_fig = plot_rating_distribution(user_id, ratings_df)
            if rating_fig:
                st.plotly_chart(rating_fig, use_container_width=True)
        else:
            st.info("暂无观看历史，这是一位新用户")
    
    with col_right:
        st.subheader("🎯 个性化推荐")
        
        if st.session_state.show_recommendations:
            btn_label = "❌ 收起推荐面板"
            btn_type = "secondary"  # 已展开时，按钮变成低调的灰色
        else:
            btn_label = "🎯 生成个性化推荐"
            btn_type = "primary"    # 未展开时，按钮是高亮的红色/蓝色
        
        # 渲染按钮，绑定动态变量
        st.button(
            label=btn_label, 
            type=btn_type, 
            use_container_width=True, 
            key="generate_rec_btn", 
            on_click=trigger_recommendation
        )

        # 纯粹的渲染逻辑：只要状态机里有货，不管页面怎么刷新，雷打不动地把它画出来！
        if st.session_state.show_recommendations and st.session_state.rec_results is not None:
            res = st.session_state.rec_results
            recs = res['recommendations']
            
            st.success(f"✅ 耗时: {res['duration']:.2f}s | 候选数: {res['candidate_count']} | 推荐数: {len(recs)}")
            
            if recs.empty:
                st.warning("没有找到推荐结果")
            else:
                for idx, (display_idx, row) in enumerate(recs.iterrows()):
                    with st.expander(f"**{idx+1}. {row['title']}**", expanded=(idx < 3)):
                        col_m1, col_m2 = st.columns([2, 1])
                        with col_m1:
                            st.write(f"📅 **年份**: {int(row['year']) if pd.notna(row['year']) else '未知'}")
                            st.write(f"🏷️ **体裁**: {row['genres']}")
                        with col_m2:
                            st.write(f"🎬 电影 ID: `{row['movieId']}`")
                            if res['use_ranking_flag'] and res['ranking_scores'] and row['movieId'] in res['ranking_scores']:
                                try:
                                    score_raw = res['ranking_scores'][row['movieId']]
                                    score = float(score_raw[0]) if hasattr(score_raw, '__len__') and not isinstance(score_raw, (str, float, int)) else float(score_raw)
                                except:
                                    score = 0.5
                                score = max(0.0, min(1.0, score))
                                st.progress(score)
                                st.caption(f"推荐置信度: {score:.3f}")
                
                genre_fig = plot_genre_distribution(recs)
                if genre_fig:
                    st.plotly_chart(genre_fig, use_container_width=True)

        # ==================== 评估结果面板 ====================
        if show_evaluation:
            st.divider()
            st.subheader("📊 离线评估结果")
            
            # 同样给评估按钮加上唯一 key 锁定状态
            if st.button("⚡ 开始运行离线评估 (耗时较长)", use_container_width=True, key="run_eval_btn"):
                with st.spinner("正在运行多策略离线评估..."):
                    try:
                        from evaluate import RecEvaluator
                        from recall import get_recommendation_func
                        
                        evaluator = RecEvaluator('processed/ratings.parquet')
                        ratings_local = ratings_df
                        user_rating_counts = ratings_local.groupby('userId').size()
                        test_users_all = user_rating_counts[user_rating_counts >= 50].index.tolist()
                        
                        if len(test_users_all) > eval_users_sample:
                            test_users = np.random.choice(test_users_all, eval_users_sample, replace=False).tolist()
                        else:
                            test_users = test_users_all
                        
                        strategies = {
                            'ItemCF': get_recommendation_func('itemcf'),
                            'UserCF': get_recommendation_func('usercf'),
                            '热门电影': get_recommendation_func('popularity'),
                            '混合召回': get_recommendation_func('hybrid', weights),
                        }
                        
                        eval_results = {}
                        for strategy_name, recommend_func in strategies.items():
                            results = evaluator.evaluate_recommendations(
                                recommend_func, test_users, [eval_k], train_ratio=0.8, verbose=False
                            )
                            eval_results[strategy_name] = results[eval_k]
                        
                        comparison_data = []
                        for strategy_name, metrics in eval_results.items():
                            comparison_data.append({
                                '策略': strategy_name,
                                'Precision': f"{metrics['precision']:.4f}",
                                'Recall': f"{metrics['recall']:.4f}",
                                'F1': f"{metrics['f1']:.4f}",
                                'NDCG': f"{metrics['ndcg']:.4f}",
                                'Hit Rate': f"{metrics['hit_rate']:.4f}",
                                'MRR': f"{metrics['mrr']:.4f}"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        st.markdown("#### 📈 指标对比图")
                        plot_data = []
                        for strategy_name, metrics in eval_results.items():
                            for metric_name in ['precision', 'recall', 'f1', 'ndcg']:
                                plot_data.append({
                                    '策略': strategy_name,
                                    '指标': metric_name.upper() + '@K',
                                    '值': metrics[metric_name]
                                })
                        
                        plot_df = pd.DataFrame(plot_data)
                        fig = px.bar(
                            plot_df, x='策略', y='值', color='指标', barmode='group',
                            title=f'不同召回策略在 K={eval_k} 时的表现对比',
                            color_discrete_sequence=px.colors.qualitative.Set2
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"评估失败: {str(e)}")
            else:
                st.info("💡 点击上方评估按钮。上方的个性化推荐卡片现在已被全局状态锁定，绝不会再消失。")

    # ==================== 数据洞察面板 ====================
    if show_monitor:
        st.divider()
        st.subheader("📊 数据洞察面板")
        
        col1, col2, col3 = st.columns(3)
        col1.success("✅ ItemCF 已就绪" if recallers['itemcf'].sim_matrix is not None else "❌ ItemCF 未训练")
        col2.success("✅ UserCF 已就绪" if recallers['usercf'].sim_matrix is not None else "❌ UserCF 未训练")
        col3.success("✅ LightGBM 精排模型已加载" if ranking_model is not None else "⚠️ 精排模型不可用")
        
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("🎬 电影库规模", f"{len(movies_df):,} 部")
        col_info2.metric("👥 用户规模", f"{ratings_df['userId'].nunique():,} 人")
        col_info3.metric("⭐ 评分数据", f"{len(ratings_df):,} 条")


if __name__ == "__main__":
    main()