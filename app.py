import streamlit as st
import pandas as pd
import pickle
import time
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 设置页面配置（必须是第一个Streamlit命令）
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
    """预加载召回器（避免重复加载）"""
    return {
        'hybrid': HybridRecall(),
        'itemcf': ItemCF(),
        'usercf': UserCF(),
        'popularity': Popularity()
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
    fig.update_layout(height=400)
    return fig

def plot_genre_distribution(recommendations_df):
    """绘制推荐结果的体裁分布"""
    if recommendations_df.empty:
        return None
    
    # 解析体裁（体裁用|分隔）
    all_genres = []
    for genres in recommendations_df['genres'].dropna():
        all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.values, y=genre_counts.index,
        orientation='h',
        title='推荐电影体裁分布',
        labels={'x': '数量', 'y': '体裁'},
        color_discrete_sequence=['#FF4B4B']
    )
    fig.update_layout(height=400)
    return fig


# ==================== 主界面 ====================
def main():
    # 加载数据
    with st.spinner("正在加载数据..."):
        movies_df, ratings_df = load_base_data()
        user_features_df, movie_features_df = load_features()
        ranking_model = load_ranking_model()
        recallers = load_recallers()
    
    # ==================== 侧边栏 ====================
    st.sidebar.title("🎬 推荐控制台")
    st.sidebar.info(
        "这是一个基于 MovieLens 25M 数据集的双层推荐系统\n\n"
        "**架构**: 召回层 + 精排层\n"
        "**召回**: ItemCF + UserCF + 热门电影\n"
        "**精排**: LightGBM 排序模型"
    )
    
    # 用户输入
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 用户设置")
    user_id = st.sidebar.number_input("用户 ID", min_value=1, max_value=162541, value=1, step=1)
    top_k = st.sidebar.slider("推荐数量", 5, 30, 10)
    
    # 召回策略选择
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 召回策略")
    recall_strategy = st.sidebar.selectbox(
        "选择召回算法",
        ["混合召回 (ItemCF + UserCF + 热门)", "仅 ItemCF", "仅 UserCF", "仅热门电影"],
        help="混合召回综合多种策略，推荐效果最佳"
    )
    
    # 混合召回权重调节（仅当选择混合召回时显示）
    weights = {'itemcf': 0.5, 'usercf': 0.3, 'pop': 0.2}
    if recall_strategy == "混合召回 (ItemCF + UserCF + 热门)":
        st.sidebar.markdown("**权重调节**")
        col1, col2, col3 = st.sidebar.columns(3)
        weights['itemcf'] = col1.slider("ItemCF", 0.0, 1.0, 0.5, 0.05)
        weights['usercf'] = col2.slider("UserCF", 0.0, 1.0, 0.3, 0.05)
        weights['pop'] = col3.slider("热门", 0.0, 1.0, 0.2, 0.05)
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
    
    # 高级选项
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔧 高级选项"):
        use_ranking = st.checkbox("启用精排模型", value=(ranking_model is not None))
        recall_top_n = st.slider("召回候选数量", 50, 300, 100)
        filter_watched = st.checkbox("过滤已看过的电影", value=True)
    
    # 系统监控开关
    show_monitor = st.sidebar.checkbox("📊 系统监控面板")
    
    # ==================== 主内容区 ====================
    st.title("🎬 MovieLens 智能推荐系统")
    st.markdown(f"### 当前模拟用户: **User {user_id}**")
    
    # 获取用户统计
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
    
    # ==================== 两栏布局 ====================
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📜 观看历史")
        user_history = get_user_history(user_id, ratings_df, movies_df, 10)
        if not user_history.empty:
            # 展示历史表格
            display_history = user_history[['title', 'genres', 'rating', 'timestamp']].copy()
            # 转换时间戳
            display_history['timestamp'] = pd.to_datetime(display_history['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            display_history.columns = ['电影名称', '体裁', '评分', '观看时间']
            st.dataframe(display_history, use_container_width=True, height=300)
            
            # 评分分布图
            rating_fig = plot_rating_distribution(user_id, ratings_df)
            if rating_fig:
                st.plotly_chart(rating_fig, use_container_width=True)
        else:
            st.info("暂无观看历史，这是一位新用户")
    
    with col_right:
        # ==================== 推荐触发 ====================
        if st.button("🎯 生成个性化推荐", type="primary", use_container_width=True):
            with st.spinner("🚀 正在运行召回与精排算法..."):
                start_time = time.time()
                
                try:
                    # 1. 根据策略选择召回器
                    if recall_strategy == "仅 ItemCF":
                        recaller = recallers['itemcf']
                        candidate_ids = recaller.recall(user_id, recall_top_n) if recaller.sim_matrix else []
                    elif recall_strategy == "仅 UserCF":
                        recaller = recallers['usercf']
                        candidate_ids = recaller.recall(user_id, recall_top_n) if recaller.sim_matrix else []
                    elif recall_strategy == "仅热门电影":
                        recaller = recallers['popularity']
                        candidate_ids = recaller.recall(user_id, recall_top_n)
                    else:  # 混合召回
                        hybrid = HybridRecall(weights=weights)
                        candidate_ids = hybrid.recall(user_id, recall_top_n)
                    
                    # 过滤已看过的电影
                    if filter_watched and user_stats and user_stats['total_ratings'] > 0:
                        watched_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())
                        candidate_ids = [mid for mid in candidate_ids if mid not in watched_movies]
                    
                    # 2. 精排阶段
                    if use_ranking and ranking_model is not None and len(candidate_ids) > 0:
                        # 准备特征
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
                        
                        # 填充缺失值
                        for col in feature_cols:
                            if col in test_data.columns:
                                test_data[col] = test_data[col].fillna(test_data[col].median())
                        
                        # 预测
                        probs = ranking_model.predict_proba(test_data[feature_cols])[:, 1]
                        test_data['score'] = probs
                        
                        # 排序取top
                        final_ids = test_data.sort_values('score', ascending=False).head(top_k)['movieId'].tolist()
                    else:
                        # 直接返回召回结果
                        final_ids = candidate_ids[:top_k]
                    
                    # 3. 获取推荐电影详情
                    recommendations = movies_df[movies_df['movieId'].isin(final_ids)].copy()
                    # 保持排序顺序
                    recommendations['rank'] = recommendations['movieId'].apply(lambda x: final_ids.index(x) if x in final_ids else 999)
                    recommendations = recommendations.sort_values('rank').drop('rank', axis=1)
                    
                    duration = time.time() - start_time
                    
                    # 显示成功信息
                    st.success(f"✅ 推荐完成！耗时: {duration:.2f} 秒 | 召回候选数: {len(candidate_ids)} | 最终推荐: {len(recommendations)}")
                    
                    # 展示推荐结果
                    st.subheader("🎯 推荐结果")
                    
                    if recommendations.empty:
                        st.warning("没有找到推荐结果，请尝试其他策略")
                    else:
                        for idx, row in recommendations.iterrows():
                            with st.expander(f"**{idx+1}. {row['title']}**", expanded=(idx < 3)):
                                col_m1, col_m2 = st.columns([3, 1])
                                with col_m1:
                                    st.write(f"📅 年份: {int(row['year']) if pd.notna(row['year']) else '未知'}")
                                    st.write(f"🏷️ 体裁: {row['genres']}")
                                with col_m2:
                                    st.write(f"🎬 电影ID: {row['movieId']}")
                                    if use_ranking and ranking_model is not None and 'score' in locals():
                                        st.progress(min(1.0, probs[idx] if idx < len(probs) else 0.5))
                                        st.caption(f"相关度评分: {probs[idx]:.3f}" if idx < len(probs) else "")
                    
                    # 体裁分布图
                    if not recommendations.empty:
                        genre_fig = plot_genre_distribution(recommendations)
                        if genre_fig:
                            st.plotly_chart(genre_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"推荐失败: {str(e)}")
                    st.info("提示：如果是新用户，系统会自动使用热门电影推荐")
    
    # ==================== 系统监控面板 ====================
    if show_monitor:
        st.divider()
        st.subheader("📈 系统监控面板")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        # 数据统计
        col_m1.metric("📊 电影总数", f"{len(movies_df):,}")
        col_m2.metric("⭐ 评分总数", f"{len(ratings_df):,}")
        col_m3.metric("👥 用户总数", f"{ratings_df['userId'].nunique():,}")
        col_m4.metric("🎯 召回模型", recall_strategy.split()[0])
        
        # 模型状态
        st.markdown("#### 🤖 模型状态")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.success("✅ ItemCF" if recallers['itemcf'].sim_matrix is not None else "❌ ItemCF")
        col_s2.success("✅ UserCF" if recallers['usercf'].sim_matrix is not None else "❌ UserCF")
        col_s3.success("✅ 精排模型" if ranking_model is not None else "⚠️ 精排模型")
        
        # 热门电影Top10
        st.markdown("#### 🔥 热门电影 Top 10")
        popular = recallers['popularity'].popular_movies[:10] if recallers['popularity'].popular_movies else []
        if popular:
            popular_movies = movies_df[movies_df['movieId'].isin(popular)].copy()
            popular_movies['popularity_rank'] = popular_movies['movieId'].apply(lambda x: popular.index(x) + 1 if x in popular else 999)
            popular_movies = popular_movies.sort_values('popularity_rank')
            st.dataframe(popular_movies[['title', 'genres', 'year']], use_container_width=True)


if __name__ == "__main__":
    main()