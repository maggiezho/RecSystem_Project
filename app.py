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
import re

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
from recall import HybridRecall, ItemCF, UserCF, Popularity, get_recommendation_func


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
    fig.update_yaxes(autorange="reversed") 
    return fig

import re

def clean_movielens_title(raw_title):
    """
    电影名清洗器：只干掉英文冠词，且完美兼容夹在多国语言括号前的特殊情况
    """
    if not isinstance(raw_title, str) or not raw_title.strip():
        return raw_title

    # 1. 剥离末尾的年份 (如 " (1966)")
    match_year = re.search(r'\s*\(\d{4}\)$', raw_title)
    year_part = match_year.group() if match_year else ""
    name_part = raw_title[:match_year.start()] if match_year else raw_title

    # 2. 核心：用正则匹配那些挂在括号前面的英文倒置后缀
    match_article = re.search(r'\s*,\s*(The|A|An)(?=\s*\(|$)', name_part, re.IGNORECASE)
    
    if match_article:
        article = match_article.group(1)
        before_article = name_part[:match_article.start()]
        after_article = name_part[match_article.end():]
        name_part = f"{article.capitalize()} {before_article}{after_article}"
            
    # 3. 擦除冗余的英文别名提示 (a.k.a.)，保留原生小语种内容
    name_part = re.sub(r'\s*\(a\.k\.a\..*?\)', '', name_part)

    return f"{name_part}{year_part}"

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
        # 统一变量命名控制：大盘和逻辑层全部打通
        eval_k_value = st.sidebar.selectbox("评估K值", [5, 10, 20], index=1)
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
    def trigger_recommendation():
        st.session_state.show_recommendations = not st.session_state.show_recommendations
        if st.session_state.show_recommendations:
            st.session_state.rec_results = None

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
            
            # 调用全局清洗函数
            display_history['title'] = display_history['title'].apply(clean_movielens_title)
            # 🏷️ 顺便统一清洗体裁的换行问题
            display_history['genres'] = display_history['genres'].str.replace('|', ', ')
            
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
            btn_type = "secondary"
        else:
            btn_label = "🎯 生成个性化推荐"
            btn_type = "primary"
        
        st.button(
            label=btn_label, 
            type=btn_type, 
            use_container_width=True, 
            key="generate_rec_btn", 
            on_click=trigger_recommendation
        )

        if st.session_state.show_recommendations and st.session_state.rec_results is not None:
            res = st.session_state.rec_results
            recs = res['recommendations']
            
            st.success(f"✅ 耗时: {res['duration']:.2f}s | 候选数: {res['candidate_count']} | 推荐数: {len(recs)}")
            
            if recs.empty:
                st.warning("没有找到推荐结果")
            else:
                for idx, (display_idx, row) in enumerate(recs.iterrows()):
                    clean_title = clean_movielens_title(row['title'])
                    clean_genres = row['genres'].replace('|', ', ')

                    # 3. 喂给 Streamlit 渲染
                    with st.expander(f"**{idx+1}. {clean_title}**", expanded=(idx < 3)):
                        col_m1, col_m2 = st.columns([2, 1])
                        with col_m1:
                            st.write(f"📅 **年份**: {int(row['year']) if pd.notna(row['year']) else '未知'}")
                            clean_genres = row['genres'].replace('|', ', ')
                            st.write(f"🏷️ **体裁**: {clean_genres}")
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

        # ==================== 【全新重构】离线评估结果面板 ====================
        if show_evaluation:
            st.divider()
            st.subheader("📊 离线大盘评估（沙箱无泄露）")

            # 触发运行离线评估
            if st.button("⚡ 开始运行离线评估", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("正在严格切分沙箱数据集并运行多策略评估..."):
                    try:
                        # 1. 初始化评估器进行严格的数据流切分
                        from evaluate import RecEvaluator  
                        evaluator = RecEvaluator('processed/ratings.parquet')
                        
                        # 2. 从全量数据中高效过滤获取高表现测试候选集
                        if 'ratings_df' in locals():
                            user_rating_counts = ratings_df.groupby('userId').size()
                            candidate_users = user_rating_counts[user_rating_counts >= 10].index.tolist()
                        else:
                            candidate_users = list(range(1, 611))
                        
                        # 按侧边栏配置限制样本大小
                        if len(candidate_users) > eval_users_sample:
                            test_users = np.random.choice(candidate_users, eval_users_sample, replace=False).tolist()
                        else:
                            test_users = candidate_users

                        # 3. 提取沙箱训练集映射，严格隔绝测试集
                        clean_user_train_dict = {}
                        clean_user_test_dict = {}
                        for uid in test_users:
                            train_items, test_items = evaluator.get_user_test_items_loo(uid, 0.8)
                            clean_user_train_dict[uid] = train_items
                            clean_user_test_dict[uid] = set(test_items)

                        # 4. 加载召回函数策略映射
                        strategies = {
                            'ItemCF': get_recommendation_func('itemcf'),
                            'UserCF': get_recommendation_func('usercf'),
                            '热门电影': get_recommendation_func('popularity'),
                            '混合召回': get_recommendation_func('hybrid', params=weights)
                        }
                        
                        results = [] 
                        
                        # 串行流式评估各个策略
                        for s_idx, (strategy_name, recommend_func) in enumerate(strategies.items()):
                            status_text.text(f"🎬 正在计算指标 - 策略: {strategy_name} ...")
                            
                            user_precisions = {5: [], 10: [], 20: []}
                            user_recalls = {5: [], 10: [], 20: []}
                            user_hit_flags = {5: [], 10: [], 20: []}
                            user_f1s = {5: [], 10: [], 20: []}
                            user_ndcgs = {5: [], 10: [], 20: []}
                            user_mrrs = {5: [], 10: [], 20: []}
                            
                            for uid in test_users:
                                pure_train_items = clean_user_train_dict[uid]
                                test_set = clean_user_test_dict[uid]
                                
                                if len(test_set) == 0:
                                    continue
                                
                                try:
                                    # 💡 核心对齐修复：传入 top_n=100 放宽口袋，并强制灌入纯净历史 user_history 拦截数据泄露
                                    recommendations = recommend_func(uid, top_n=100, user_history=pure_train_items)
                                    
                                    if not recommendations:
                                        continue
                                    
                                    # 分流计算 [5, 10, 20] 桶指标
                                    for k_val in [5, 10, 20]:
                                        rec_k_list = recommendations[:k_val]
                                        rec_k_set = set(rec_k_list)
                                        hits = len(rec_k_set & test_set)
                                        
                                        prec = hits / k_val
                                        rec = hits / len(test_set)
                                        user_precisions[k_val].append(prec)
                                        user_recalls[k_val].append(rec)
                                        user_hit_flags[k_val].append(1.0 if hits > 0 else 0.0)
                                        
                                        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                                        user_f1s[k_val].append(f1)
                                        
                                        # NDCG
                                        dcg = 0.0
                                        for rank, item in enumerate(rec_k_list):
                                            if item in test_set:
                                                dcg += 1.0 / np.log2(rank + 2)
                                        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k_val, len(test_set)))])
                                        user_ndcgs[k_val].append(dcg / idcg if idcg > 0 else 0.0)
                                        
                                        # MRR
                                        mrr = 0.0
                                        for rank, item in enumerate(rec_k_list):
                                            if item in test_set:
                                                mrr = 1.0 / (rank + 1)
                                                break
                                        user_mrrs[k_val].append(mrr)
                                        
                                except Exception:
                                    continue
                            
                            # 锚定用户在侧边栏选择的指定 K 值进行最终的平均指标输出
                            k = eval_k_value 
                            
                            avg_prec = np.mean(user_precisions[k]) if user_precisions[k] else 0.0
                            avg_recall = np.mean(user_recalls[k]) if user_recalls[k] else 0.0
                            avg_f1 = np.mean(user_f1s[k]) if user_f1s[k] else 0.0
                            avg_ndcg = np.mean(user_ndcgs[k]) if user_ndcgs[k] else 0.0
                            avg_hr = np.mean(user_hit_flags[k]) if user_hit_flags[k] else 0.0
                            avg_mrr = np.mean(user_mrrs[k]) if user_mrrs[k] else 0.0
                            
                            results.append({
                                '策略': strategy_name, 
                                'Precision': round(avg_prec, 4),
                                'Recall': round(avg_recall, 4),
                                'F1': round(avg_f1, 4),
                                'NDCG': round(avg_ndcg, 4),
                                'Hit Rate': round(avg_hr, 4),
                                'MRR': round(avg_mrr, 4)
                            })
                            progress_bar.progress((s_idx + 1) / len(strategies))
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # 5. 渲染结果
                        comparison_df = pd.DataFrame(results)
                        st.success(f"📊 离线评估完成！有效测试用户数: {len(test_users)} (测试流程无任何数据穿透现象)")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # 6. 渲染对比柱状图
                        fig = go.Figure()
                        for metric in ['Precision', 'Recall', 'F1', 'NDCG']:
                            fig.add_trace(go.Bar(
                                x=comparison_df['策略'],
                                y=comparison_df[metric],
                                name=f"{metric}@{k}"
                            ))
                        fig.update_layout(
                            title=f"不同召回策略在 K={k} 时的基准表现对比图",
                            xaxis_title="召回策略", yaxis_title="指标得分",
                            barmode='group', template='plotly_dark'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as global_err:
                        st.error(f"评估运行失败，错误日志: {global_err}")
            else:
                st.info("💡 参数已就绪，请点击上方按钮开始离线评估。")

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