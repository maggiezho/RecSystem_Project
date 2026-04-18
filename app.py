import streamlit as st
import pandas as pd
import pickle
import time
from recommender import get_final_recommendation

# --- 页面配置 ---
st.set_page_config(page_title="MovieLens 智能推荐系统", layout="wide")

# --- 加载基础数据（用于显示） ---
@st.cache_data # 使用缓存，避免每次操作都重新读取大数据文件
def load_base_data():
    movies = pd.read_parquet('processed/movies.parquet')
    return movies

movies_df = load_base_data()

# --- 侧边栏 ---
st.sidebar.title("🎬 推荐控制台")
st.sidebar.info("这是一个基于 MovieLens 25M 数据集的双层推荐系统（召回+精排）")

user_id = st.sidebar.number_input("请输入用户 ID", min_value=1, value=1, step=1)
top_k = st.sidebar.slider("推荐数量", 5, 20, 10)

# --- 主界面 ---
st.title("🍿 电影推荐系统展示")
st.markdown(f"### 当前模拟用户: **User {user_id}**")

# 展示该用户看过的电影（为了对比效果）
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("观看历史 (部分)")
    ratings = pd.read_parquet('processed/ratings.parquet')
    user_history = ratings[ratings['userId'] == user_id].sort_values('timestamp', ascending=False).head(5)
    history_display = pd.merge(user_history, movies_df, on='movieId')
    st.table(history_display[['title', 'genres', 'rating']])

# --- 推荐触发 ---
if st.sidebar.button("生成个性化推荐"):
    with col2:
        st.subheader("算法实时推荐")
        with st.spinner('🚀 正在运行召回与精排算法...'):
            start_time = time.time()
            
            # 调用你之前的核心函数
            try:
                recommendations = get_final_recommendation(user_id, top_k=top_k)
                
                duration = time.time() - start_time
                st.success(f"计算完成！耗时: {duration:.2f} 秒")
                
                # 漂亮地展示结果
                for i, row in recommendations.iterrows():
                    with st.expander(f"Top {i+1}: {row['title']}"):
                        st.write(f"🏷️ 类型: {row['genres']}")
                        st.write(f"📅 上映年份: {row['year']}")
                        st.progress(0.85) # 这里可以放模型算出的得分（可选）
                        
            except Exception as e:
                st.error(f"推荐失败: {e}")
                st.info("提示：如果是新用户，可能需要先处理冷启动逻辑。")

# --- 数据统计页 (体现工作量) ---
if st.sidebar.checkbox("查看系统监控"):
    st.divider()
    st.subheader("📈 系统运行状态")
    m1, m2, m3 = st.columns(3)
    m1.metric("索引电影数", len(movies_df))
    m2.metric("训练样本量", "20M+")
    m3.metric("当前算法", "LGBM + ItemCF")