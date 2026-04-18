import pandas as pd
import numpy as np
import os
import sys

def preprocess_data():
    # 获取当前脚本所在的绝对路径，防止找不到文件
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_path, 'ml-25m')
    save_dir = os.path.join(base_path, 'processed')
    
    print(f"🔍 当前工作目录: {os.getcwd()}")
    print(f"📂 尝试查找数据集路径: {data_dir}")

    # 检查数据集文件夹是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 错误：找不到 '{data_dir}' 文件夹！")
        print("请确保你解压后的 MovieLens 文件夹名为 'ml-25m' 且与本脚本在同一目录下。")
        return

    try:
        # --- 1. 加载 Movies 数据 ---
        print("⏳ 正在读取 movies.csv...")
        movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
        
        # 简单处理：提取年份
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce').fillna(0).astype(np.int32)
        
        # --- 2. 加载 Ratings 数据 ---
        print("⏳ 正在读取 ratings.csv (25M条数据，请耐心等待)...")
        dtypes = {
            'userId': np.int32, 'movieId': np.int32, 
            'rating': np.float32, 'timestamp': np.int64
        }
        # 加载数据
        ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), dtype=dtypes)
        
        print(f"✅ 成功加载！总行数: {len(ratings)}")

        # --- 3. 创建文件夹并保存 ---
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"📁 已创建文件夹: {save_dir}")

        print("💾 正在转换为 Parquet 格式（这能让下次加载快 10 倍）...")
        movies.to_parquet(os.path.join(save_dir, 'movies.parquet'), index=False)
        ratings.to_parquet(os.path.join(save_dir, 'ratings.parquet'), index=False)
        
        print("✨ 预处理全部完成！你现在可以查看 'processed' 文件夹了。")

    except Exception as e:
        print(f"💥 运行中出错: {e}")

if __name__ == "__main__":
    preprocess_data()
    # 强制刷新缓冲区
    sys.stdout.flush()