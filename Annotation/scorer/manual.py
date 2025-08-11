import streamlit as st
import numpy as np
import pandas as pd

def interactive_manual_scoring_gui(df_training, save_path="Annotation/data/output/manual_scores.csv"):
    st.set_page_config(layout="wide")
    st.title("📝 Manual Annotation")

    # 1. 复制数据并保留原始索引为列，避免索引错乱
    df = df_training.copy().reset_index(drop=False)  # 原索引保存在列 "index"

    if "manual_score" not in df.columns:
        df["manual_score"] = None

    # 2. 初始化 session 状态
    if "scores" not in st.session_state:
        st.session_state.scores = {}
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "remaining_indices" not in st.session_state:
        st.session_state.remaining_indices = list(range(len(df)))  # 使用行号

    all_scores = df["score"].astype(float).values
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)

    total = len(df)
    completed = len(st.session_state.scores)
    st.markdown(f"📊 当前进度：`{completed}` / `{total}` 已完成")
    st.progress(completed / total)

    # 3. 全部完成时展示结果和保存按钮
    if st.session_state.current_idx >= len(st.session_state.remaining_indices):
        st.success("🎉 所有样本已完成打分！")
        # 将打分写回 df
        for idx_, val in st.session_state.scores.items():
            df.loc[df["index"] == int(idx_), "manual_score"] = val
        st.dataframe(df.drop(columns=["index"]))
        if st.button("💾 保存到CSV"):
            df.drop(columns=["index"]).to_csv(save_path, index=False)
            st.success(f"✅ 文件已保存到 {save_path}")
        return

    # 4. 当前样本读取
    row_idx = st.session_state.remaining_indices[st.session_state.current_idx]
    current = df.iloc[row_idx]
    real_idx = current["index"]  # 原始索引

    st.markdown("---")
    st.markdown(f"### 🛍 产品：`{current['product']}`")
    with st.expander("📚 展开语义群", expanded=False):
        st.markdown(current["semantic_group"])
    st.markdown(f"🏷 标签：`{current['label']}`")
    st.markdown(f"模型得分：`{current['score']}`")

    z = (current["score"] - mean_score) / std_score
    st.markdown(f"📈 Z-Score: `{z:.2f}`，当前分数位于均值 {'上方' if z > 0 else '下方'}")
    label_df = df[df["label"] == current["label"]]
    label_scores = label_df["score"].astype(float)
    label_mean = label_scores.mean()
    label_std = label_scores.std()

    st.markdown(f"📌 当前标签（{current['label']}）样本分数均值：`{label_mean:.4f}`，标准差：`{label_std:.4f}`")

    # 5. 使用 form 保证提交原子性
    with st.form(key=f"form_{real_idx}"):
        choice = st.radio("你的判断：", ["-1 模型低估", "0 合适", "1 模型高估", "跳过"], horizontal=True)
        submitted = st.form_submit_button("✅ 提交打分")

    if submitted:
        if choice != "跳过":
            st.session_state.scores[real_idx] = int(choice.split()[0])
        st.session_state.current_idx += 1
        st.rerun()

    # 6. 上一条 和 保存按钮
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("⬅️ 上一条") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()

    with col2:
        if st.button("💾 保存当前结果"):
            for idx_, val in st.session_state.scores.items():
                df.loc[df["index"] == int(idx_), "manual_score"] = val
            df.drop(columns=["index"]).to_csv(save_path, index=False)
            st.success(f"✅ 当前结果已保存到 {save_path}")
