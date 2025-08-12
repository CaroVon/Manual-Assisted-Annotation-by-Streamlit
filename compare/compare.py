import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_text(text):
    if isinstance(text, list):
        return " ".join(text)
    elif pd.isna(text) or text == "":
        return ""
    else:
        return str(text)

def update_scores_with_new_model(df, embedding_model, label_embeddings, need_names,
                                 text_col="semantic_group", label_col="label"):
    df = df.copy()

    texts = df[text_col].apply(preprocess_text).tolist()
    sem_embeddings = embedding_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to(device)

    label2idx = {name: idx for idx, name in enumerate(need_names)}

    new_scores = []
    for i, label in enumerate(df[label_col]):
        idx = label2idx.get(label)
        if idx is None:
            new_scores.append(None)
        else:
            score = util.cos_sim(sem_embeddings[i], label_embeddings[idx]).item()
            new_scores.append(round(score, 4))

    df["old_score"] = df.get("score", None)
    df["score"] = new_scores
    return df


if __name__ == "__main__":
    input_path = "train/data/manual_scores.csv"
    need_emb_path = "data/need_label_embeddings/without_label.pt"

    # 读取输入数据
    df = pd.read_csv(input_path)

    # 读取标签嵌入字典，key: 标签名，value: tensor
    need_embeddings = torch.load(need_emb_path, map_location=device)
    need_names = list(need_embeddings.keys())
    label_embeddings = torch.stack([need_embeddings[name] for name in need_names]).to(device)

    # 加载新语义向量模型
    embedding_model_new = SentenceTransformer("models/model_without_label", device=str(device))

    # 运行更新函数
    df_updated = update_scores_with_new_model(
        df,
        embedding_model=embedding_model_new,
        label_embeddings=label_embeddings,
        need_names=need_names,
        text_col="semantic_group",
        label_col="label"
    )

    print(df_updated.head())

    df_updated.to_csv("data/output/compare_without_label.csv", index=False)
