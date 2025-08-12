import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from data_process.utils import match_texts_to_products, assign_needs_by_semantic_similarity, restructure_training_input

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nlp = spacy.load("en_core_web_lg")
kw_model = KeyBERT(model='all-MiniLM-L6-v2') 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

df = pd.read_csv("data/input/text.csv").fillna("")
df = df.sample(n=10).reset_index(drop=True)
TEXT_COLUMNS = ["text"]
df["full_text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)

# 词典路径
product_dict_path = "data/input/product_dict.csv"
result_df = match_texts_to_products(df, text_column="full_text", dict_path=product_dict_path)

result_df = pd.DataFrame(result_df, columns=["url", "产品名", "英文", "语义群"])
print(result_df.head())

# 加载 need label embeddings，并放到设备
need_emb_path = "data/need_label_embeddings/without_label.pt"
need_embeddings = torch.load(need_emb_path, map_location=device)

need_names = list(need_embeddings.keys())
label_embeddings = torch.stack([need_embeddings[name] for name in need_names]).to(device)

# 相似度计算
need_df = assign_needs_by_semantic_similarity(result_df, label_embeddings, need_names, device=device)

# 重构训练数据
df_training = restructure_training_input(need_df)
print(df_training.head())
