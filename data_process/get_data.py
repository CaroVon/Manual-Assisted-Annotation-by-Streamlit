import pandas as pd
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from .utils import  match_texts_to_products, assign_needs_by_semantic_similarity, restructure_training_input


df = pd.read_csv("Annotation/data/input/text.csv")
TEXT_COLUMNS = ["text"]
df["full_text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)


product_dict_path = "Annotation/data/input/产品词典.csv"
result_df = match_texts_to_products(df, text_column="full_text", dict_path=product_dict_path)

result_df = pd.DataFrame(result_df, columns=["url", "产品名", "英文", "语义群"])

need_emb_path = "data_process/get_label_without_label.pt"

need_embeddings = torch.load(need_emb_path)
need_names = list(need_embeddings.keys())
label_embeddings = torch.stack([need_embeddings[name] for name in need_names])


need_df = assign_needs_by_semantic_similarity(result_df, label_embeddings, need_names)
print(need_df.head())
df_training = restructure_training_input(need_df)
print(df_training.head())