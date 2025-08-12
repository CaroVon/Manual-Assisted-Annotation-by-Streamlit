import sys
import os

project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sentence_transformers import SentenceTransformer
import torch
from data.input.need_labels import need_labels

def save_label_embeddings(need_labels, model_name, save_path):
    embedding_model = SentenceTransformer(model_name)
    need_embeddings = {}
    for need, descs in need_labels.items():
        embeddings = embedding_model.encode(descs, convert_to_tensor=True, normalize_embeddings=True)
        mean_embedding = embeddings.mean(dim=0)
        need_embeddings[need] = mean_embedding.cpu()
    # 保存为torch文件（字典形式）
    torch.save(need_embeddings, save_path)
    print(f"✅ 标签嵌入保存完成，路径：{save_path}")

if __name__ == "__main__":
    save_label_embeddings(
        need_labels=need_labels,
        model_name="all-MiniLM-L6-v2",
        save_path="data/need_label_embeddings/without_label.pt"
    )
