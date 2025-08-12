import sys
import os

project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sentence_transformers import SentenceTransformer
import torch
from data_process.utils import get_combined_label_embeddings
from data.input.need_labels import need_labels


if __name__ == "__main__":
    model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)

    need_embeddings = get_combined_label_embeddings(need_labels, embedding_model, alpha=0.5)

    save_path = "data/need_label_embeddings/with_label.pt"
    torch.save(need_embeddings, save_path)
    print(f"✅ 标签嵌入已保存，路径：{save_path}")
