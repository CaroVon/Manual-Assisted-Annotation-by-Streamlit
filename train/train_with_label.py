import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from tqdm import tqdm
import random
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')


# 分句和平均嵌入
def encode_text_with_mean_pooling(text, model):
    sentences = sent_tokenize(text.strip())
    embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings.mean(dim=0)

# 加载标签
def load_label_embeddings(path):
    label_embeds = torch.load(path)
    return label_embeds


# 1. 自定义数据集
class ProductLabelDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.dropna(subset=['manual_score']).reset_index(drop=True)
        self.label_map = {-1: -1, 0: 0, 1: 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        semantic_group = row['semantic_group']
        label = row['label']
        manual_score = self.label_map.get(row['manual_score'], 0)
        return semantic_group, label, manual_score

# 2. 转换成 InputExample 格式
def convert_to_input_examples(dataset):
    examples = []
    for semantic_group, label, manual_score in dataset:
        examples.append({
            "semantic_group": semantic_group,
            "label_text": label,
            "label_name": label,
            "score": float(manual_score)
        })
    return examples

def collate_fn(batch):
    return batch  # 保留原始结构

# 4. 句对编码辅助函数
def tokenize_sentence_pairs(model, sentences1, sentences2, device):
    tokenizer = model.tokenizer
    encoder = model._first_module()
    embeddings1, embeddings2 = [], []

    for s1, s2 in zip(sentences1, sentences2):
        inputs1 = tokenizer(s1, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        inputs2 = tokenizer(s2, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)

        with torch.set_grad_enabled(True):
            output1 = encoder(inputs1)['token_embeddings']
            output2 = encoder(inputs2)['token_embeddings']
            emb1 = output1.mean(dim=1).squeeze(0)
            emb2 = output2.mean(dim=1).squeeze(0)

        embeddings1.append(emb1)
        embeddings2.append(emb2)

    return torch.stack(embeddings1), torch.stack(embeddings2)


# 5. 自定义Loss
class CombinedContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.5):  # alpha 控制两路径权重
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fct = nn.MSELoss()
        self.alpha = alpha

    def forward(self, emb_text_a, emb_text_b, emb_label_b, labels):
        target = torch.zeros_like(labels)
        target[labels == 1] = 0.3
        target[labels == 0] = 0.5
        target[labels == -1] = 0.7

        sim_text = self.cos(emb_text_a, emb_text_b)
        sim_embed = self.cos(emb_text_a, emb_label_b)

        loss_text = self.loss_fct(sim_text, target)
        loss_embed = self.loss_fct(sim_embed, target)

        return self.alpha * loss_text + (1 - self.alpha) * loss_embed
    

def corrupt_data(df, corruption_ratio=0.05, seed=42):
    random.seed(seed)
    corrupt_indices = df.sample(frac=corruption_ratio, random_state=seed).index
    df.loc[corrupt_indices, 'manual_score'] = None  
    return df


# === 5. 主训练函数 ===
def train(data_path, model_save_path, label_emb_path, epochs=3, batch_size=16, alpha=0.5):
    df = pd.read_csv(data_path)
    df = corrupt_data(df, corruption_ratio=0.05)
    dataset = ProductLabelDataset(df)
    examples = convert_to_input_examples(dataset)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    label_embeddings = load_label_embeddings(label_emb_path)

    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

    loss_fn = CombinedContrastiveLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            sentences1 = [item["semantic_group"] for item in batch]
            sentences2 = [item["label_text"] for item in batch]
            label_keys = [item["label_name"] for item in batch]
            scores = torch.tensor([item["score"] for item in batch], dtype=torch.float).to(device)

            optimizer.zero_grad()

            emb_text_a, emb_text_b = tokenize_sentence_pairs(model, sentences1, sentences2, device)
            emb_label_b = torch.stack([label_embeddings[label].to(device) for label in label_keys])

            loss = loss_fn(emb_text_a, emb_text_b, emb_label_b, scores)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

    model.save(model_save_path)
    print(f"✅ 模型已保存至: {model_save_path}")


# 7. 主入口
if __name__ == "__main__":
    train(
        data_path="train/data/manual_scores.csv", 
        model_save_path="models/model_with_label",
        label_emb_path="data/need_label_embeddings/with_label.pt",
        epochs=3,
        batch_size=8,
        alpha=0  # label权重
    )


