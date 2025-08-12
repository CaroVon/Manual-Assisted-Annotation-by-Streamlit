import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import random
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# 分句和平均嵌入
def encode_text_with_mean_pooling(text, model):
    sentences = sent_tokenize(text.strip())
    embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings.mean(dim=0)

# 加载标签
def load_label_embeddings(path):
    """
    从文件加载标签嵌入字典，key是标签名，value是tensor向量
    """
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
        examples.append(InputExample(texts=[semantic_group, label], label=float(manual_score)))
    return examples

# 3. 自定义 collate 函数
def collate_fn(batch):
    return batch

# 4. 句对编码辅助函数
from nltk.tokenize import sent_tokenize
import torch

def tokenize_sentence_label_pairs(model, semantic_groups, labels, label_embeddings, device):
    tokenizer = model.tokenizer
    encoder = model._first_module()
    embeddings1 = []

    for text in semantic_groups:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        with torch.set_grad_enabled(True):
            output = encoder(inputs)['token_embeddings']
            emb = output.mean(dim=1).squeeze(0)
        embeddings1.append(emb)

    batch_emb1 = torch.stack(embeddings1).to(device)

    # 从预加载的label向量中获取label嵌入
    embeddings2 = [label_embeddings[label].to(device) for label in labels]
    batch_emb2 = torch.stack(embeddings2)

    return {'embedding1': batch_emb1, 'embedding2': batch_emb2}


# 5. 自定义Loss
class CustomContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cos = nn.CosineSimilarity(dim=1)
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features, labels):
    # 从句向量字典直接取embeddings
        embed1 = sentence_features['embedding1']
        embed2 = sentence_features['embedding2']

        cosine_sim = self.cos(embed1, embed2)

        target = torch.zeros_like(labels)
        target[labels == 1] = 0.3
        target[labels == 0] = 0.5
        target[labels == -1] = 0.7

        loss = self.loss_fct(cosine_sim, target)
        return loss
import random

def corrupt_data(df, corruption_ratio=0.05, seed=42):
    random.seed(seed)
    corrupt_indices = df.sample(frac=corruption_ratio, random_state=seed).index
    df.loc[corrupt_indices, 'manual_score'] = None  # 模拟错误标注为缺失
    return df

# 6. 训练函数
def train(data_path, model_save_path, label_emb_path, epochs=3, batch_size=16):
    df = pd.read_csv(data_path)
    df = corrupt_data(df, corruption_ratio=0.05)
    dataset = ProductLabelDataset(df)
    examples = convert_to_input_examples(dataset)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    label_embeddings = load_label_embeddings(label_emb_path)

    train_dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    train_loss = CustomContrastiveLoss(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            sentences1 = [ex.texts[0] for ex in batch]
            sentences2 = [ex.texts[1] for ex in batch]
            labels = torch.tensor([ex.label for ex in batch], dtype=torch.float).to(device)

            optimizer.zero_grad()

            features = tokenize_sentence_label_pairs(model, sentences1, sentences2, label_embeddings, device)

            loss = train_loss(features, labels)

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
        model_save_path="models/model_without_label",
        label_emb_path="data/need_label_embeddings/without_label.pt",
        epochs=3,
        batch_size=8
    )

