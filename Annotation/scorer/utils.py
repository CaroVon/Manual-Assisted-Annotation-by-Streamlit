import pandas as pd
import spacy
from tqdm import tqdm
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_lg")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: 加载产品词典
# ------------------------
def load_product_dictionary(dict_path: str):
    product_df = pd.read_csv(dict_path)
    # alias_to_product: alias(英文名) -> (中文产品名, alias)
    alias_to_product = {}
    for _, row in product_df.iterrows():
        product_name = row["产品名"]
        aliases = str(row["英文"]).split(",")
        for alias in aliases:
            alias = alias.strip().lower()
            if alias:
                alias_to_product[alias] = (product_name, alias)

    return alias_to_product

# Step 2: 查找产品与上下文
# ------------------------
kw_model = KeyBERT(model='all-MiniLM-L6-v2')  

def find_products_with_context(text, alias_to_product):
    doc = nlp(text)
    matches = []

    for sent in doc.sents:
        sent_text = sent.text.lower()
        sent_start = sent.start_char
        sent_end = sent.end_char

        # 在句子中查找所有alias
        for alias, (product_name, _) in alias_to_product.items():
            if alias in sent_text:
                keywords = kw_model.extract_keywords(
                    sent.text, 
                    keyphrase_ngram_range=(1, 4),
                    stop_words='english', # 可自定义
                    top_n=5
                )
                matches.append((
                    product_name,  # 产品名
                    alias,         # 匹配别名
                    sent_start,    # 句子起始字符索引
                    sent_end,      # 句子结束字符索引
                    sent.text,     # 原始句子文本
                    [k[0] for k in keywords]  # 关键词列表
                ))
    return matches

# Step 3: 构建产品组合
# ------------------------
def remove_overlapping_matches(matches):
    matches.sort(key=lambda x: x[2])  # 按起始字符索引排序
    grouped = []
    current_group = []

    for match in matches:
        if not current_group:
            current_group.append(match)
            continue
        last = current_group[-1]
        if match[2] <= last[3]:  # 当前句子起始 <= 上一句子结束，说明重叠
            current_group.append(match)
        else:
            grouped.append(current_group)
            current_group = [match]

    if current_group:
        grouped.append(current_group)

    final = []
    for group in grouped:
        if len(group) == 1:
            final.append(group[0])
        else:
            products = sorted(set([m[0] for m in group]))
            aliases = sorted(set([m[1] for m in group]))
            # 取第一个句子的起止和文本
            start_pos = group[0][2]
            end_pos = group[-1][3]
            context_text = " ".join([m[4] for m in group])  # 合并所有句子文本作为组合语义群上下文
            keywords_combined = []
            for m in group:
                keywords_combined.extend(m[5])
            keywords_combined = list(set(keywords_combined))  # 去重关键词

            final.append((
                "+".join(products),
                ",".join(aliases),
                start_pos,
                end_pos,
                context_text,
                keywords_combined
            ))

    return final

# Step 4: 抽取语义群结构
# ------------------------
def extract_product_semantic_groups(text, alias_to_product):
    raw_matches = find_products_with_context(text, alias_to_product)
    filtered_matches = remove_overlapping_matches(raw_matches)

    product_contexts = {}

    for prod_name, alias, _, _, context, keywords in filtered_matches:
        products = prod_name.split("+")
        aliases = alias.split(",") if isinstance(alias, str) else [alias]

        if len(products) == 1:
            prod = products[0]
            if prod not in product_contexts:
                product_contexts[prod] = {"aliases": set(), "contexts": [], "semantic_groups": []}
            product_contexts[prod]["aliases"].update(aliases)
            if context not in product_contexts[prod]["contexts"]:
                product_contexts[prod]["contexts"].append(context)
            if keywords not in product_contexts[prod]["semantic_groups"]:
                product_contexts[prod]["semantic_groups"].append(keywords)
        else:
            combo_key = "+".join(products)
            if combo_key not in product_contexts:
                product_contexts[combo_key] = {"aliases": set(), "contexts": [], "semantic_groups": []}
            product_contexts[combo_key]["aliases"].update(aliases)
            if context not in product_contexts[combo_key]["contexts"]:
                product_contexts[combo_key]["contexts"].append(context)
            if keywords not in product_contexts[combo_key]["semantic_groups"]:
                product_contexts[combo_key]["semantic_groups"].append(keywords)

    # 转为 list
    for prod in product_contexts:
        product_contexts[prod]["aliases"] = list(product_contexts[prod]["aliases"])
    return product_contexts

# Step 5: 主函数，传入文本和词典路径
# ------------------------
def match_texts_to_products(text_df: pd.DataFrame, text_column: str, dict_path: str):
    print("🔄 加载产品词典...")
    alias_to_product = load_product_dictionary(dict_path)

    tqdm.pandas(desc="🔍 分析中")
    text_df["产品语义结构"] = text_df[text_column].progress_apply(
        lambda x: extract_product_semantic_groups(x, alias_to_product)
    )

    output = []
    for _, row in text_df.iterrows():
        url = row["url"] 
        prod_dict = row["产品语义结构"]
        for prod_name, data in prod_dict.items():
            output.append([url, prod_name, data["aliases"], data["contexts"]])
    return output

# Step 6: 计算语义相似度
# ------------------------
def assign_needs_by_semantic_similarity(result_df, label_embeddings, need_names, threshold=0.05):
    matched_records = []

    for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc="🔍 分析中"):
        semantic_group = row["语义群"]
        if isinstance(semantic_group, list):
            sem_text = " ".join(semantic_group)
        else:
            sem_text = str(semantic_group)

        sem_embedding = embedding_model.encode(sem_text, convert_to_tensor=True)

        cos_scores = util.cos_sim(sem_embedding, label_embeddings)[0]

        matched_needs = {
            need_names[i]: round(score.item(), 2)
            for i, score in enumerate(cos_scores)
            # if score >= threshold
        }

        matched_records.append({
            "url": row["url"],
            "产品名": row["产品名"],
            "英文": row["英文"],
            "语义群": semantic_group,
            "匹配需求": matched_needs
        })

    return pd.DataFrame(matched_records)

# Step 7: 重构训练输入
# ------------------------
def restructure_training_input(need_df: pd.DataFrame):
    rows = []

    for _, row in need_df.iterrows():
        product = row["产品名"]
        semantic_group = ", ".join(row["语义群"]) if isinstance(row["语义群"], list) else str(row["语义群"])
        label_scores = row["匹配需求"]  # 是一个 dict: {label: score}

        for label, score in label_scores.items():
            rows.append({
                "product": product,
                "semantic_group": semantic_group,
                "label": label,
                "score": round(score, 4),
                "manual_score": None
            })

    return pd.DataFrame(rows)


   