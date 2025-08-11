import pandas as pd
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from scorer.utils import  match_texts_to_products, assign_needs_by_semantic_similarity, restructure_training_input


nlp = spacy.load("en_core_web_lg")
kw_model = KeyBERT(model='all-MiniLM-L6-v2') 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


df = pd.read_csv("Annotation/data/input/text.csv").fillna("")
df = df.sample(n=10).reset_index(drop=True)
TEXT_COLUMNS = ["text"]
df["full_text"] = df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)


# 词典路径
product_dict_path = "Annotation/data/input/产品词典.csv"
result_df = match_texts_to_products(df, text_column="full_text", dict_path=product_dict_path)

result_df = pd.DataFrame(result_df, columns=["url", "产品名", "英文", "语义群"])
print(result_df.head())


# 定义需求标签及其描述
need_labels = {
    "Immersion": [
    "Feels like stepping into another world.",
    "Sets the vibe instantly—super immersive!",
    "Makes you forget your surroundings, in a good way.",
    "Feels cinematic—like you're part of a story.",
    "Creates that ‘wow’ moment that draws you right in.",
    "You can’t help but stare—it pulls you into the scene.",
    "I lose track of time when I’m here.",
    "This setup helps me get fully in the zone.",
    "I don’t feel distracted anymore",
    "The lighting, sound, and layout just suck me in.",
    "When I step into this room, I know it’s time to focus",
    "I don’t even notice the outside world anymore."   
]
,
    "Entertainment": [
    "This is just plain fun to look at!",
    "This setup just looks so fun and badass.",
    "Low-key obsessed—it’s kinda addictive.",
    "Adds a spark of joy every time you see it.",
    "It’s like a gaming spaceship—I love it."
    "You won’t get bored with this around.",
    "It’s giving ✨entertainment value✨.",
    "Every time I sit here, it’s a whole vibe.",
    "This is where I game for hours.",
    "I’m actually obsessed with how cool this looks.",
    "This is my fun zone.",
    "It’s giving full-on gamer aesthetic.",
    "Kinda makes me feel like I’m in a sci-fi movie.",
    "This whole space just feels alive.",
    "Super sleek and just fun to be around.",
    "There’s something so satisfying about how everything comes together.",
    "The energy in this setup is just insane—it’s hype.",
    "It looks clean but still packs a punch.",
    "I literally don’t wanna leave this setup—it’s that cool.",
    "Not just functional—it’s straight-up fun and fire.",
    "Honestly, it makes gaming feel cinematic.",
    "Everything just pops—it’s like eye candy for gamers."
]
,
    "Order": [
    "Everything just makes sense—clean and tidy.",
    "Satisfyingly organized. My brain is happy.",
    "Nothing’s out of place. Total clarity.",
    "Gives off major ‘I’ve got my life together’ vibes.",
    "No clutter, no chaos—just clean lines and logic.",
    "Streamlined setup = smooth workflow.",
    "You just know this person gets things done.",
    "away from all the chaos",
    "I have a whole routine when I sit down at this desk.",
    "Everything has its place.",
    "feels decluttered",
    "feels minimal",
    "Now I can think clearly",
    "I got tired of chaos",
    "My brain feels calmer in an organized setup."
]
,
    "Sociability": [
    "This is perfect for sharing with friends.",
    "You just *have* to talk about it—it’s a convo starter.",
    "Makes you wanna invite people over.",
    "So good, you’ll want to show it off.",
    "Designed to be enjoyed together.",
    "Screams ‘let’s hang out’ energy.",
    "Instantly makes the space feel more welcoming.",
    "It feels good to show this off."
    "I love having friends over to enjoy this space.",
]
,
    "Aesthetics": [
    "This is just plain beautiful.",
    "Looks like it belongs in a magazine.",
    "Every detail is just gorgeous.",
    "It’s like art you can use every day.",
    "Makes the space feel more stylish.",
    "You can’t help but admire it.",
    "It’s giving ‘aesthetic goals’ vibes.",
    "everything to be color cohesive",
    "have them be even cuter over time ",
    "It just looks so good.",
    "This color combo makes me happy.",
    "It’s visually satisfying.",
    "I wanted a clean / warm / dark / cozy look.",
    "Every angle is pleasing."
]
,
    "Self-Expression": [
    "you can cut it at any length or get super creative with shapes on the wall",
    "This just feels like me / represents who I am.",
    "I’ve always wanted a setup that reflects my personality.",
    "Every detail here means something to me",
    "I just had to include this fandom / IP / color",
    "It’s totally my vibe / energy",
    "It’s a reflection of my taste and interests.",
]
,
    "Safety": [
    "It just feels like a safe space",
    "I just feel calm / at peace here",
    "I don’t feel anxious anymore",
    "I feel safe and secure here",
    "This setup makes me feel protected.",
    "It’s like a cozy hug for my mind.",
    "I can finally relax and unwind here.",
]
,
    "Efficiency": [
    "quickly charge my phone my camera without cluttering the floor.",
    "made everything faster.",
    "Everything’s just within reach now.",
    "This setup saves me so much time.",
    "I don’t have to waste time looking for things anymore.",
    "It’s like a time-saving machine.",
    "I can finally get things done faster.",
    "It just flows better.",
    "I don’t waste energy setting up anymore."
]
,
    "Relaxation": [
    "This is where I wind down after a long day.",
    "I just chill here with some lo-fi music and a cup of coffee.",
    "I can zone out and just enjoy my games.",
    "make the mood more relaxing",
    "I love spending my quiet time here.",
    "This is where I wind down.",
    "I come here to decompress.",
    "It’s like therapy for me.",
    "I just chill with music / tea / candles.",
    "I love ending my day here."
]
,
    "Ritual": [
    "It just feels complete now.",
    "Everything ties together now",
    "I have a ritual every time I sit down",
    "makes it feel real.",
    "marks the start of my day."
    "This is something I designed from scratch.",
    "I customized every part of this to fit my needs.",
    "I built this from scratch.",
    "Every decision here was mine.",
    "I finally feel in charge of my environment.",
    "I don’t rely on anyone else anymore.",
    "It’s my space. Period."
]
,
    "Health": [
    "This is perfect for a healthy lifestyle.",
    "I can keep up with my fitness goals here.",
    "It’s like a workout in itself.",
    "I can focus on my health without, as well",
    "It’s designed to support my well-being.",
    "It’s like a little sanctuary for my body.",
    "I feel more energized and motivated here.",
    "Long sitting is no longer a problem.",
    "Backache is gone.",
    "Uncomfort is a thing of the past."
]   
}

# 聚合后的标签嵌入向量
need_embeddings = {
    need: embedding_model.encode(descs, convert_to_tensor=True).mean(dim=0)
    for need, descs in need_labels.items()
}

label_embeddings = torch.stack(list(need_embeddings.values()))
need_names = list(need_embeddings.keys())

need_df = assign_needs_by_semantic_similarity(result_df, label_embeddings, need_names)

df_training = restructure_training_input(need_df)
print(df_training.head())