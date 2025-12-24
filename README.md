# Manual-Assisted-Annotation-by-Streamlit
本仓库构建了人工提示性轻量级模型（all-MiniLM-L6-v2）微调方案，使用streamlit对无监督数据设计了更方便友好的人工标注页面，完成训练集构造后再采用Sentencetransformer的经典训练框架进行嵌入微调，输出base模型和微调模型结果对比;
项目数据已经过脱敏处理（去除原数据中的偶数id数据）；
项目目的旨在通过语义嵌入模型，识别用户产品需求，通过人工构造标签，让模型学会使用者所期望的嵌入模式（尽管由于复杂度较低，本项目还不能完全实现这个目标）

This repository builds a fine-tuning pipeline for a lightweight prompt-based semantic model (all-MiniLM-L6-v2). It uses Streamlit to design a more convenient and user-friendly manual annotation interface for unlabeled data. After constructing the training dataset, the project applies the classic SentenceTransformers training framework to fine-tune the embeddings and outputs a comparison between the base model and the fine-tuned model.

All project data have been anonymized (by removing records with even-numbered IDs from the original data).

The goal of the project is to leverage a semantic embedding model to identify user product needs and, through manually constructed labels, teach the model to learn the embedding patterns expected by users. However, due to the relatively low complexity of the setup, the project does not yet fully achieve this objective.
