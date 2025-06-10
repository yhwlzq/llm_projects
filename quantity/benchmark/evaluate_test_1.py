import numpy as np
import pandas as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score
import datasets
import torch
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba
import matplotlib.pyplot as plt
from datasets import load_dataset


# 1. 数据准备（以LCSTS中文摘要数据集为例）
def load_data(sample_size=500):
    """加载数据集并预处理"""
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
        dataset = dataset.shuffle().select(range(sample_size))

        # 中文分词处理
        def tokenize_chinese(text):
            return " ".join(jieba.cut(text))

        references = [[tokenize_chinese(ref)] for ref in dataset["abstract"]]
        candidates = [tokenize_chinese(cand) for cand in dataset["text"]]

        return references, candidates
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # 备用数据
        references = [[["这 是 一个 样例 参考 摘要"]], [["这是 第二个 示例"]]]
        candidates = ["这 是 生成 的 摘要", "这是 生成 结果"]
        return references, candidates
a,b = load_data(500)
print(a,b)