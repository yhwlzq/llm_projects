import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import os
import time
from tqdm import tqdm

# Configuration
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
rouge = evaluate.load('rouge')


def get_model(model_id):
    """Load model with error handling and device management"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_id}: {str(e)}")
        raise


def create_summaries(texts_list, tokenizer, model, max_new_tokens=256, model_name=""):
    """Improved summarization with better prompts and progress tracking"""
    system_prompt = (
        "You are an expert AI assistant trained to summarize news articles. "
        "Generate concise summaries that capture key information in 3-5 sentences."
    )
    user_prompt_template = "ARTICLE: {text}\nSUMMARY:"

    summaries_list = []
    gen_times = []

    for text in tqdm(texts_list, desc=f"Generating summaries ({model_name})"):
        # Format input based on model type
        if "qwen" in model_name.lower():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(text=text)}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            input_text = f"{system_prompt}\n\n{user_prompt_template.format(text=text)}"

        start_time = time.time()
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        gen_time = time.time() - start_time
        gen_times.append(gen_time)

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean output by removing the input prompt
        summary = summary.replace(input_text, "").strip()
        summaries_list.append(summary)

    avg_time = np.mean(gen_times)
    print(f"{model_name} average generation time: {avg_time:.2f}s per sample")
    return summaries_list


def compute_metrics(generated, references):
    """Enhanced evaluation with multiple metrics"""
    # Basic ROUGE scores
    rouge_scores = rouge.compute(
        predictions=generated,
        references=references,
        use_stemmer=True
    )

    # Additional metrics could be added here
    return rouge_scores


def load_data(filename, sample_size=None):
    """Data loading with optional sampling"""
    datas = pd.read_csv(filename)
    if sample_size and len(datas) > sample_size:
        datas = datas.sample(sample_size, random_state=42)
    return datas["Article Body"].tolist()


def main():
    # Model paths
    baseline_model = "/mnt/c/Users/zhouq/AI学习/Qwen2.5-0.5B-Instruct"
    optimized_model = "/home/mystic/PycharmProjects/finetuning/ragbi/quantity/qwen3-8b-qptq"

    # Load data (sample for quick testing)
    data = load_data("/home/mystic/PycharmProjects/llm/llm_projects/llm_evaluation/articles.csv", sample_size=10)  # Adjust sample_size as needed

    # Load models
    print("Loading baseline model...")
    b_token, b_model = get_model(baseline_model)
    print("Loading optimized model...")
    a_token, a_model = get_model(optimized_model)

    # Generate summaries
    print("\nGenerating baseline summaries...")
    b_results = create_summaries(data, b_token, b_model, model_name="Baseline")
    print("\nGenerating optimized summaries...")
    a_results = create_summaries(data, a_token, a_model, model_name="Optimized")

    # Evaluation
    print("\nEvaluating results...")
    b_metrics = compute_metrics(b_results, b_results)  # Self-comparison as baseline
    a_metrics = compute_metrics(a_results, b_results)  # Compare optimized vs baseline

    print("\n=== Results ===")
    print(f"Baseline ROUGE Scores (self-comparison): {b_metrics}")
    print(f"Optimized vs Baseline ROUGE Scores: {a_metrics}")

    # Additional comparison: Human evaluation samples
    print("\nSample Comparison:")
    for i in range(min(3, len(data))):  # Show first 3 comparisons
        print(f"\nArticle {i + 1}:")
        print(f"Baseline Summary: {b_results[i]}")
        print(f"Optimized Summary: {a_results[i]}")
        print("-----")


if __name__ == "__main__":
    main()