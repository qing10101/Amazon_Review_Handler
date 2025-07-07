# ==============================================================================
#      PARALLELIZED & ENHANCED SENTIMENT ANALYSIS SCRIPT
# ==============================================================================
#
# This script performs sentiment analysis using multiple CPU cores for VADER/TextBlob
# and efficient batching for Transformers to maximize performance.
#
# ==============================================================================
import logging
import time
import random
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
from textblob import TextBlob
import sys
import pandas as pd
import numpy as np
from ollama_llm_handler import ask_ollama
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import multiprocessing
import os
from functools import partial

# --- 1. MODEL/ANALYZER INITIALIZATION (Global for Workers) ---
# These are initialized once and inherited by the worker processes
print("INITIALIZING REQUIRED MODELS...")
vader_analyzer = SentimentIntensityAnalyzer()

# NOTE: The transformer pipeline will now be initialized within the worker
# processes for multi-GPU safety. We define its configuration here.
TRANSFORMER_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print("REQUIRED MODELS LOADED...")


# --- 2. HARDWARE DETECTION ---
def detect_hardware():
    """Detects available hardware and returns a descriptive dictionary."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"✅ Found {count} NVIDIA CUDA GPU(s).")
        return {"type": "cuda", "count": count}
    elif torch.backends.mps.is_available():
        print("✅ Found Apple Silicon (MPS) GPU.")
        return {"type": "mps", "count": 1}
    else:
        count = os.cpu_count()
        print(f"⚠️ No supported GPU found. Falling back to {count} CPU cores.")
        return {"type": "cpu", "count": count}


# --- 3. SENTIMENT ANALYSIS FUNCTIONS ---
def parse_first_number_from_llm_response(text: str):
    """
    Parses the first number (integer or float, positive or negative) from a string.

    Generated code
    Args:
        text: The input string to search for a number.

    Returns:
        The first number found as an int or float, or None if no number is found.
    """
    # This regular expression is designed to find the first integer or float.
    # -?         - an optional negative sign
    # \d+        - one or more digits
    # (\.\d+)?   - an optional decimal point followed by one or more digits
    # The combination covers integers (e.g., "123", "-45") and floats (e.g., "19.99", "-0.5").
    match = re.search(r'-?\d+(\.\d+)?', text)

    if match:
        # If a match is found, extract the matched string
        number_str = match.group(0)

        # Convert the string to a number (float or int)
        if '.' in number_str:
            return float(number_str)
        else:
            return int(number_str)

    # If no number is found, return None
    return None


def llm_prompt_constructor(review_text: str):
    prompt = (f"You are asked to perform sentiment analysis for a user review.\nThe review is: {review_text}\n"
              f"Respond with a sentiment score with two digits after decimal that is between -1 (extreme negative) and "
              f"1 (extreme positive)")
    return prompt


def ollama_analyze(review_text: str):
    response = ask_ollama(llm_prompt_constructor(review_text))
    polarity = parse_first_number_from_llm_response(response)
    if polarity is None: return 0.0
    return max(-1.0, min(1.0, polarity))


def textblob_analyze(review_text: str):
    polarity = TextBlob(review_text).sentiment.polarity
    return max(-1.0, min(1.0, polarity))


def vader_analyze(review_text: str):
    polarity = vader_analyzer.polarity_scores(review_text)['compound']
    return max(-1.0, min(1.0, polarity))


def transformer_gpu_worker(texts_chunk, gpu_id, batch_size):
    """A dedicated worker for a single NVIDIA GPU."""
    print(f"[GPU Worker {gpu_id}] Process started on CUDA device {gpu_id}...")
    pipe = pipeline(
        "sentiment-analysis", model=TRANSFORMER_MODEL_NAME, device=f"cuda:{gpu_id}",
        torch_dtype=torch.float16 # Use half-precision for huge speedup on supported GPUs
    )
    print(f"[GPU Worker {gpu_id}] Pipeline loaded. Analyzing {len(texts_chunk)} texts with batch size {batch_size}...")
    # --- THE FIX IS HERE ---
    results = pipe(
        texts_chunk,
        batch_size=batch_size,
        truncation=True,
        max_length=512  # Add this crucial argument
    )
    polarities = []
    for r in results:
        s = r['score']
        polarities.append(-s if r['label'].lower() == 'negative' else (s if r['label'].lower() == 'positive' else 0.0))
    print(f"[GPU Worker {gpu_id}] Analysis complete.")
    return polarities


def transformer_cpu_worker(texts_chunk):
    """A dedicated worker for a chunk of data on the CPU."""
    print(f"[CPU Worker {os.getpid()}] Process started...")
    # NOTE: No batch_size passed to pipeline, but specified in the call to pipe()
    pipe = pipeline("sentiment-analysis", model=TRANSFORMER_MODEL_NAME, device="cpu")
    print(f"[CPU Worker {os.getpid()}] Pipeline loaded. Analyzing {len(texts_chunk)} texts...")
    # Use a smaller batch size for CPU as large batches can be less efficient
    # --- THE FIX IS HERE ---
    results = pipe(
        texts_chunk,
        batch_size=16,  # A smaller batch size is often better for CPU
        truncation=True,
        max_length=512  # Add this crucial argument
    )
    polarities = []
    for r in results:
        s = r['score']
        polarities.append(-s if r['label'].lower() == 'negative' else (s if r['label'].lower() == 'positive' else 0.0))
    print(f"[CPU Worker {os.getpid()}] Analysis complete.")
    return polarities


# Placeholder for menu mapping
def transformer_analyze_placeholder(): pass


# --- 4. CORE DATA HANDLING & ANALYSIS ORCHESTRATION ---
def reservoir_sample_from_file(file_path, k):
    """
    Selects k random items from a file using Reservoir Sampling.

    This is memory-efficient as it only keeps k items in memory at a time,
    making it ideal for files larger than available RAM.

    Args:
        file_path (str): The path to the JSON Lines file.
        k (int): The number of items to sample.

    Returns:
        list: A list containing k randomly sampled review dictionaries.
    """
    reservoir = []
    lines_seen = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # Skip empty lines

            # The current item is the parsed JSON from the line
            item = json.loads(line)

            # Phase 1: Fill the reservoir with the first k items
            if lines_seen < k:
                reservoir.append(item)
            # Phase 2: For items beyond k, randomly decide whether to replace an element
            else:
                # Generate a random index from 0 to lines_seen
                j = random.randint(0, lines_seen)
                # The probability of this being true is k / (lines_seen + 1)
                if j < k:
                    reservoir[j] = item

            lines_seen += 1

    if lines_seen < k:
        print(
            f"⚠️ Warning: File contained only {lines_seen} reviews, which is less than the requested {k}. Returning all reviews.")

    return reservoir


def load_reviews_from_file(file_path, max_reviews=None):
    """
    Loads reviews from a file. Uses memory-efficient Reservoir Sampling for random sampling.
    """
    print("\n[PHASE 1: LOADING REVIEWS FROM FILE]")
    print("-" * 50)

    original_reviews = []

    try:
        # If max_reviews is not a positive number, load the entire file.
        if not max_reviews or max_reviews <= 0:
            print("Loading all reviews from the file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        original_reviews.append(json.loads(line))
        # Otherwise, use Reservoir Sampling for a memory-efficient random sample.
        else:
            print(f"🚀 Using Reservoir Sampling to select {max_reviews} random reviews...")
            print("(This is memory-safe and processes the file in a single pass)")
            original_reviews = reservoir_sample_from_file(file_path, max_reviews)

    except FileNotFoundError:
        print(f"\n[ERROR] File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] A line in '{file_path}' is not valid JSON. Halting.")
        print(f"        Details: {e}")
        return None
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during file loading: {e}")
        return None

    print(f"✅ Successfully loaded {len(original_reviews)} reviews into memory.")
    return original_reviews


def run_analysis(reviews_data, analysis_function, hardware_info, batch_size=256):
    """Orchestrates analysis using the optimal strategy for the detected hardware."""
    print("\n[PHASE 2: PERFORMING SENTIMENT ANALYSIS]")
    print(f"Using analysis method: {analysis_function.__name__}")
    print("-" * 50)

    texts_to_analyze = [f"{review.get('title', '')}. {review.get('text', '')}" for review in reviews_data]
    all_polarities = []

    # --- STRATEGY 1: TRANSFORMER ANALYSIS ---
    if analysis_function == transformer_analyze_placeholder:
        # --- SUB-STRATEGY: NVIDIA MULTI-GPU (CUDA) ---
        if hardware_info["type"] == "cuda":
            num_gpus = hardware_info["count"]
            print(f"🚀 Starting Multi-GPU analysis on {num_gpus} CUDA device(s)...")
            data_chunks = np.array_split(texts_to_analyze, num_gpus)
            worker_args = [(chunk.tolist(), i, batch_size) for i, chunk in enumerate(data_chunks)]
            with multiprocessing.Pool(processes=num_gpus) as pool:
                results_chunks = pool.starmap(transformer_gpu_worker, worker_args)
            all_polarities = [item for sublist in results_chunks for item in sublist]

        # --- SUB-STRATEGY: APPLE SILICON (MPS) ---
        elif hardware_info["type"] == "mps":
            print(f"🚀 Starting GPU analysis on Apple Silicon (MPS) device...")
            pipe = pipeline("sentiment-analysis", model=TRANSFORMER_MODEL_NAME, device="mps")
            print(f"Analyzing {len(texts_to_analyze)} texts with batch size {batch_size}...")
            # --- THE FIX IS HERE ---
            results = pipe(
                texts_to_analyze,
                batch_size=batch_size,
                truncation=True,
                max_length=512  # Add this crucial argument
            )
            for r in results:
                s = r['score']
                all_polarities.append(
                    -s if r['label'].lower() == 'negative' else (s if r['label'].lower() == 'positive' else 0.0))

        # --- SUB-STRATEGY: CPU FALLBACK ---
        else:
            print(
                f"⚠️ GPU not available for Transformer. Using {hardware_info['count']} CPU cores (this may be slow)...")
            data_chunks = np.array_split(texts_to_analyze, hardware_info['count'])
            with multiprocessing.Pool(processes=hardware_info['count']) as pool:
                results_chunks = pool.map(transformer_cpu_worker, data_chunks)
            all_polarities = [item for sublist in results_chunks for item in sublist]

    # --- STRATEGY 2: CPU-BOUND PARALLEL ANALYSIS (VADER, TEXTBLOB) ---
    else:
        num_cores = hardware_info['count']
        print(f"🚀 Starting parallel analysis on {num_cores} CPU cores...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            all_polarities = pool.map(analysis_function, texts_to_analyze)

    print("✅ Analysis complete.")

    # --- MERGE RESULTS ---
    print("\n[PHASE 3: MERGING RESULTS WITH ORIGINAL DATA]")
    for review, polarity in zip(reviews_data, all_polarities):
        review['sentiment_polarity'] = polarity
        review['sentiment_label'] = "Positive" if polarity > 0.05 else ("Negative" if polarity < -0.05 else "Neutral")
    print("✅ Merging complete.")
    print("-" * 50)
    return reviews_data


# --- [UNCHANGED] SUMMARY AND PLOTTING FUNCTIONS ---
# print_analysis_summary, plot_sentiment_density_curve, etc. all remain the same.
def print_analysis_summary(analyzed_reviews):
    """Prints a detailed summary of the analysis results to the terminal."""
    print("\n[PHASE 2: ANALYSIS SUMMARY]")
    print("-" * 30)

    total_reviews = len(analyzed_reviews)
    if total_reviews == 0:
        print("No reviews were analyzed.")
        return

    labels = [review['sentiment_label'] for review in analyzed_reviews]
    sentiment_counts = Counter(labels)
    positive_count = sentiment_counts.get("Positive", 0)
    neutral_count = sentiment_counts.get("Neutral", 0)
    negative_count = sentiment_counts.get("Negative", 0)

    total_polarity = sum(review['sentiment_polarity'] for review in analyzed_reviews)
    average_polarity = total_polarity / total_reviews if total_reviews > 0 else 0

    print(f"Total Reviews Analyzed: {total_reviews}")
    print(f"  - Positive Reviews: {positive_count} ({positive_count / total_reviews:.1%})")
    print(f"  - Neutral Reviews:  {neutral_count} ({neutral_count / total_reviews:.1%})")
    print(f"  - Negative Reviews: {negative_count} ({negative_count / total_reviews:.1%})")
    print("-" * 30)
    print(f"Average Sentiment Polarity: {average_polarity:.4f}")

    if average_polarity > 0.1:
        print("Overall sentiment of the sample is Positive.")
    elif average_polarity < -0.1:
        print("Overall sentiment of the sample is Negative.")
    else:
        print("Overall sentiment of the sample is Neutral.")

    print("-" * 30)
    print("[PHASE 2 COMPLETE]")


# --- 4. PLOTTING FUNCTIONS ---
# This section has been expanded with new graph types.

def plot_sentiment_density_curve(analyzed_reviews):
    """
    Generates a density curve (KDE plot) of sentiment polarities.
    This shows the distribution (or "spectrum") of sentiment scores.
    This plot type satisfies the "Spectrum," "Density," and "Curve" requirements.
    """
    print("  -> Creating Sentiment Polarity Density Curve (Spectrum Graph)...")
    polarities = [review['sentiment_polarity'] for review in analyzed_reviews]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(polarities, fill=True, color="skyblue", bw_adjust=0.5)
    plt.title('Spectrum of Sentiment Polarity (Density Curve)', fontsize=16)
    plt.xlabel('Sentiment Polarity (-1 to 1)')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(0, color='grey', linestyle='--', label='Neutral')
    plt.text(0.05, plt.gca().get_ylim()[1] * 0.9, 'Positive ->', ha='left', va='center', backgroundcolor='white')
    plt.text(-0.05, plt.gca().get_ylim()[1] * 0.9, '<- Negative', ha='right', va='center', backgroundcolor='white')
    plt.legend()


def plot_rating_sentiment_heatmap(analyzed_reviews):
    """
    Generates a heatmap to show the frequency of each sentiment label
    for each star rating.
    """
    print("  -> Creating Rating vs. Sentiment Heatmap...")
    df = pd.DataFrame(analyzed_reviews)

    if 'rating' not in df.columns or 'sentiment_label' not in df.columns:
        print("     [SKIP] Heatmap requires 'rating' and 'sentiment_label' data.")
        return

    contingency_table = pd.crosstab(df['sentiment_label'], df['rating'])
    contingency_table = contingency_table.reindex(['Positive', 'Neutral', 'Negative'])

    plt.figure(figsize=(10, 7))
    sns.heatmap(contingency_table, annot=True, fmt='.2f', cmap='YlGnBu', linewidths=.5)
    plt.title('Heatmap of Sentiment Label vs. Star Rating', fontsize=16)
    plt.xlabel('Star Rating')
    plt.ylabel('Sentiment Label')


def plot_sentiment_bubble_chart(analyzed_reviews):
    """
    Generates a bubble chart to visualize three dimensions:
    - X-axis: Star Rating
    - Y-axis: Sentiment Polarity
    - Bubble Size: Length of the review text
    """
    print("  -> Creating Sentiment Bubble Chart...")
    df = pd.DataFrame(analyzed_reviews)

    if 'rating' not in df.columns or 'text' not in df.columns:
        print("     [SKIP] Bubble chart requires 'rating' and 'text' data.")
        return

    df['text_length'] = df['text'].str.len().fillna(0) + 1
    color_map = {'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'}
    df['color'] = df['sentiment_label'].map(color_map)

    # Add jitter to x-axis to prevent points from overlapping vertically
    x_jitter = 0.15 * np.random.randn(len(df['rating']))

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        x=df['rating'] + x_jitter,
        y=df['sentiment_polarity'],
        s=df['text_length'] / 4,  # Scale down size for better visualization
        c=df['color'],
        alpha=0.5,
        edgecolors='w',
        linewidth=0.5
    )

    plt.title('Bubble Chart: Rating vs. Sentiment (Size = Review Length)', fontsize=16)
    plt.xlabel('Star Rating (with jitter)')
    plt.ylabel('Sentiment Polarity (-1 to 1)')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create a legend for colors and sizes
    color_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
                     for l, c in color_map.items()]
    size_handles = [plt.scatter([], [], s=s / 4, c='gray', alpha=0.6, label=f'{s} chars') for s in [100, 500, 1000]]
    plt.legend(title="Legend", handles=color_handles + size_handles)


def plot_avg_sentiment_by_rating_curve(analyzed_reviews):
    """
    Generates a curve graph showing the average sentiment polarity for each
    star rating.
    """
    print("  -> Creating Average Sentiment by Rating Curve...")
    df = pd.DataFrame(analyzed_reviews)
    if 'rating' not in df.columns:
        print("     [SKIP] Curve graph requires 'rating' data.")
        return

    avg_sentiment = df.groupby('rating')['sentiment_polarity'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_sentiment, x='rating', y='sentiment_polarity', marker='o', color='purple', linewidth=2.5)
    plt.title('Curve of Average Sentiment Polarity per Star Rating', fontsize=16)
    plt.xlabel('Star Rating')
    plt.ylabel('Average Sentiment Polarity')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(avg_sentiment['rating'])


def plot_sentiment_distribution(analyzed_reviews):
    """Original bar chart plot."""
    print("  -> Creating Sentiment Distribution Bar Chart...")
    labels = [review['sentiment_label'] for review in analyzed_reviews]
    sentiment_counts = Counter(labels)
    all_categories = ["Positive", "Neutral", "Negative"]
    counts = [sentiment_counts.get(cat, 0) for cat in all_categories]
    colors = ['green', 'grey', 'red']
    plt.figure(figsize=(8, 6))
    plt.bar(all_categories, counts, color=colors)
    plt.title(f'Sentiment Distribution of {len(analyzed_reviews)} Reviews', fontsize=16)
    plt.xlabel('Sentiment Category')
    plt.ylabel('Number of Reviews')
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')


def plot_rating_vs_sentiment(analyzed_reviews):
    """Original scatter plot."""
    print("  -> Creating Rating vs. Sentiment Score Scatter Plot...")
    ratings = [review.get('rating', 0) for review in analyzed_reviews]
    polarities = [review['sentiment_polarity'] for review in analyzed_reviews]
    plt.figure(figsize=(10, 7))
    sns.regplot(x=ratings, y=polarities, ci=None, scatter_kws={'alpha': 0.6})
    plt.title('Star Rating vs. Calculated Sentiment Polarity', fontsize=16)
    plt.xlabel('Original Star Rating')
    plt.ylabel('Sentiment Polarity (-1 to +1)')
    plt.grid(True)


# --- 6. SCRIPT EXECUTION BLOCK ---
# --- 6. SCRIPT EXECUTION BLOCK ---
if __name__ == "__main__":
    # This is critical for CUDA + multiprocessing to work safely
    multiprocessing.set_start_method('spawn', force=True)

    # 1. Detect Hardware and Build Menu
    hardware = detect_hardware()

    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS CONFIGURATION")
    print("=" * 50)

    MAX_REVIEWS_TO_ANALYZE = int(input("Enter number of entries to analyze (0 for all): "))

    # Build a dynamic menu based on hardware
    print("\nEnter Analysis Method:")
    print("  1. VADER (Fast, Parallel CPU)")
    print("  2. TextBlob (Fast, Parallel CPU)")

    if hardware['type'] == 'cuda':
        print("  3. Transformer (Recommended: Multi-GPU, Optimized)")
    elif hardware['type'] == 'mps':
        print("  3. Transformer (Recommended: Apple Silicon GPU, Optimized)")
    else:
        print("  3. Transformer (Parallel CPU Fallback)")

    ANALYSIS_OPTION = int(input("\nSelection: "))

    function_map = {
        1: vader_analyze,
        2: textblob_analyze,
        3: transformer_analyze_placeholder,
    }
    analysis_func = function_map.get(ANALYSIS_OPTION)

    if not analysis_func:
        print("Invalid selection! Quitting.")
        sys.exit(1)

    start_time = time.time()

    # --- Main Workflow ---
    reviews = load_reviews_from_file("Cell_Phones_and_Accessories.jsonl", MAX_REVIEWS_TO_ANALYZE)

    if reviews:
        # Run analysis using the optimal strategy
        analysis_results = run_analysis(reviews, analysis_func, hardware, batch_size=128)

        end_time = time.time()
        print(f"\n✅ Total Analysis Finished In {end_time - start_time:.2f} seconds.\n")

        # Generate summary and plots
        print_analysis_summary(analysis_results)
        plot_rating_sentiment_heatmap(analysis_results)
        plot_sentiment_distribution(analysis_results)
        plot_rating_vs_sentiment(analysis_results)
        plot_avg_sentiment_by_rating_curve(analysis_results)
        plot_sentiment_density_curve(analysis_results)
        plt.show()
    else:
        print("\n--- Analysis failed or no data was loaded. ---")