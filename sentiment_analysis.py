# ==============================================================================
#      ENHANCED SENTIMENT ANALYSIS SCRIPT WITH ADVANCED VISUALIZATIONS
# ==============================================================================
#
# This script performs sentiment analysis on a JSON Lines file and generates
# a variety of advanced visualizations, including density curves, heatmaps,
# and bubble charts, in addition to standard bar and scatter plots.
# It includes a configurable limit for processing large files.
#
# ==============================================================================

# --- 1. IMPORT LIBRARIES ---
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
import sys
import pandas as pd  # Added for advanced data manipulation
import numpy as np  # Added for numerical operations (e.g., jitter)
from gemini_llm_handler import ask_gemini


def parse_first_number_from_llm_response(text: str):
    """
    Parses the first number (integer or float, positive or negative) from a string.

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


def llm_prompt_constructor(review_text:str):
    prompt = (f"You are asked to perform sentiment analysis for a user review.\nThe review is: {review_text}\n"
              f"Respond with a sentiment score between -1 (extreme negative) and 1 (extreme positive)")
    return prompt


# --- 2. DEFINE THE CORE ANALYSIS FUNCTION (Unchanged) ---
def llm_api_analyze_review_sentiment_json_lines(file_path, max_reviews=None):
    """
    Loads and analyzes reviews from a JSON Lines file, stopping after
    'max_reviews' have been processed.

    Args:
        file_path (str): The path to the file.
        max_reviews (int, optional): The maximum number of reviews to analyze.
                                     If None, the entire file is processed.
                                     Defaults to None.
    Returns:
        list: A list of review dictionaries with added sentiment data.
    """
    analyzed_reviews = []
    print("\n[PHASE 1: ANALYZING REVIEWS]")
    print("-" * 30)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_reviews is not None and len(analyzed_reviews) >= max_reviews:
                    print(f"\n---> Reached the analysis limit of {max_reviews} reviews. Stopping file read.")
                    break

                if not line.strip():
                    continue

                review = json.loads(line)
                combined_text = f"{review.get('title', '')}. {review.get('text', '')}"

                response = ask_gemini(llm_prompt_constructor(combined_text))
                polarity = parse_first_number_from_llm_response(response)

                if polarity > 0.05:
                    sentiment = "Positive"
                elif polarity < -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                # print(f"  -> Processing Review #{i + 1}...")
                # print(f"     Polarity Score: {polarity:.4f}  =>  Sentiment: {sentiment}")

                review['sentiment_polarity'] = polarity
                review['sentiment_label'] = sentiment
                analyzed_reviews.append(review)

    except FileNotFoundError:
        print(f"\n[ERROR] The file '{file_path}' was not found. Please check the path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] A line in '{file_path}' is not valid JSON. Halting analysis.")
        print(f"        Details: {e}")
        return None

    print("-" * 30)
    print("[PHASE 1 COMPLETE]")
    return analyzed_reviews



def analyze_review_sentiment_json_lines(file_path, max_reviews=None):
    """
    Loads and analyzes reviews from a JSON Lines file, stopping after
    'max_reviews' have been processed.

    Args:
        file_path (str): The path to the file.
        max_reviews (int, optional): The maximum number of reviews to analyze.
                                     If None, the entire file is processed.
                                     Defaults to None.
    Returns:
        list: A list of review dictionaries with added sentiment data.
    """
    analyzed_reviews = []
    print("\n[PHASE 1: ANALYZING REVIEWS]")
    print("-" * 30)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_reviews is not None and len(analyzed_reviews) >= max_reviews:
                    print(f"\n---> Reached the analysis limit of {max_reviews} reviews. Stopping file read.")
                    break

                if not line.strip():
                    continue

                review = json.loads(line)
                combined_text = f"{review.get('title', '')}. {review.get('text', '')}"

                blob = TextBlob(combined_text)
                polarity = blob.sentiment.polarity

                if polarity > 0.05:
                    sentiment = "Positive"
                elif polarity < -0.05:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                # print(f"  -> Processing Review #{i + 1}...")
                # print(f"     Polarity Score: {polarity:.4f}  =>  Sentiment: {sentiment}")

                review['sentiment_polarity'] = polarity
                review['sentiment_label'] = sentiment
                analyzed_reviews.append(review)

    except FileNotFoundError:
        print(f"\n[ERROR] The file '{file_path}' was not found. Please check the path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] A line in '{file_path}' is not valid JSON. Halting analysis.")
        print(f"        Details: {e}")
        return None

    print("-" * 30)
    print("[PHASE 1 COMPLETE]")
    return analyzed_reviews


# --- 3. DEFINE THE SUMMARY FUNCTION (Unchanged) ---

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
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
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


# --- 5. SCRIPT EXECUTION BLOCK ---

if __name__ == "__main__":
    # ======================================================================
    #                          CONFIGURATION
    # ======================================================================
    MAX_REVIEWS_TO_ANALYZE = int(input("Enter number of entries for processing.\nEnter zero or negative for processing"
                                       "entire file:"))
    # ======================================================================

    print("=" * 60)
    print("      ENHANCED SENTIMENT ANALYSIS PROGRAM INITIALIZED")
    print("=" * 60)

    json_file = "Cell_Phones_and_Accessories.jsonl"
    print(f"Target file: '{json_file}'")

    if MAX_REVIEWS_TO_ANALYZE > 0:
        print(f"Analysis limit: Processing a maximum of {MAX_REVIEWS_TO_ANALYZE} reviews.")
    else:
        print("Analysis limit: Processing all reviews in the file.")

    analysis_results = analyze_review_sentiment_json_lines(
        json_file,
        max_reviews=MAX_REVIEWS_TO_ANALYZE
    )

    if analysis_results:
        print_analysis_summary(analysis_results)

        print("\n[PHASE 3: GENERATING GRAPHS]")
        # --- Call all plotting functions ---
        # New plots fulfilling the request
        plot_sentiment_density_curve(analysis_results)  # Spectrum/Density/Curve
        plot_rating_sentiment_heatmap(analysis_results)  # Heatmap
        plot_sentiment_bubble_chart(analysis_results)  # Bubble
        plot_avg_sentiment_by_rating_curve(analysis_results)  # Explicit Curve

        # Original plots
        plot_sentiment_distribution(analysis_results)
        plot_rating_vs_sentiment(analysis_results)
        print("[PHASE 3 COMPLETE]")

        print("\n[FINAL STEP: DISPLAYING PLOTS]")
        print("Close the plot windows to exit the program.")
        print("=" * 60)

        plt.tight_layout()
        plt.show()
    else:
        print("\n--- Analysis failed or was halted. No results to show. ---")
        print("=" * 60)