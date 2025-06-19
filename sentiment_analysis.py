# ==============================================================================
#      LIMITED & VERBOSE SENTIMENT ANALYSIS SCRIPT (FOR JSON LINES)
# ==============================================================================
#
# This script performs sentiment analysis on a JSON Lines file but includes
# a configurable limit on the number of entries to process. This is essential
# for handling very large files efficiently.
#
# ==============================================================================

# --- 1. IMPORT LIBRARIES ---
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
import sys


# --- 2. DEFINE THE CORE ANALYSIS FUNCTION ---

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
                # --- THIS IS THE KEY LOGIC FOR LIMITING REVIEWS ---
                # Check if we have reached the specified limit.
                # This check happens *before* processing the line for max efficiency.
                if max_reviews is not None and len(analyzed_reviews) >= max_reviews:
                    print(f"\n---> Reached the analysis limit of {max_reviews} reviews. Stopping file read.")
                    break  # Exit the loop entirely.

                if not line.strip():
                    continue

                review = json.loads(line)
                combined_text = f"{review.get('title', '')}. {review.get('text', '')}"

                blob = TextBlob(combined_text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiment = "Positive"
                elif polarity < -0.1:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                print(f"  -> Processing Review #{i + 1}...")
                print(f"     Polarity Score: {polarity:.4f}  =>  Sentiment: {sentiment}")

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


# --- 3. DEFINE THE SUMMARY FUNCTION ---

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


# --- 4. PLOTTING FUNCTIONS (Unchanged) ---
# These do not need to be modified as they just work with the data they receive.

def plot_sentiment_distribution(analyzed_reviews):
    print("\n[PHASE 3: GENERATING GRAPHS]")
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
    print("  -> Creating Rating vs. Sentiment Score Scatter Plot...")
    ratings = [review.get('rating', 0) for review in analyzed_reviews]
    polarities = [review['sentiment_polarity'] for review in analyzed_reviews]
    plt.figure(figsize=(10, 7))
    sns.regplot(x=ratings, y=polarities, ci=None, scatter_kws={'alpha': 0.6})
    plt.title('Star Rating vs. Calculated Sentiment Polarity', fontsize=16)
    plt.xlabel('Original Star Rating')
    plt.ylabel('Sentiment Polarity (-1 to +1)')
    plt.grid(True)
    print("[PHASE 3 COMPLETE]")


# --- 5. SCRIPT EXECUTION BLOCK ---

if __name__ == "__main__":
    # ======================================================================
    #                          CONFIGURATION
    # ======================================================================
    # Set the maximum number of reviews to analyze from the file.
    # - To analyze the first 100 reviews, set this to 100.
    # - To analyze the entire file, set this to None.
    MAX_REVIEWS_TO_ANALYZE = 1000000
    # ======================================================================

    print("=" * 60)
    print("          SENTIMENT ANALYSIS PROGRAM INITIALIZED")
    print("=" * 60)

    json_file = "Cell_Phones_and_Accessories.jsonl"
    print(f"Target file: '{json_file}'")

    # Update the startup message to reflect the limit.
    if MAX_REVIEWS_TO_ANALYZE is not None:
        print(f"Analysis limit: Processing a maximum of {MAX_REVIEWS_TO_ANALYZE} reviews.")
    else:
        print("Analysis limit: Processing all reviews in the file.")

    # Pass the limit to the analysis function.
    analysis_results = analyze_review_sentiment_json_lines(
        json_file,
        max_reviews=MAX_REVIEWS_TO_ANALYZE
    )

    if analysis_results:
        print_analysis_summary(analysis_results)
        plot_sentiment_distribution(analysis_results)
        plot_rating_vs_sentiment(analysis_results)

        print("\n[FINAL STEP: DISPLAYING PLOTS]")
        print("Close the plot windows to exit the program.")
        print("=" * 60)

        plt.tight_layout()
        plt.show()
    else:
        print("\n--- Analysis failed or was halted. No results to show. ---")
        print("=" * 60)