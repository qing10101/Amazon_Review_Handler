# ==============================================================================
#      COMPREHENSIVE STATISTICAL & GRAPHICAL ANALYZER (IMPROVED)
# ==============================================================================
#
# This single script performs a full analysis of the review data:
#   1. It prints a detailed statistical "deep dive" report to the terminal,
#      now with robust quantile analysis for skewed data.
#   2. It then generates and displays a graphical dashboard, using a log scale
#      for better visualization of skewed distributions.
#
# Required libraries: pip install pandas matplotlib seaborn
#
# ==============================================================================

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def load_and_prepare_data(file_path, max_reviews=None):
    """
    Reads review data, loads it into a Pandas DataFrame, and performs
    necessary cleaning and feature engineering.
    """
    print("\n[PHASE 1: LOADING & PREPARING DATA]")
    print("-" * 50)

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_reviews is not None and i >= max_reviews:
                    print(f"---> Reached the processing limit of {max_reviews:,} reviews.")
                    break
                if line.strip():
                    data.append(json.loads(line))

        if not data:
            print("No data was loaded. Aborting.")
            return None
        print(f"Successfully loaded {len(data):,} reviews into memory.")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse a line in the file: {e}")
        return None

    # Create DataFrame and engineer features
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    df['text_length'] = df['text'].str.len().fillna(0).astype(int)
    df['has_images'] = df['images'].apply(lambda x: isinstance(x, list) and len(x) > 0)
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid timestamps

    print("Data preparation complete.")
    print("-" * 50)
    return df


def print_statistical_deep_dive(df):
    """
    Prints a comprehensive statistical deep dive report to the terminal.
    """
    print("\n[PHASE 2: STATISTICAL DEEP DIVE REPORT (TEXT)]")
    print("=" * 50)

    # Ratings Analysis
    print("\n--- RATING ANALYSIS (DEEP DIVE) ---")
    print("a) Descriptive Statistics:")
    print(df['rating'].describe().to_string())
    print("\nb) Central Tendency & Distribution:")
    print(
        f"   - Mean: {df['rating'].mean():.2f}, Median: {df['rating'].median():.2f}, Mode: {df['rating'].mode().iloc[0]:.2f}")
    print(f"   - Skewness: {df['rating'].skew():.2f}, Kurtosis: {df['rating'].kurt():.2f}")
    print("\nc) Frequency Distribution:")
    rating_counts = df['rating'].value_counts().sort_index()
    rating_percent = df['rating'].value_counts(normalize=True).sort_index() * 100
    print(pd.DataFrame({'Count': rating_counts, 'Percentage': rating_percent.round(2)}).to_string())
    print("-" * 50)

    # --- MODIFIED: REVIEW TEXT LENGTH ANALYSIS ---
    print("\n--- REVIEW TEXT LENGTH ANALYSIS (in characters) ---")
    print("This data is heavily skewed. Quantiles give a better picture than the mean.")
    # Define specific quantiles to get a better sense of the distribution
    quantiles = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0]
    length_desc = df['text_length'].describe(percentiles=quantiles)
    print(length_desc.apply("{:,.1f}".format).to_string())
    print("\nInterpretation:")
    print(f"  - 50% of reviews (the median) are shorter than {length_desc.loc['50%']:.0f} characters.")
    print(f"  - 95% of reviews are shorter than {length_desc.loc['95%']:.0f} characters.")
    print(f"  - The longest review is {length_desc.loc['max']:.0f} characters long, an extreme outlier.")
    print("-" * 50)

    # Time-Based Analysis
    print("\n--- TIME-BASED ANALYSIS ---")
    print("Reviews per Day of the Week:")
    print(df['day_of_week'].value_counts().to_string())
    print("-" * 50)

    print("[TEXT REPORT COMPLETE]")


def create_visual_dashboard(df):
    """
    Generates a graphical dashboard of plots from the review data.
    """
    print("\n[PHASE 3: GENERATING VISUAL DASHBOARD]")
    print("This may take a moment...")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(f'Statistical Dashboard of {len(df):,} Reviews', fontsize=20, weight='bold')

    # Plot 1: Rating Distribution
    sns.countplot(ax=axes[0, 0], x='rating', data=df, palette='viridis')
    axes[0, 0].set_title('Distribution of Star Ratings', fontsize=14)

    # --- MODIFIED: Plot 2: Review Text Length Distribution with Log Scale ---
    # Using a log scale is ideal for visualizing heavily skewed data like text length.
    # It makes the distribution of the vast majority of reviews visible.
    sns.histplot(ax=axes[0, 1], data=df, x='text_length', bins=50, color='purple', log_scale=True)
    axes[0, 1].set_title('Distribution of Review Text Length (Log Scale)', fontsize=14)
    axes[0, 1].set_xlabel('Text Length (characters) - Log Scaled')


    # Plot 3: Verified Purchase
    verified_counts = df['verified_purchase'].value_counts()
    axes[1, 0].pie(verified_counts, labels=verified_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=['#4CAF50', '#FFC107'])
    axes[1, 0].set_title('Verified vs. Unverified Purchases', fontsize=14)

    # Plot 4: Reviews Over Time
    df.set_index('timestamp').resample('M').size().plot(ax=axes[1, 1], marker='o')
    axes[1, 1].set_title('Number of Reviews Over Time (Monthly)', fontsize=14)
    plt.setp(axes[1, 1].get_xticklabels(), rotation=30, ha="right")

    # Plot 5: Rating by Image Presence
    sns.boxplot(ax=axes[2, 0], x='has_images', y='rating', data=df, palette='coolwarm')
    axes[2, 0].set_title('Rating vs. Image Presence', fontsize=14)

    # Plot 6: Rating by Verified Purchase
    sns.boxplot(ax=axes[2, 1], x='verified_purchase', y='rating', data=df, palette='coolwarm')
    axes[2, 1].set_title('Rating vs. Verified Purchase', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    print("Dashboard generated successfully.")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    MAX_REVIEWS_TO_ANALYZE = 100000

    print("=" * 50)
    print("     COMPREHENSIVE STATISTICAL ANALYZER")
    print("=" * 50)

    json_file = "Cell_Phones_and_Accessories.jsonl"
    print(f"Target file: '{json_file}'")

    if MAX_REVIEWS_TO_ANALYZE is not None:
        print(f"Processing limit: Analyzing a maximum of {MAX_REVIEWS_TO_ANALYZE:,} reviews.")
    else:
        print("Processing limit: Analyzing all reviews in the file.")

    # --- Main Execution Flow ---
    # Step 1: Load and prepare the data
    df_reviews = load_and_prepare_data(json_file, max_reviews=MAX_REVIEWS_TO_ANALYZE)

    if df_reviews is not None and not df_reviews.empty:
        # Step 2: Print the detailed text report to the console
        print_statistical_deep_dive(df_reviews)

        # Step 3: Create the graphical dashboard
        create_visual_dashboard(df_reviews)

        # Step 4: Display the plots
        print("\n[FINAL STEP: DISPLAYING PLOTS]")
        print("Close the plot window to exit the program.")
        print("=" * 50)
        plt.show()  # This line displays the plots and pauses the script
    else:
        print("\n--- Analysis failed or no data was loaded. ---")
        print("=" * 50)