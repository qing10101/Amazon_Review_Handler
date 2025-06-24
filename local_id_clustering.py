# ==============================================================================
#      JSONL CLUSTERING SCRIPT BY USER_ID (WITH ENTRY LIMIT)
# ==============================================================================
#
# This script reads a JSON Lines file, groups the entries by the 'user_id'
# field, and saves each user's group of reviews into a separate JSON file.
#
# It includes a configurable limit on the number of entries to process from
# the start of the file, which is useful for testing on large datasets.
#
# ==============================================================================

import json
import os
from collections import defaultdict
import sys

# --- 1. HELPER FUNCTION TO CREATE SAFE FILENAMES (Unchanged) ---

def sanitize_filename(name):
    """
    Removes characters from a string that are invalid in filenames.
    Args:
        name (str): The original string (e.g., a user_id).
    Returns:
        str: A sanitized string safe for use as a filename.
    """
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:100]

# --- 2. CORE CLUSTERING FUNCTION (MODIFIED) ---

def cluster_reviews_by_user(input_file_path, output_directory, max_entries=None):
    """
    Reads a JSON Lines file, clusters reviews by user_id, and saves each
    cluster to a new file. Stops after processing 'max_entries'.

    Args:
        input_file_path (str): The path to the source JSON Lines file.
        output_directory (str): The name of the directory to save clustered files.
        max_entries (int, optional): The maximum number of entries to process.
                                     If None, the entire file is processed.
                                     Defaults to None.
    """
    print("\n[PHASE 1: READING AND GROUPING REVIEWS]")
    print("-" * 40)

    if max_entries is not None:
        print(f"  -> Processing a maximum of {max_entries} entries from the input file.")
    else:
        print(f"  -> Processing all entries from the input file.")

    user_clusters = defaultdict(list)

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Use enumerate to get a counter for limiting the entries
            for line_count, line in enumerate(f, 1):
                # --- THIS IS THE KEY LOGIC FOR LIMITING ENTRIES ---
                # Check if we have exceeded the specified limit.
                # This check happens *before* processing the line for max efficiency.
                if max_entries is not None and line_count > max_entries:
                    print(f"\n---> Reached the processing limit of {max_entries} entries. Stopping file read.")
                    break  # Exit the loop entirely.

                if not line.strip():
                    continue

                try:
                    review = json.loads(line)
                    user_id = review.get('user_id')

                    if user_id:
                        user_clusters[user_id].append(review)
                    else:
                        print(f"  -> [WARNING] Line {line_count} is missing a 'user_id'. Skipping.")

                except json.JSONDecodeError:
                    print(f"  -> [WARNING] Could not decode JSON on line {line_count}. Skipping.")

    except FileNotFoundError:
        print(f"[ERROR] The input file '{input_file_path}' was not found.")
        sys.exit(1)

    total_reviews = sum(len(reviews) for reviews in user_clusters.values())
    total_users = len(user_clusters)

    if total_users == 0:
        print("No users found or no valid reviews processed. Exiting.")
        return

    print(f"  -> Successfully processed {total_reviews} reviews for {total_users} unique users.")
    print("[PHASE 1 COMPLETE]")


    print("\n[PHASE 2: WRITING CLUSTERED FILES]")
    print("-" * 40)

    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"  -> Output will be saved in the '{output_directory}/' directory.")
    except OSError as e:
        print(f"[ERROR] Could not create output directory '{output_directory}'. Reason: {e}")
        sys.exit(1)

    files_written = 0
    for user_id, reviews in user_clusters.items():
        safe_filename = sanitize_filename(user_id) + ".json"
        output_path = os.path.join(output_directory, safe_filename)

        try:
            with open(output_path, 'w', encoding='utf-8') as out_file:
                json.dump(reviews, out_file, indent=2)
            files_written += 1
        except IOError as e:
            print(f"  -> [ERROR] Could not write file for user {user_id}. Reason: {e}")

    print(f"  -> Successfully wrote {files_written} files.")
    print("[PHASE 2 COMPLETE]")


# --- 3. SCRIPT EXECUTION BLOCK ---

if __name__ == "__main__":
    # ======================================================================
    #                          CONFIGURATION
    # ======================================================================
    INPUT_JSONL_FILE = "Cell_Phones_and_Accessories.jsonl"
    OUTPUT_CLUSTER_DIR = "user_clusters"

    # Set the maximum number of entries (lines) to read from the file.
    # - To process only the first 10,000 entries, set this to 10000.
    # - To process the entire file, set this to None.
    MAX_ENTRIES_TO_PROCESS = int(input("Enter number of entries for processing.\nEnter zero or negative for processing "
                                       "entire file"))
    # ======================================================================

    print("=" * 60)
    print("      JSONL USER CLUSTERING PROGRAM (WITH LIMIT) INITIALIZED")
    print("=" * 60)

    print(f"Target file: '{INPUT_JSONL_FILE}'")
    if MAX_ENTRIES_TO_PROCESS > 0:
        print(f"Processing limit: First {MAX_ENTRIES_TO_PROCESS} entries.")
    else:
        print("Processing limit: None (entire file will be processed).")


    # Pass the limit to the clustering function
    cluster_reviews_by_user(
        INPUT_JSONL_FILE,
        OUTPUT_CLUSTER_DIR,
        max_entries=MAX_ENTRIES_TO_PROCESS
    )

    print("\n--- Clustering process finished. ---")
    print("=" * 60)