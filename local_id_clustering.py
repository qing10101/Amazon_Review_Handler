# ==============================================================================
#      OPTIMIZED & RANDOMIZED JSONL CLUSTERING SCRIPT BY USER_ID
# ==============================================================================
#
# This script reads a JSON Lines file, takes a memory-efficient random sample
# of entries, groups them by 'user_id', and writes the output files in
# parallel to maximize speed.
#
# ==============================================================================

import json
import os
import random
import sys
from collections import defaultdict
import multiprocessing
from functools import partial


# --- 1. HELPER FUNCTION TO CREATE SAFE FILENAMES (Unchanged) ---

def sanitize_filename(name):
    """Removes characters from a string that are invalid in filenames."""
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name[:100]


# --- 2. OPTIMIZED DATA LOADING & CLUSTERING ---

def load_and_cluster_reviews(input_file_path, num_to_sample=None):
    """
    Loads reviews from a file, clusters them by user_id.
    Uses Reservoir Sampling for memory-efficient random sampling.
    """
    print("\n[PHASE 1: LOADING & CLUSTERING REVIEWS]")
    print("-" * 50)

    user_clusters = defaultdict(list)

    try:
        if num_to_sample and num_to_sample > 0:
            # --- OPTIMIZATION 1: RESERVOIR SAMPLING ---
            # This reads the file ONCE and uses minimal memory.
            print(f"🚀 Using Reservoir Sampling to select {num_to_sample} random reviews...")

            reservoir = []
            lines_seen = 0
            with open(input_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue

                    if lines_seen < num_to_sample:
                        reservoir.append(json.loads(line))
                    else:
                        j = random.randint(0, lines_seen)
                        if j < num_to_sample:
                            reservoir[j] = json.loads(line)
                    lines_seen += 1

            print(f"Sampled {len(reservoir)} reviews from {lines_seen} total lines.")
            # Now, cluster the sampled reviews
            for review in reservoir:
                user_id = review.get('user_id')
                if user_id:
                    user_clusters[user_id].append(review)

        else:
            # Load all reviews if no sampling is requested
            print("Loading all reviews from the file...")
            with open(input_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    review = json.loads(line)
                    user_id = review.get('user_id')
                    if user_id:
                        user_clusters[user_id].append(review)

    except FileNotFoundError:
        print(f"[ERROR] The input file '{input_file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] A line in the file is not valid JSON. Halting. Details: {e}")
        return None

    total_reviews = sum(len(reviews) for reviews in user_clusters.values())
    total_users = len(user_clusters)

    if total_users == 0:
        print("No users found or no valid reviews processed. Exiting.")
        return None

    print(f"✅ Successfully processed {total_reviews} reviews for {total_users} unique users.")
    print("[PHASE 1 COMPLETE]")
    return user_clusters


# --- 3. PARALLEL FILE WRITING ---

def write_cluster_to_file(user_reviews_tuple, output_directory):
    """
    Worker function to write a single user's cluster to a JSON file.
    Designed to be called by a multiprocessing pool.
    """
    user_id, reviews = user_reviews_tuple
    safe_filename = sanitize_filename(user_id) + ".json"
    output_path = os.path.join(output_directory, safe_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(reviews, out_file, indent=2)
        return True
    except IOError as e:
        print(f"  -> [ERROR] Could not write file for user {user_id}. Reason: {e}")
        return False


def write_clusters_parallel(user_clusters, output_directory):
    """
    Writes all user clusters to files in parallel using all available CPU cores.
    """
    print("\n[PHASE 2: WRITING CLUSTERED FILES IN PARALLEL]")
    print("-" * 50)

    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"  -> Output will be saved in the '{output_directory}/' directory.")
    except OSError as e:
        print(f"[ERROR] Could not create output directory '{output_directory}'. Reason: {e}")
        sys.exit(1)

    # --- OPTIMIZATION 2: PARALLEL PROCESSING ---
    # Prepare the data for the pool: a list of (user_id, reviews) tuples
    tasks = user_clusters.items()
    num_cores = multiprocessing.cpu_count()
    print(f"🚀 Writing {len(tasks)} files using {num_cores} CPU cores...")

    # partial freezes the 'output_directory' argument for the worker function
    worker_func = partial(write_cluster_to_file, output_directory=output_directory)

    with multiprocessing.Pool(processes=num_cores) as pool:
        # map() distributes the tasks among the worker processes
        results = pool.map(worker_func, tasks)

    files_written = sum(1 for result in results if result is True)

    print(f"✅ Successfully wrote {files_written} files.")
    print("[PHASE 2 COMPLETE]")


# --- 4. SCRIPT EXECUTION BLOCK ---

if __name__ == "__main__":
    # Best practice for multiprocessing on macOS/Windows
    multiprocessing.set_start_method('spawn', force=True)

    # ======================================================================
    #                          CONFIGURATION
    # ======================================================================
    INPUT_JSONL_FILE = "Cell_Phones_and_Accessories.jsonl"
    OUTPUT_CLUSTER_DIR = "user_clusters"

    print("=" * 60)
    print("      OPTIMIZED JSONL USER CLUSTERING PROGRAM")
    print("=" * 60)

    try:
        num_input = int(input("Enter number of RANDOM reviews to process (0 or negative for all): "))
    except ValueError:
        print("Invalid input. Please enter a number. Exiting.")
        sys.exit(1)

    # ======================================================================

    print(f"Target file: '{INPUT_JSONL_FILE}'")
    if num_input > 0:
        print(f"Processing limit: A random sample of {num_input} reviews.")
    else:
        print("Processing limit: All reviews in the file.")
        num_input = None  # Set to None to process the entire file

    # Run the main workflow
    clusters = load_and_cluster_reviews(INPUT_JSONL_FILE, num_to_sample=num_input)

    if clusters:
        write_clusters_parallel(clusters, OUTPUT_CLUSTER_DIR)

    print("\n--- Clustering process finished. ---")
    print("=" * 60)