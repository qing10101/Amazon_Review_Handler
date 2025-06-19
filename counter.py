# ==============================================================================
#                      FAST ENTRY COUNTER SCRIPT
# ==============================================================================
#
# This script is designed for one purpose: to efficiently count the number of
# valid entries in a large JSON Lines file (one JSON object per line).
#
# WHY THIS APPROACH?
# ------------------
# Instead of reading the entire file into memory with methods like
# `file.readlines()`, which would crash on very large files, this script
# streams the file. It reads one line at a time, increments a counter, and
# then discards the line, using a minimal amount of memory.
#
# It also correctly ignores any blank lines in the file.
#
# No external libraries are needed.
#
# ==============================================================================

import sys  # Used for exiting the script gracefully on error


def count_valid_entries(file_path):
    """
    Counts the number of non-empty lines in a given file.

    Args:
        file_path (str): The path to the file to be processed.

    Returns:
        int: The total count of non-empty lines, or None if the file is not found.
    """
    entry_count = 0

    print(f"Opening '{file_path}' to count entries...")
    print("This may take a while for very large files. Please wait.")

    try:
        # The 'with' statement ensures the file is handled safely and closed automatically.
        # We specify a buffer size to potentially speed up reading from disk.
        with open(file_path, 'r', encoding='utf-8', buffering=8192) as f:
            # This is the most memory-efficient way to iterate over a file in Python.
            for line in f:
                # We only increment the counter if the line is not empty or just whitespace.
                if line.strip():
                    entry_count += 1

        return entry_count

    except FileNotFoundError:
        print(f"\n[ERROR] The file '{file_path}' was not found.")
        print("        Please make sure the file exists in the same directory as the script.")
        return None
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Set the name of the file you want to count.
    FILENAME = "Cell_Phones_and_Accessories.jsonl"
    # ---------------------

    print("=" * 40)
    print("      FAST FILE ENTRY COUNTER")
    print("=" * 40)

    # Call the counting function and store the result.
    total_entries = count_valid_entries(FILENAME)

    # The 'if total_entries is not None' check handles the case where the
    # function returned an error (e.g., FileNotFoundError).
    if total_entries is not None:
        print("\n--- COUNT COMPLETE ---")
        # The f-string format specifier `{:,}` automatically adds commas
        # to large numbers, making them much more readable.
        print(f"✅ Total number of entries found: {total_entries:,}")
    else:
        print("\n--- SCRIPT HALTED DUE TO ERROR ---")

    print("=" * 40)