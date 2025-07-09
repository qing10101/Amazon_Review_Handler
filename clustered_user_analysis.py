# ==============================================================================
#      LLM-POWERED RE-IDENTIFICATION RISK ANALYZER
# ==============================================================================
#
# This script iterates through user-specific JSON files, consolidates all of
# a user's review text, and uses an LLM to analyze the potential for
# re-identification based on the content.
#
# ==============================================================================

import json
import os
import sys
import glob
# Assume ollama_llm_handler.py containing ask_ollama is in the same directory
from ollama_llm_handler import ask_ollama


# --- 1. PROMPT ENGINEERING FUNCTION ---

def create_reidentification_prompt(combined_user_text: str) -> str:
    """
    Creates a detailed prompt for the LLM to analyze text for re-identification risks.

    Args:
        combined_user_text: A single string containing all reviews from one user.

    Returns:
        A string containing the fully formatted prompt.
    """
    prompt = f"""
Your task is to analyze the following block of text, which contains all reviews written by a single anonymous user, and assess the risk of re-identification.

Look for specific phrases, patterns, or pieces of information that could, when combined, uniquely identify an individual. This includes, but is not limited to:
- **Direct PII:** Names of people, specific addresses, email addresses, phone numbers.
- **Quasi-Identifiers:** Specific locations (e.g., "my office in downtown Seattle," "the playground on Elm Street"), employer names ("I work at Acme Corp"), specific dates and events ("my son's birthday party on June 5th").
- **Unique Personal Stories:** Highly detailed anecdotes or life events that are unique.
- **Cross-Linkable Information:** Mentions of other platforms, usernames, or unique hobbies that could be searched for elsewhere.

Analyze the following text block:
---
"{combined_user_text}"
---

Based on your analysis, provide a summary of the re-identification risk. 
1.  State the overall risk level (e.g., None, Low, Medium, High).
2.  List the specific phrases or data points that contribute to this risk.
3.  For each point, briefly explain WHY it poses a risk.
4.  If no significant risks are found, state that clearly.
"""
    return prompt


# --- 2. SINGLE USER ANALYSIS FUNCTION ---

def analyze_single_user_file(file_path: str) -> str:
    """
    Reads a user's JSON file, combines their review texts, and gets an LLM analysis.

    Args:
        file_path: The full path to the user's .json file.

    Returns:
        A string containing the LLM's risk analysis.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return "Analysis Result: SKIPPED - File is empty."

        # Consolidate all text from that user's reviews
        all_texts = []
        for review in data:
            title = review.get('title', '')
            text = review.get('text', '')
            if title: all_texts.append(title)
            if text: all_texts.append(text)

        combined_text = ' '.join(all_texts)

        if not combined_text.strip():
            return "Analysis Result: SKIPPED - No review text found for this user."

        # Get the LLM analysis
        prompt = create_reidentification_prompt(combined_text)
        risk_analysis = ask_ollama(prompt)  # Using a slightly larger model for better reasoning

        return f"LLM Analysis Result:\n{risk_analysis}"

    except json.JSONDecodeError:
        return "Analysis Result: FAILED - Could not parse the JSON file."
    except Exception as e:
        return f"Analysis Result: FAILED - An unexpected error occurred: {e}"


# --- 3. MAIN ORCHESTRATOR ---

def analyze_all_users_in_directory(directory_path: str):
    """
    Finds all user JSON files in a directory and analyzes each one for risk.
    """
    print("=" * 60)
    print("      Starting Re-identification Risk Analysis")
    print("=" * 60)

    # Use glob to easily find all .json files in the directory
    user_files = glob.glob(os.path.join(directory_path, '*.json'))

    if not user_files:
        print(f"[ERROR] No user cluster files (.json) found in the directory: '{directory_path}'")
        print("Please run the clustering script first.")
        sys.exit(1)

    print(f"Found {len(user_files)} user files to analyze...\n")

    for user_file_path in user_files:
        filename = os.path.basename(user_file_path)
        print(f"--- Analyzing User File: {filename} ---")

        # Get and print the analysis for the current user
        risk_result = analyze_single_user_file(user_file_path)
        print(risk_result)

        print("-" * 60 + "\n")


# --- 4. SCRIPT EXECUTION BLOCK ---

if __name__ == "__main__":
    # This should be the directory where your user JSON files are saved
    USER_CLUSTER_DIRECTORY = "user_clusters"

    # Make sure your Ollama server is running before executing this script
    print("NOTE: Ensure your local Ollama server is running.")

    analyze_all_users_in_directory(USER_CLUSTER_DIRECTORY)

    print("=" * 60)
    print("      All user files have been analyzed.")
    print("=" * 60)