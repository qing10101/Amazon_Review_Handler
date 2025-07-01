import pandas as pd
import json
import gemini_llm_handler


# --- Helper function for string similarity ---
def jaro_winkler_similarity(s1, s2):
    if not s1 or not s2: return 0.0
    len1, len2 = len(s1), len(s2)
    match_distance = max(len1, len2) // 2 - 1
    matches1, matches2, matches = [False] * len1, [False] * len2, 0
    for i in range(len1):
        start, end = max(0, i - match_distance), min(i + match_distance + 1, len2)
        for j in range(start, end):
            if not matches2[j] and s1[i] == s2[j]:
                matches1[i], matches2[j] = True, True
                matches += 1
                break
    if matches == 0: return 0.0
    transpositions, k = 0, 0
    for i in range(len1):
        if matches1[i]:
            while not matches2[k]: k += 1
            if s1[i] != s2[k]: transpositions += 1
            k += 1
    jaro_sim = ((matches / len1) + (matches / len2) + ((matches - transpositions) / 2) / matches) / 3.0
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro_sim + prefix * 0.1 * (1 - jaro_sim)


# --- Function to prepare data for the LLM ---
def serialize_record(record):
    print((f"User '{record.get('user_name', 'N/A')}' birth year '{record.get('birth_year', 'N/A')}' "
            f"from city '{record.get('city', 'N/A')}' "
            f"occupation'{record.get('occupation', 'N/A')}' "
            f"on date '{record.get('review_date', 'N/A')}'."))
    return (f"User '{record.get('user_name', 'N/A')}' birth year '{record.get('birth_year', 'N/A')}' "
            f"from city '{record.get('city', 'N/A')}' "
            f"occupation'{record.get('occupation', 'N/A')}' "
            f"on date '{record.get('review_date', 'N/A')}'.")


# --- Main Processing Logic ---
def process_two_blocks(block1, block2):
    matches = []
    for i in range(len(block1)):
        for j in range(len(block2)):
            record1 = block1[i]
            record2 = block2[j]

            # sim_score = jaro_winkler_similarity(record1['user_name'], record2['user_name'])

            # Construct the prompt for the LLM
            prompt = (
                "You are an expert entity resolution analyst. Determine if the following two records refer to the same "
                "person. "
                "Respond ONLY with a JSON object containing two keys: 'match' (boolean) and 'reason' (string).\n\n"
                f"Record 1: {serialize_record(record1)}\n"
                f"Record 2: {serialize_record(record2)}"
                f"Respond with a short conclusion"
            )

            reasons = gemini_llm_handler.ask_gemini_async(prompt)

            matches.append((record1['review_id'], record2['review_id'], reasons['match'], reasons['reason']))
    return matches


# --- Simulate the data and run the pipeline ---
simulated_block1 = [
    {"review_id": 1, "user_name": "jerry wang", "birth_year": "1986", "occupation": "engineer", "city": "london",
     "review_date": "2023-10-15"},
    {"review_id": 99, "user_name": "jason zhang", "birth_year": "1979", "occupation": "teacher", "city": "london",
     "review_date": "2023-10-16"},
    {"review_id": 150, "user_name": "jane", "birth_year": "1996", "occupation": "accountant", "city": "new york",
     "review_date": "2023-11-01"},
    {"review_id": 151, "user_name": "william", "birth_year": "1992", "occupation": "engineer", "city": "london",
     "review_date": "2023-10-15"}
]
simulated_block2 = [
    {"review_id": 2, "user_name": "jerry", "birth_year": "1986", "occupation": "engineer", "city": "boston",
     "review_date": "2023-03-05"},
    {"review_id": 59, "user_name": "j z", "birth_year": "1979", "occupation": "teacher", "city": "london",
     "review_date": "2023-06-26"},
    {"review_id": 100, "user_name": "jane doe", "birth_year": "1996", "occupation": "accountant", "city": "new york",
     "review_date": "2023-05-21"},
    {"review_id": 134, "user_name": "william wang", "birth_year": "1998", "occupation": "engineer", "city": "london",
     "review_date": "2023-01-11"}
]

print("--- Processing a single block of candidate records ---")
result = process_two_blocks(simulated_block1, simulated_block2)
print("\n--- Final Matches Found ---")
df = pd.DataFrame(result, columns=["ID1", "ID2", "Method", "Explanation"])
print(df)
