# !pip install recordlinkage

import pandas as pd
import recordlinkage

# --- Step 1: Create our two datasets (Same as before) ---
pseudo_data = [
    {"pseudo_id": 1, "first_name": "john", "last_name": "smith", "dob": "1990-03-15", "city": "london"},
    {"pseudo_id": 2, "first_name": "mary", "last_name": "jones", "dob": "1985-11-20", "city": "paris"},
    {"pseudo_id": 3, "first_name": "susan", "last_name": "williams", "dob": "1992-07-30", "city": "tokyo"},
    {"pseudo_id": 4, "first_name": "jon", "last_name": "smyth", "dob": "1990-03-14", "city": "london"},
]
df_pseudo = pd.DataFrame(pseudo_data).set_index('pseudo_id') # RLTK works well with indices

public_data = [
    {"public_id": 101, "full_name": "Johnathan Smith", "first_name": "john", "last_name": "smith", "dob": "1990-03-15", "city": "london"},
    {"public_id": 102, "full_name": "Peter Pan", "first_name": "peter", "last_name": "pan", "dob": "2000-01-01", "city": "london"},
    {"public_id": 103, "full_name": "Mary Jones", "first_name": "mary", "last_name": "jones", "dob": "1985-11-20", "city": "paris"},
    {"public_id": 104, "full_name": "Sue Williams", "first_name": "susan", "last_name": "williams", "dob": "1992-07-30", "city": "tokyo"},
]
df_public = pd.DataFrame(public_data).set_index('public_id') # RLTK works well with indices

# --- Step 2: Generate Candidate Links (Indexing) ---
# This prevents comparing every record from df_pseudo with every record from df_public.
indexer = recordlinkage.Index()
# We'll only compare records where the 'city' is the same.
indexer.block('city')
candidate_links = indexer.index(df_pseudo, df_public)
print(f"Number of candidate pairs after blocking: {len(candidate_links)}")


# --- Step 3: Compare Candidate Pairs ---
compare_cl = recordlinkage.Compare()

# Define how to compare each column.
# Jaro-Winkler is good for short strings like names. Threshold means similarity must be > value.
compare_cl.string('first_name', 'first_name', method='jarowinkler', threshold=0.85, label='first_name_sim')
compare_cl.string('last_name', 'last_name', method='jarowinkler', threshold=0.85, label='last_name_sim')
# Levenshtein is good for comparing structured strings like dates.
compare_cl.string('dob', 'dob', method='levenshtein', threshold=0.80, label='dob_sim')

# Compute the feature vectors for the candidate pairs
features = compare_cl.compute(candidate_links, df_pseudo, df_public)
print("\nFeature comparison results (1.0 = similar, 0.0 = not similar):")
print(features)


# --- Step 4: Classify the Pairs ---
# We can sum the scores to get a total match confidence.
# A match on all 3 fields gives a score of 3.0. A match on 2 gives 2.0.
potential_matches = features[features.sum(axis=1) > 2] # Require at least 2 of the 3 fields to be very similar
print("\nIdentified Matches (score > 2):")
print(potential_matches)

# To see the full records for the matches:
print("\nFull data for identified matches:")
for pseudo_id, public_id in potential_matches.index:
    print("\n--- Match ---")
    print(df_pseudo.loc[pseudo_id])
    print("---")
    print(df_public.loc[public_id])
