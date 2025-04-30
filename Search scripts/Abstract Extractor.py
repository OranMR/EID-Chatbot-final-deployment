import pandas as pd
import requests


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return None  # Return None if there's no abstract

    # Create a list to store words in the correct order
    abstract_words = []
    
    # Flatten and sort positions
    position_word_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word_map[pos] = word

    # Sort positions and reconstruct text
    sorted_positions = sorted(position_word_map.keys())
    for pos in sorted_positions:
        abstract_words.append(position_word_map[pos])

    return " ".join(abstract_words)  # Join words into a readable abstract
# Load CSV file (replace 'your_file.csv' with the actual file path)
df = pd.read_csv("/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Search/edinburgh_papers.csv")

# Ensure 'DOI' column is correctly named
doi_list = df['doi'].dropna().tolist()

# Function to fetch abstract from OpenAlex
def fetch_abstract_openalex(doi):
    base_url = "https://api.openalex.org/works/https://doi.org/"
    url = base_url + doi
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            abstract_index = data.get('abstract_inverted_index', None)
            return reconstruct_abstract(abstract_index) if abstract_index else None
        else:
            print(f"Failed to fetch {doi}, Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {doi}: {e}")
        return None

# Fetch abstracts for all DOIs
df['Abstract'] = df['doi'].apply(fetch_abstract_openalex)

# Save results
df.to_csv("output_with_abstracts.csv", index=False)
print("Abstracts Saved")
