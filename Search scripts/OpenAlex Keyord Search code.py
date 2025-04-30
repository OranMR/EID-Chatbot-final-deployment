import requests
import pandas as pd

# Define search parameters
city = "Edinburgh"
keyword = "antimicrobial resistance"

# Step 1: Get institutions in the UK
inst_url = "https://api.openalex.org/institutions?filter=country_code:GB"
inst_response = requests.get(inst_url).json()

# Step 2: Filter institutions based on city name
edinburgh_institutions = [
    inst for inst in inst_response.get("results", []) 
    if city.lower() in inst["display_name"].lower()
]

if not edinburgh_institutions:
    print("No institutions found in the specified city.")
    exit()

# Step 3: Get institution IDs
institution_ids = [inst["id"] for inst in edinburgh_institutions]
inst_filter = "|".join(institution_ids)  # Combine for OR filtering

# Step 4: Retrieve all works using pagination
all_papers = []
works_url = f"https://api.openalex.org/works?filter=institutions.id:{inst_filter},title.search:{keyword}&per_page=200"

while works_url:
    works_response = requests.get(works_url).json()
    
    # Extract required details
    papers = [
        {
            "title": work["title"],
            "doi": work.get("doi", "No DOI available"),
        }
        for work in works_response.get("results", [])
    ]

    all_papers.extend(papers)

    # Get next page URL
    works_url = works_response.get("next_page")

# Step 5: Save results to CSV
df = pd.DataFrame(all_papers)
df.to_csv("edinburgh_papers.csv", index=False)

print(f"CSV file saved with {len(all_papers)} papers")



