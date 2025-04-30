import requests
from bs4 import BeautifulSoup
import csv
import time
from urllib.parse import urlparse, parse_qs

def extract_pmid_from_url(url):
    """Extract PubMed ID from a PubMed URL."""
    parsed_url = urlparse(url)
    
    # Handle different URL formats
    if 'pubmed' in parsed_url.netloc or 'pubmed' in parsed_url.path:
        # For URLs like https://pubmed.ncbi.nlm.nih.gov/12345678/
        path_parts = parsed_url.path.strip('/').split('/')
        if path_parts and path_parts[-1].isdigit():
            return path_parts[-1]
        
        # For URLs with query parameters like ?term=12345678
        query_params = parse_qs(parsed_url.query)
        if 'term' in query_params and query_params['term'][0].isdigit():
            return query_params['term'][0]
    
    return None

def get_article_info(pmid):
    """Get article title and DOI from PubMed ID using the PubMed API."""
    time.sleep(1)  # Be nice to the PubMed server
    
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if pmid in data.get('result', {}):
            article_data = data['result'][pmid]
            title = article_data.get('title', 'Title not found')
            
            # Try to get DOI - it's usually in the article ids
            doi = None
            if 'articleids' in article_data:
                for id_obj in article_data['articleids']:
                    if id_obj.get('idtype') == 'doi':
                        doi = id_obj.get('value')
                        break
            
            return {
                'pmid': pmid,
                'title': title,
                'doi': doi
            }
    
    return {'pmid': pmid, 'title': 'Error retrieving data', 'doi': None}

def scrape_pubmed_links(url):
    """Scrape all PubMed links from a webpage."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links
        links = soup.find_all('a', href=True)
        
        # Filter for PubMed links
        pubmed_links = []
        for link in links:
            href = link['href']
            if 'pubmed' in href.lower() or 'ncbi.nlm.nih.gov' in href.lower():
                pubmed_links.append(href)
        
        return pubmed_links
    
    except Exception as e:
        print(f"Error scraping webpage: {e}")
        return []

def main():
    # Get the webpage URL from user
    url = "https://edinburgh-infectious-diseases.ed.ac.uk/our-research/our-publications/recent-publications"
    
    print("Scraping PubMed links from the webpage...")
    pubmed_links = scrape_pubmed_links(url)
    
    if not pubmed_links:
        print("No PubMed links found on the webpage.")
        return
    
    print(f"Found {len(pubmed_links)} PubMed links.")
    
    # Extract PubMed IDs from the links
    pmids = []
    for link in pubmed_links:
        pmid = extract_pmid_from_url(link)
        if pmid and pmid not in pmids:  # Avoid duplicates
            pmids.append(pmid)
    
    if not pmids:
        print("Could not extract any valid PubMed IDs from the links.")
        return
    
    print(f"Processing {len(pmids)} unique PubMed articles...")
    
    # Get article info for each PMID
    articles = []
    for i, pmid in enumerate(pmids):
        print(f"Processing article {i+1}/{len(pmids)}: PMID {pmid}")
        article_info = get_article_info(pmid)
        articles.append(article_info)
    
    # Save to CSV
    output_file = "recent_pubmed_articles.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['pmid', 'title', 'doi']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for article in articles:
            writer.writerow(article)
    
    print(f"\nScraped {len(articles)} articles. Data saved to {output_file}")

if __name__ == "__main__":
    main()