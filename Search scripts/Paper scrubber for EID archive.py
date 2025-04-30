import os
import re
import csv
import glob
import time
import requests
from bs4 import BeautifulSoup
import PyPDF2
from tqdm import tqdm

def extract_references_from_pdf(pdf_path):
    """Extract text from PDF file and find reference patterns."""
    references = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Add progress bar for page extraction if there are multiple pages
            if len(reader.pages) > 1:
                for page in tqdm(reader.pages, desc="Extracting pages", leave=False):
                    text += page.extract_text() + "\n"
            else:
                text = reader.pages[0].extract_text()
        
        # Pattern to match references: Authors (date). Title. Journal.
        pattern = r'([^(]+)\((\d{4})\)\.\s+([^.]+)\.\s+([^.]+)\.'
        pattern_alt = r'([^.]+)\.\s+(\d{4})\.\s+([^.]+)\.\s+([^.]+)\.'
        matches = re.findall(pattern, text)
        if not matches:
            matches = re.findall(pattern_alt, text)
        
        for match in matches:
            authors, year, title, journal = match
            references.append({
                'authors': authors.strip(),
                'year': year.strip(),
                'title': title.strip(),
                'journal': journal.strip(),
                'source_pdf': os.path.basename(pdf_path)
            })
            
        return references
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []

def search_pubmed(title):
    """Search PubMed for a paper title and extract DOI and PMID."""
    
    # Remove special characters that might affect the search
    clean_title = re.sub(r'[^\w\s]', ' ', title)
    
    # Construct the search URL
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    search_url = f"{base_url}?term={requests.utils.quote(clean_title)}"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the first result's PMID
        pmid = None
        pmid_element = soup.select_one('.docsum-pmid')
        if pmid_element:
            pmid = pmid_element.text.strip()
        
        # If we found a PMID, get the article details for DOI
        doi = None
        if pmid:
            article_url = f"{base_url}{pmid}/"
            article_response = requests.get(article_url)
            article_response.raise_for_status()
            
            article_soup = BeautifulSoup(article_response.text, 'html.parser')
            
            # Look for DOI in the article page
            doi_element = article_soup.select_one('.identifier.doi')
            if doi_element:
                doi = doi_element.text.strip().replace('DOI: ', '')
        
        return {
            'pmid': pmid,
            'doi': doi
        }
    
    except Exception as e:
        print(f"Error searching PubMed for '{title}': {e}")
        return {'pmid': None, 'doi': None}
    
    # Add a delay to avoid overloading the PubMed server
    time.sleep(1)

def main():
    # Directory containing PDF files
    pdf_dir = "C:/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Scrubbed papers/Archive"

    # Normalize the path and check if it exists
    pdf_dir = os.path.normpath(os.path.expanduser(pdf_dir))
    if not os.path.isdir(pdf_dir):
        print(f"Error: '{pdf_dir}' is not a valid directory")
        return
    
    print(f"Searching for PDF files in: {pdf_dir}")
    
    # Get all file paths regardless of case, but avoid duplicates with case-insensitive comparison
    # This approach works on both case-sensitive and case-insensitive file systems
    pdf_paths = set()
    for ext in ['.pdf', '.PDF']:
        found_paths = glob.glob(os.path.join(pdf_dir, f"*{ext}"))
        for path in found_paths:
            # Use canonical path to handle case differences on case-insensitive systems
            canonical_path = os.path.normcase(os.path.normpath(path))
            if canonical_path not in [os.path.normcase(os.path.normpath(p)) for p in pdf_paths]:
                pdf_paths.add(path)
    
    pdf_files = list(pdf_paths)
    
    # If no files found, try to list all files to help debug
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        print("Listing all files in the directory:")
        all_files = os.listdir(pdf_dir)
        for file in all_files:
            print(f"  - {file}")
        
        # Ask user if they want to specify a different extension
        other_ext = input("Enter another file extension to try (e.g., 'pdfs', leave empty to quit): ")
        if other_ext:
            if not other_ext.startswith('.'):
                other_ext = '.' + other_ext
            pdf_files = glob.glob(os.path.join(pdf_dir, f"*{other_ext}"))
            if not pdf_files:
                print(f"No files with extension '{other_ext}' found either.")
                return
        else:
            return
    
    print(f"Found {len(pdf_files)} unique PDF files")
    
    # List the first few files to confirm
    for i, file in enumerate(pdf_files[:5], 1):
        print(f"  {i}. {os.path.basename(file)}")
    if len(pdf_files) > 5:
        print(f"  ... (and {len(pdf_files) - 5} more)")
    
    output_file = os.path.join(pdf_dir, "pubmed_results.csv")
    
    # Create CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['source_pdf', 'authors', 'year', 'title', 'journal', 'pmid', 'doi']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Main progress bar for PDF processing
        with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar_pdfs:
            for pdf_path in pdf_files:
                pbar_pdfs.set_postfix_str(f"{os.path.basename(pdf_path)}")
                
                references = extract_references_from_pdf(pdf_path)
                
                # Progress bar for references in current PDF
                if references:
                    ref_count = len(references)
                    pbar_pdfs.write(f"Found {ref_count} references in {os.path.basename(pdf_path)}")
                    
                    with tqdm(total=ref_count, desc="Searching PubMed", unit="ref", leave=False) as pbar_refs:
                        for ref in references:
                            short_title = ref['title'][:30] + "..." if len(ref['title']) > 30 else ref['title']
                            pbar_refs.set_postfix_str(f"{short_title}")
                            
                            pubmed_info = search_pubmed(ref['title'])
                            
                            # Combine reference and PubMed info
                            ref_data = {**ref, **pubmed_info}
                            writer.writerow(ref_data)
                            
                            # Flush to write immediately
                            csvfile.flush()
                            pbar_refs.update(1)
                else:
                    pbar_pdfs.write(f"No references found in {os.path.basename(pdf_path)}")
                
                pbar_pdfs.update(1)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()