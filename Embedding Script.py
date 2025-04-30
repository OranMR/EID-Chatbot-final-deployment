import os
import re
from pdfminer.high_level import extract_text
import numpy as np
import faiss
import openai
from tqdm import tqdm  # Import tqdm for progress bars
#for pipinstall, install pdfminer.six, faiss-cpu, tqdm

# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


def preprocess_text(text):
    """Clean and normalize extracted PDF text"""
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\b\d+\s*\|\s*Page\b', '', text)
    text = re.sub(r'\bPage\s*\d+\s*of\s*\d+\b', '', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'\f', ' ', text)  # Form feed characters
    
    # Remove headers/footers (customize based on your PDFs)
    text = re.sub(r'(?i)confidential|draft|internal use only', '', text)
    
    # Fix hyphenated words that span line breaks (common in PDFs)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # Clean up quotes and apostrophes
    text = (text
    .replace("‘", "'")
    .replace("’", "'")
    .replace("“", '"')
    .replace("”", '"')
)
    
    # Remove URLs and emails if needed
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    
    # Strip extra spaces and normalize spacing
    text = text.strip()
    return text


def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using pdfMiner."""
    pdf_texts = {}
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    
    # Progress bar for PDF loading
    for filename in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        filepath = os.path.join(directory, filename)
        raw_text = extract_text(filepath)
        pdf_texts[filename] = preprocess_text(raw_text)  # Apply preprocessing
    return pdf_texts


def chunk_text(text, chunk_size=500, overlap=150):
    """Chunk text into pieces with overlap between chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end position for this chunk
        end = min(start + chunk_size, len(text))
        
        if end < len(text):
            # Try to find a sentence boundary (., !, ?) near the end of chunk
            sentence_end = -1
            for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                pos = text.rfind(punct, start, end + 50)
                if pos > sentence_end and pos <= end + 50:
                    sentence_end = pos + 1  # Include the punctuation
            if sentence_end > start:
                end = sentence_end
            else:
                # No sentence boundary found, try to find a space
                space = text.rfind(' ', end - 50, end + 50)
                if space > start:
                    end = space + 1
        
        # Add the chunk to our list
        chunks.append(text[start:end].strip())
        
        # Move start position for next chunk, accounting for overlap
        start = end - overlap if end - overlap > start else end
    return chunks


def generate_embeddings(chunks, batch_size=20): #in batches, is cheaper and faster
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings", unit="batch"):
        batch = chunks[i:i + batch_size]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings, dtype=np.float32)


def store_embeddings_and_chunks(file, chunks, embeddings, filenames, output_dir='embeddings'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving chunks and embeddings for {file}...")

    # Save chunks with optional metadata
    chunks_filepath = os.path.join(output_dir, f"{file}.txt")
    with open(chunks_filepath, 'w', encoding='utf-8') as f:
        for i, (chunk, filename) in enumerate(zip(chunks, filenames)):
            # write the chunk
            f.write(f"Source: {filename}\n{chunk}")
            # only append the delimiter if it's not the last chunk
            if i < len(chunks) - 1:
                f.write("\n----\n")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create and save FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_filepath = os.path.join(output_dir, f"{file}.index")
    faiss.write_index(index, index_filepath)

    print(f"Successfully saved {len(chunks)} chunks and embeddings for {file}")


def processpdfs(base_directory):
    """Process all PDFs."""
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    
    # Main progress bar for overall subdirectories
    for file in tqdm(subdirectories, desc="Processing directories", unit="dir"):  
        file_dir = os.path.join(base_directory, f"{file}")
        if os.path.exists(file_dir):
            print(f"\nProcessing directory: {file}")
            # Load PDFs and extract text
            pdf_texts = load_pdfs_from_directory(file_dir)
            
            # Process each PDF's text
            all_chunks = []
            all_filenames = []
            
            # Progress bar for processing individual PDFs
            for filename, text in tqdm(pdf_texts.items(), desc="Processing PDFs", unit="pdf"):
                chunks = chunk_text(text, chunk_size=500, overlap=150)
                all_chunks.extend(chunks)
                all_filenames.extend([filename] * len(chunks))
            
            # Generate embeddings
            embeddings = generate_embeddings(all_chunks)
            
            # Store the embeddings and chunks
            store_embeddings_and_chunks(file, all_chunks, embeddings, all_filenames)
        else:
            print(f"{file} directory not found. Skipping.")

# Run the processing
if __name__ == "__main__":
    base_dir = '/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Embedding Papers'
    processpdfs(base_dir)
    print("\n✅ Processing complete!")