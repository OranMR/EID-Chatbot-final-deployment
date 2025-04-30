import os
import pandas as pd
import numpy as np
import faiss
import openai
from pdfminer.high_level import extract_text
import itertools
from tqdm import tqdm  # For progress bars
import re
from difflib import SequenceMatcher  # For fuzzy matching
import datetime  # Timestamp directory names

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def load_pdfs_from_directory(directory):
    """Load all PDFs from a directory and extract their text using PDFMiner."""
    pdf_texts = {}
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in {directory}")
        return pdf_texts
        
    for filename in tqdm(pdf_files, desc="Loading PDFs"): 
        filepath = os.path.join(directory, filename)
        try:
            text = extract_text(filepath)
            if not text.strip():
                print(f"WARNING: No text extracted from {filename}")
                continue
            # text.strip() = normalize whitespace (converts spaces, tabs, enters etc into 1 space)
            text = re.sub(r'\s+', ' ', text).strip()
            pdf_texts[filename] = text
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return pdf_texts

def chunk_text_overlap(text, chunk_size, overlap):
    """Chunk text into pieces of approximately chunk_size characters with a given overlap. Includes function to split at full stops"""
    chunks = []
    start = 0
    text = text.strip()
    # Use larger chunk size to avoid splitting answers
    if len(text) <= chunk_size:
        return [text]
    while start < len(text):
        # Find end position for this chunk
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to find a sentence end (period followed by space) near the chunk_size
        sentence_end = text.rfind('. ', start + chunk_size - 100, start + chunk_size + 100)
        if sentence_end != -1:
            end = sentence_end + 1  # Include the period
        else:
            # If no sentence end, find a space
            space = text.rfind(' ', start + chunk_size - 100, start + chunk_size + 100)
            if space != -1:
                end = space      
        chunks.append(text[start:end])
        # Move start position for next chunk, accounting for overlap
        start = end - overlap
    return chunks

def generate_embeddings(chunks, model_name):
    """Generate embeddings for each text chunk using the specified OpenAI embedding model. Returns a list of embeddings."""
    embeddings = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        if not chunk.strip():
            print(f"WARNING: Empty chunk at index {i}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 1536)  # Using 1536 for text-embedding-3-large dimension. 3-large is usually 3072, we can specify. 
            continue
        try:
            response = openai.embeddings.create(
                model=model_name,
                input=[chunk]
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error embedding chunk {i+1}/{total_chunks}: {e}")
            # Use dimension appropriate for the model
            dimension = 3072 if "large" in model_name else 1536
            embeddings.append([0.0] * dimension)
    return embeddings

def build_library_index(library_dir, chunk_size, overlap, embedding_model):
    """Loads PDFs from the given library directory, chunks each PDF with the given parameters,
    computes embeddings with the chosen embedding model, and builds a FAISS index."""
    pdf_texts = load_pdfs_from_directory(library_dir)
    if not pdf_texts:
        raise ValueError(f"No valid PDF texts found in {library_dir}")
    all_chunks = []
    metadata = []  # e.g., (source filename, chunk number, chunk_text)
    print("Chunking documents...")
    for filename, text in pdf_texts.items(): #processes and chunks text
        chunks = chunk_text_overlap(text, chunk_size=chunk_size, overlap=overlap)
        chunk_start_idx = len(all_chunks)
        all_chunks.extend(chunks)
        for i, chunk in enumerate(chunks): # Store more metadata including the actual text
            metadata.append({
                "filename": filename, 
                "chunk_index": i + chunk_start_idx,
                "text": chunk   
            })   
    print(f"Total chunks created: {len(all_chunks)}")
    if len(all_chunks) == 0:
        raise ValueError("No chunks were created from the PDFs")
    
    # Generate embeddings
    embed_list = generate_embeddings(all_chunks, embedding_model)
    # Convert embeddings to a numpy float32 array (storage format)
    embeddings = np.array(embed_list).astype('float32')
    # Validate embeddings
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings were generated")
    if np.isnan(embeddings).any():
        print("WARNING: NaN values found in embeddings")
        # Replace NaN values with zeros
        embeddings = np.nan_to_num(embeddings)
    
    # Build FAISS index (using cosine similarity instead of L2 for better semantic matching)
    dimension = embeddings.shape[1]
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    # Use an inner product index (equivalent to cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, all_chunks, metadata

def similarity_score(text1, text2):
    """Calculate fuzzy matching similarity between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def calculate_f1(text1, text2):
    """Calculate F1 score between two texts based on token overlap"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    true_positives = len(tokens1.intersection(tokens2))
    precision = true_positives / len(tokens2) if tokens2 else 0
    recall = true_positives / len(tokens1) if tokens1 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def binary_relevance(chunk, answer, threshold=0.5):
    """Determine if a chunk is relevant to the answer based on a similarity threshold"""
    # Normalize texts
    chunk_norm = re.sub(r'\s+', ' ', chunk).lower()
    answer_norm = re.sub(r'\s+', ' ', str(answer)).lower()
    
    # Direct substring match
    if answer_norm in chunk_norm or chunk_norm in answer_norm:
        return 1
    
    # Fuzzy matching
    sim = similarity_score(answer_norm, chunk_norm)
    if sim >= threshold:
        return 1
    
    # Word overlap ratio
    answer_words = set(answer_norm.split())
    chunk_words = set(chunk_norm.split())
    if answer_words and len(answer_words.intersection(chunk_words)) / len(answer_words) >= threshold:
        return 1
    
    return 0

def evaluate_multi_k(query_results, correct_answer, max_k=20):
    """
    Evaluate the retrieval results at multiple k values (1, 3, 5, 10, 20)
    Returns metrics at each k value
    """
    results = {}
    
    # Define the k values we want to evaluate
    k_values_to_test = [1, 3, 5, 10, min(max_k, len(query_results))]
    k_values_to_test = sorted(list(set(k_values_to_test)))  # Remove duplicates and sort
    
    for k in k_values_to_test:
        top_k_chunks = query_results[:k]
        
        # Check if the answer is in any of the top-k chunks
        relevance_scores = [binary_relevance(chunk["text"], correct_answer) for chunk in top_k_chunks]
        found_in_top_k = any(relevance_scores)
        
        # Calculate the best match score within top-k
        best_match_score = 0
        best_f1_score = 0
        best_chunk = ""
        
        for chunk in top_k_chunks:
            chunk_text = chunk["text"]
            sim_score = similarity_score(correct_answer, chunk_text)
            f1 = calculate_f1(correct_answer, chunk_text)
            
            if sim_score > best_match_score:
                best_match_score = sim_score
                best_chunk = chunk_text
            
            if f1 > best_f1_score:
                best_f1_score = f1
        
        # For precision and recall at k
        true_positives = sum(relevance_scores)
        precision_at_k = true_positives / k if k > 0 else 0
        recall_at_k = 1.0 if true_positives > 0 else 0  # Assuming 1 correct answer
        
        # F1 at k
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0
        
        results[f"top_{k}"] = {
            "found": found_in_top_k,
            "best_similarity": best_match_score,
            "best_f1": best_f1_score,
            "precision": precision_at_k,
            "recall": recall_at_k,
            "f1": f1_at_k,
            "best_chunk": best_chunk
        }
    
    return results

def run_experiment(library_dir, queries_csv, chunk_size, overlap, embedding_model, k, results_dir):
    """Runs one experiment with the given parameters with improved multi top-k and F1 scoring."""
    print(f"\nRunning experiment : chunk_size={chunk_size}, overlap={overlap}, "
          f"embedding_model={embedding_model}, k={k}")
          
    # Build library index
    index, chunks, metadata = build_library_index(library_dir, chunk_size, overlap, embedding_model)
    
    # Load queries CSV. Assumes CSV has columns "query" and "answer".
    try:
        df = pd.read_csv(queries_csv, encoding='windows-1252')
        if "query" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV file must contain 'query' and 'answer' columns")
    except Exception as e:
        print(f"Error loading queries CSV: {e}")
        return 0
    
    # Track multiple metrics
    metrics = {
        "similarity_scores": [],
        "f1_scores": [],
        "top_1_accuracy": [],
        "top_3_accuracy": [],
        "top_5_accuracy": [],
        "top_k_accuracy": []  # For the specified k value
    }
    
    # For detailed result analysis
    all_query_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
        query = row["query"]
        correct_answer = str(row["answer"]).strip()
        if not correct_answer:
            print(f"WARNING: Empty answer for query {idx}: '{query}'")
            continue
            
        try:
            # Embed the query
            response = openai.embeddings.create(
                model=embedding_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
        except Exception as e:
            print(f"Error embedding query {idx}: {e}")
            continue

        # Use a larger k for evaluation (we'll evaluate at different k values)
        max_k = max(k, 20)  # Use at least 20 for thorough evaluation
        scores, indices_result = index.search(query_embedding, max_k)
        
        # Prepare retrieved chunks with their metadata
        retrieved_chunks = []
        for i, idx in enumerate(indices_result[0]):
            if idx < len(chunks):
                retrieved_chunks.append({
                    "chunk_index": idx,
                    "text": chunks[idx],
                    "score": float(scores[0][i]),
                    "metadata": metadata[idx] if idx < len(metadata) else {}
                })
        
        # Multi-k evaluation
        evaluation_results = evaluate_multi_k(retrieved_chunks, correct_answer, max_k)
        
        # Store metrics for this query
        metrics["similarity_scores"].append(evaluation_results[f"top_{k}"]["best_similarity"])
        metrics["f1_scores"].append(evaluation_results[f"top_{k}"]["best_f1"])
        metrics["top_1_accuracy"].append(1 if evaluation_results.get("top_1", {}).get("found", False) else 0)
        metrics["top_3_accuracy"].append(1 if evaluation_results.get("top_3", {}).get("found", False) else 0)
        metrics["top_5_accuracy"].append(1 if evaluation_results.get("top_5", {}).get("found", False) else 0)
        metrics["top_k_accuracy"].append(1 if evaluation_results[f"top_{k}"]["found"] else 0)
        
        # Store detailed results for this query
        query_result = {
            "query_id": idx,
            "query": query,
            "answer": correct_answer,
            "evaluation": evaluation_results,
            "top_chunk": retrieved_chunks[0]["text"] if retrieved_chunks else "",
            "top_score": retrieved_chunks[0]["score"] if retrieved_chunks else 0,
        }
        all_query_results.append(query_result)

    # Calculate aggregate metrics
    results_summary = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": embedding_model,
        "k": k,
        "mean_similarity": np.mean(metrics["similarity_scores"]) if metrics["similarity_scores"] else 0,
        "mean_f1": np.mean(metrics["f1_scores"]) if metrics["f1_scores"] else 0,
        "top_1_accuracy": np.mean(metrics["top_1_accuracy"]) if metrics["top_1_accuracy"] else 0,
        "top_3_accuracy": np.mean(metrics["top_3_accuracy"]) if metrics["top_3_accuracy"] else 0,
        "top_5_accuracy": np.mean(metrics["top_5_accuracy"]) if metrics["top_5_accuracy"] else 0,
        "top_k_accuracy": np.mean(metrics["top_k_accuracy"]) if metrics["top_k_accuracy"] else 0,
    }
    
    print(f"Results: Mean Similarity={results_summary['mean_similarity']:.3f}, "
          f"Mean F1={results_summary['mean_f1']:.3f}, "
          f"Top-{k} Accuracy={results_summary['top_k_accuracy']:.3f}")
    
    # Save detailed results for analysis
    results_df = pd.DataFrame([{
        "query": r["query"],
        "answer": r["answer"],
        "top_chunk": r["top_chunk"],
        "top_score": r["top_score"],
        "similarity_score": r["evaluation"][f"top_{k}"]["best_similarity"],
        "f1_score": r["evaluation"][f"top_{k}"]["best_f1"],
        "found_in_top_k": r["evaluation"][f"top_{k}"]["found"],
        "found_in_top_1": r["evaluation"].get("top_1", {}).get("found", False),
        "found_in_top_3": r["evaluation"].get("top_3", {}).get("found", False),
        "found_in_top_5": r["evaluation"].get("top_5", {}).get("found", False),
    } for r in all_query_results])
    
    results_file = os.path.join(results_dir, f"Results_cs{chunk_size}_ol{overlap}_k{k}.csv")
    results_df.to_csv(results_file, index=False)
    
    # Return the combined success metric (average of similarity and F1)
    return results_summary

def main():
    # Define your paths
    library_dir = "Experiment Test Papers"
    queries_csv = "Experiment Questions gemini.csv"
    
    # Create a timestamp-based directory for this experimental run
    timestamp = datetime.datetime.now().strftime("%m_%d_%H%M")
    base_results_dir = "Experiment_Results"
    run_dir = os.path.join(base_results_dir, f"Run {timestamp}")
    
    # Create directories if they don't exist
    if not os.path.exists(base_results_dir):
        os.mkdir(base_results_dir)
    os.mkdir(run_dir)
    
    print(f"Results will be saved in: {run_dir}")
    
    # Define parameter ranges for experiments
    chunk_sizes = [475, 500, 525]  
    overlaps = [125, 150, 175] 
    embedding_models = ["text-embedding-3-small"]
    k_values = [10]  # Make sure k is at least 1
    
    # List to hold results
    experiment_results = []
    
    # Iterate over all parameter combinations
    experiment_counter = 1
    total_experiments = len(list(itertools.product(chunk_sizes, overlaps, embedding_models, k_values)))
    
    for chunk_size, overlap, model_name, k in itertools.product(chunk_sizes, overlaps, embedding_models, k_values):
        print(f"\n=== Running experiment {experiment_counter} of {total_experiments} ===")
        try:
            results = run_experiment(library_dir, queries_csv, chunk_size, overlap, model_name, k, run_dir)
            experiment_results.append(results)
        except Exception as e:
            print(f"Experiment failed for parameters "
                  f"chunk_size={chunk_size}, overlap={overlap}, model={model_name}, k={k}: {e}")
        experiment_counter += 1
    
    # Convert results to DataFrame and print/save
    results_df = pd.DataFrame(experiment_results)
    print("\nExperiment Results:")
    print(results_df)
    
    # Save to CSV in the run directory
    summary_file = os.path.join(run_dir, "experiment_results_summary.csv")
    results_df.to_csv(summary_file, index=False)
    
    # Show best performing configuration based on combined metric
    if not results_df.empty:
        # Add a combined metric column (average of F1 and top-k accuracy)
        results_df["combined_score"] = (results_df["mean_f1"] + results_df["top_k_accuracy"]) / 2
        
        best_row = results_df.loc[results_df['combined_score'].idxmax()]
        print("\nBest configuration:")
        print(f"Chunk size: {best_row['chunk_size']}")
        print(f"Overlap: {best_row['overlap']}")
        print(f"Model: {best_row['embedding_model']}")
        print(f"k: {best_row['k']}")
        print(f"Mean Similarity: {best_row['mean_similarity']:.3f}")
        print(f"Mean F1 Score: {best_row['mean_f1']:.3f}")
        print(f"Top-1 Accuracy: {best_row['top_1_accuracy']:.3f}")
        print(f"Top-3 Accuracy: {best_row['top_3_accuracy']:.3f}")
        print(f"Top-5 Accuracy: {best_row['top_5_accuracy']:.3f}")
        print(f"Top-{int(best_row['k'])} Accuracy: {best_row['top_k_accuracy']:.3f}")
        print(f"Combined Score: {best_row['combined_score']:.3f}")
        
        # Save best results in the run directory
        best_config_file = os.path.join(run_dir, f"Best_results_cs{int(best_row['chunk_size'])}_ol{int(best_row['overlap'])}_k{int(best_row['k'])}.csv")
        results_df.to_csv(best_config_file, index=False)

if __name__ == "__main__":
    main()