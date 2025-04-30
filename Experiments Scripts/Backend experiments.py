from flask import Flask
import numpy as np
import faiss
import openai
import os
import glob
import time
import hashlib
import pandas as pd
from functools import lru_cache
from tqdm import tqdm

# Initialize Flask app but don't run it
app = Flask(__name__, template_folder='frontend')

gpt_model = "gpt-4o"
evaluation_model = "gpt-4o"  # Model to evaluate responses

# Test queries to run with query type metadata
TEST_QUERIES = [
    {"query": "What is HIV?", "type": "fact"},
    {"query": "What exactly are antimicrobial resistance genes and why are they a problem?", "type": "fact"},
    {"query": "Are antibiotics used a lot in farming animals like cows and chickens?", "type": "fact"},
    {"query": "How do scientists use sewage to find out about antibiotic resistance?", "type": "methods/results"},
    {"query": "What does it mean when researchers 3D-print bacteria — and why would they do that?", "type": "methods/results"},
    {"query": "How do they figure out if animals, people, and the environment are sharing resistant bacteria?", "type": "methods/results"},
    {"query": "What does ‘One Health’ mean when people talk about fighting antibiotic resistance?", "type": "concepts"},
    {"query": "How can bacteria in soil affect human health?", "type": "concepts"},
    {"query": "Why is antibiotic resistance sometimes compared to climate change?", "type": "concepts"},
    {"query": "Are scientists close to finding new ways to stop bacteria from becoming resistant?", "type": "state of research"},
    {"query": "Is it true that the COVID pandemic made antibiotic resistance worse? How?", "type": "state of research"},
    {"query": "What are countries doing to keep track of antibiotic resistance?", "type": "state of research"},
    {"query": "If so many bacteria are already resistant, is it too late to fix the problem?", "type": "challenges with solutions"},
    {"query": "Can better sanitation really help stop antibiotic resistance? How would that work?", "type": "challenges with solutions"},
    {"query": "How could using machine learning computers actually help doctors fight infections better?", "type": "challenges with solutions"},
    ]

# System prompts to test
SYSTEM_PROMPTS = {
    '3': "You are an informed, approachable assistant for Edinburgh Infectious Diseases, tasked with educating the public about EID’s research.\
Base all answers solely on the provided context, synthesizing information where appropriate into clear, story-like explanations.\
Use plain language and avoid scientific jargon to ensure accessibility.\
Reference specific studies (e.g., Study title, 2021) to give credibility to your points.\
Keep your tone professional yet genuinely enthusiastic about the research.\
If information is missing or uncertain, ask for clarification.",
    }

# Best parameters determined from previous testing 
BEST_PARAMETERS = {
    'chunk_count': 10,
    'vector_weight': 0.7,
    'keyword_weight': 0.3
}

# Cache for embeddings and responses (TTL = 1 hour)
CACHE_TTL = 3600
EMBEDDING_CACHE = {}
RESPONSE_CACHE = {}

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Raise an error if the API key is not found
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Get embedding directory from environment variable or use default
embedding_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")

# Find all available FAISS index files and text chunk files
index_files = glob.glob(os.path.join(embedding_dir, "*.index")) + glob.glob(os.path.join(embedding_dir, "*.faiss"))
text_files = glob.glob(os.path.join(embedding_dir, "*.txt"))

# Create dictionaries to store loaded indices and text chunks
loaded_indices = {}
loaded_text_chunks = {}

# Load all available FAISS indices and convert to HNSW where appropriate
for index_path in index_files:
    index_name = os.path.splitext(os.path.basename(index_path))[0]
    try:
        # Load the original index
        original_index = faiss.read_index(index_path)
        
        # Check if it's a flat index that could benefit from HNSW
        if isinstance(original_index, faiss.IndexFlat) or isinstance(original_index, faiss.IndexFlatL2):
            dimension = original_index.d
            vector_count = original_index.ntotal
            
            # Only convert if there are enough vectors to benefit from it
            if vector_count > 1000:
                
                # Create HNSW index
                hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
                hnsw_index.hnsw.efConstruction = 40  # Higher build quality (more accurate, slower build)
                hnsw_index.hnsw.efSearch = 16  # Search quality parameter (higher = more accurate but slower)
                
                # Add vectors from original index
                if vector_count > 0:
                    vectors = np.vstack([original_index.reconstruct(i) for i in range(vector_count)])
                    hnsw_index.add(vectors)
                
                loaded_indices[index_name] = hnsw_index
            else:
                # Keep the original for small indices
                loaded_indices[index_name] = original_index
                print(f"Keeping original index for {index_name} (only {vector_count} vectors)")
        else:
            # Keep the original if it's already not a flat index
            loaded_indices[index_name] = original_index
            print(f"Using existing index for {index_name}")
            
    except Exception as e:
        print(f"Error loading/optimizing FAISS index {index_name}: {str(e)}")

# Load all available text chunk files
for text_path in text_files:
    text_name = os.path.basename(text_path).split('.')[0]
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read().split("\n----\n")
            loaded_text_chunks[text_name] = text_content
    except Exception as e:
        print(f"Error loading text chunks {text_name}: {str(e)}")

# Helper functions for vector operations
@lru_cache(maxsize=500)
def get_embedding(text):
    """Get embedding with caching using LRU cache decorator"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    
    # Check cache first
    if cache_key in EMBEDDING_CACHE:
        cache_entry = EMBEDDING_CACHE[cache_key]
        if time.time() - cache_entry['timestamp'] < CACHE_TTL:
            return cache_entry['embedding']
    
    # Get embedding from OpenAI
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    embedding = np.array(response.data[0].embedding)
    
    # Store in cache
    EMBEDDING_CACHE[cache_key] = {
        'embedding': embedding,
        'timestamp': time.time()
    }
    
    return embedding

def deduplicate_chunks(chunks, threshold=0.5):
    """Remove near-duplicate chunks to avoid redundancy"""
    if not chunks:
        return []
        
    unique_chunks = [chunks[0]]
    
    for chunk in chunks[1:]:
        is_duplicate = False
        chunk_words = set(chunk.lower().split())
        
        for unique_chunk in unique_chunks:
            unique_words = set(unique_chunk.lower().split())
            
            # Calculate Jaccard similarity
            overlap = len(chunk_words.intersection(unique_words))
            union = len(chunk_words.union(unique_words))
            
            if union > 0 and overlap / union > threshold:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_chunks.append(chunk)
            
    return unique_chunks

def rerank_results(query, chunks, distances, vector_weight=0.7, keyword_weight=0.3):
    """Rerank chunks using a hybrid scoring approach with configurable weights"""
    query_terms = set(query.lower().split())
    reranked = []
    
    for i, chunk in enumerate(chunks):
        # Vector similarity score (convert distance to similarity)
        # FAISS L2 distance: smaller is better, convert to similarity
        vector_score = 1.0 / (1.0 + distances[i])
        
        # Keyword match score
        chunk_terms = set(chunk.lower().split())
        keyword_score = len(query_terms.intersection(chunk_terms)) / max(1, len(query_terms))
        
        # Calculate combined score using provided weights
        combined_score = vector_weight * vector_score + keyword_weight * keyword_score
        
        reranked.append((chunk, combined_score))
    
    # Sort by score (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    # Return reranked chunks
    return [item[0] for item in reranked]

def process_query(query, system_prompt_id, index_name, text_chunks_name, 
                  chunk_count=BEST_PARAMETERS['chunk_count'], 
                  vector_weight=BEST_PARAMETERS['vector_weight'], 
                  keyword_weight=BEST_PARAMETERS['keyword_weight']):
    """Process a single query with specified parameters"""
    try:
        # Get the system prompt
        system_prompt = SYSTEM_PROMPTS[system_prompt_id]
        
        # Get the selected index and text chunks
        embeddings = loaded_indices[index_name]
        text_chunks = loaded_text_chunks[text_chunks_name]
        
        # Create a cache key for this parameter combination
        cache_key = f"{query}_{system_prompt_id}_{index_name}_{text_chunks_name}_{chunk_count}_{vector_weight}_{keyword_weight}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check response cache
        if cache_hash in RESPONSE_CACHE:
            cache_entry = RESPONSE_CACHE[cache_hash]
            if time.time() - cache_entry['timestamp'] < CACHE_TTL:
                return cache_entry['response']
        
        # Step 1: Embed the user's query
        query_embedding = get_embedding(query).reshape(1, -1)
        
        # Step 2: Search for relevant text chunks using FAISS
        k = min(30, len(text_chunks))  # Retrieve more candidates for reranking
        distances, indices_result = embeddings.search(query_embedding, k)

        # Check if any relevant text chunks were found
        if len(indices_result[0]) == 0:
            return "No relevant text found"

        # Gather the most relevant text chunks and their sources
        relevant_texts = []
        sources = []
        chunk_distances = []
        
        for i, idx in enumerate(indices_result[0]):
            if idx < len(text_chunks):
                relevant_texts.append(text_chunks[idx])
                source = text_chunks[idx].split('\n')[0].replace('Source: ', '')
                sources.append(source)
                chunk_distances.append(distances[0][i])

        # Rerank the results with provided weights
        reranked_texts = rerank_results(query, relevant_texts, chunk_distances, 
                                       vector_weight, keyword_weight)
        
        # Deduplicate to remove very similar chunks
        unique_texts = deduplicate_chunks(reranked_texts)
        
        # Limit context size based on specified chunk count
        top_chunks = unique_texts[:chunk_count]
        
        # Combine the relevant texts into a single context
        context = "\n\n".join(top_chunks)
        source_info = "\n".join([f"Source: {source}" for source in sources[:5]])

        # Prepare the conversation for GPT-4o
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context: {context}\n{source_info}"},
            {"role": "user", "content": query},
        ]
            
        # Call GPT-4o with the context
        response = openai.chat.completions.create(
            model=gpt_model,
            messages=messages
        )

        # Extract the response
        gpt_response = response.choices[0].message.content
        
        # Store in cache
        RESPONSE_CACHE[cache_hash] = {
            'response': gpt_response,
            'timestamp': time.time()
        }

        return gpt_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def evaluate_response(query, response, context_chunks):
    """Evaluate the quality of a response using GPT"""
    try:
        # Creating a detailed evaluation prompt
        evaluation_prompt = f"""
        Please evaluate the following AI response to a user query based on these criteria. The responses are replies from a
        chatbot and should reflect the nature of the medium:
        
        1. Accuracy (0-100): Does the response contain factually correct information based on the available context?
        2. Relevance (0-100): How well does the response address the specific user query?
        3. Entailment (0-100): Is all information in the response supported by the context provided?
        4. Clarity (0-100): Is the response well-structured, easy to understand (for someone without scientific knowledge), concise, and appropriately formatted?\
            Bear in mind this is a chatbot so messages should be snappy, conversational, and brief when possible.
        
        User Query: "{query}"
        
        AI Response: 
        {response}
        
        Context Summary (from which the AI generated its response):
        {context_chunks[:3000]}... [truncated]
        
        Provide a numerical score (0-100) for each criterion, along with a brief justification.
        Then calculate a total score as the average of the four individual scores (also out of 100).
        Format your response as follows:
        
        Accuracy: [score] - [justification]
        
        Relevance: [score] - [justification]
        
        Entailment: [score] - [justification]
        
        Clarity: [score] - [justification]
        
        Total Score: [average of the four scores]
        
        Overall Assessment: [2-3 sentence summary]
        All context provided, and studies mentioned in responses, will have been written and conducted by researchers working in Edinburgh"""
        
        messages = [
            {"role": "system", "content": "You are an expert evaluator of AI-generated responses. Your task is to objectively evaluate responses \
             based on accuracy, relevance, entailment, and clarity. Always provide scores between 0-100 for each criterion. The total score should \
             be the average of all four scores."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        # Call evaluation model
        evaluation_response = openai.chat.completions.create(
            model=evaluation_model,
            messages=messages
        )
        
        return evaluation_response.choices[0].message.content
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Evaluation Error: {str(e)}"

def extract_scores(evaluation_text):
    """Extract scores from evaluation text including individual metrics"""
    try:
        scores = {}
        metrics = ['Accuracy', 'Relevance', 'Entailment', 'Clarity', 'Total Score']
        
        for metric in metrics:
            for line in evaluation_text.split('\n'):
                if f"{metric}:" in line:
                    # Extract the number
                    score_text = line.split(f"{metric}:")[1].strip()
                    try:
                        # Extract just the numeric part
                        score_value = score_text.split()[0].replace('/', '').strip()
                        score = float(score_value)
                        # Ensure score is between 0-100
                        score = max(0, min(100, score))
                        scores[metric] = score
                        break
                    except (ValueError, IndexError):
                        # If extraction fails, set to 0
                        scores[metric] = 0
                        break
        
        # If total score wasn't found or is invalid, calculate it from individual metrics
        if 'Total Score' not in scores or scores['Total Score'] < 0 or scores['Total Score'] > 100:
            individual_metrics = ['Accuracy', 'Relevance', 'Entailment', 'Clarity']
            valid_metrics = [scores[m] for m in individual_metrics if m in scores]
            if valid_metrics:
                scores['Total Score'] = sum(valid_metrics) / len(valid_metrics)
            else:
                scores['Total Score'] = 0
                
        return scores
    except Exception as e:
        print(f"Error extracting scores: {str(e)}")
        # Return default values
        return {
            'Accuracy': 0,
            'Relevance': 0,
            'Entailment': 0,
            'Clarity': 0,
            'Total Score': 0
        }

def analyze_performance_by_query_type(results_df):
    """Analyzes how each system prompt performs against different query types
    Args:
        results_df: DataFrame containing test results
    Returns:
        DataFrame showing performance by query type for each system prompt"""
    # Create a pivoted dataframe to show performance by query type for each prompt
    pivot_df = results_df.pivot_table(
        index='query_type',
        columns='prompt_id',
        values='score',
        aggfunc='mean'
    )
    
    
    # Add count of each query type
    query_type_counts = results_df['query_type'].value_counts()
    pivot_df['count'] = pivot_df.index.map(lambda x: query_type_counts.get(x, 0))
    
    # Add standard deviation to see consistency
    std_df = results_df.pivot_table(
        index='query_type', 
        columns='prompt_id',
        values='score',
        aggfunc='std'
    )
    
    for col in std_df.columns:
        pivot_df[f'{col}_std'] = std_df[col]
    
    # Calculate best prompt for each query type
    pivot_df['best_prompt'] = pivot_df.iloc[:, :-1].idxmax(axis=1)
    
    # Calculate overall rank of each system prompt
    prompt_columns = [col for col in pivot_df.columns if isinstance(col, str) and col in SYSTEM_PROMPTS.keys()]
    prompt_means = pivot_df[prompt_columns].mean()
    prompt_ranks = prompt_means.rank(ascending=False)
    
    # Print summary
    print("\n=== PERFORMANCE BY QUERY TYPE ===")
    print("Overall prompt rankings:")
    for prompt_id in prompt_means.index:
        print(f"Prompt {prompt_id}: Avg Score = {prompt_means[prompt_id]:.2f}, Rank = {prompt_ranks[prompt_id]:.0f}")
    
    print("\nBest prompt for each query type:")
    for query_type in pivot_df.index:
        best_prompt = pivot_df.loc[query_type, 'best_prompt']
        score = pivot_df.loc[query_type, best_prompt]
        print(f"{query_type}: Prompt {best_prompt} (Score: {score:.2f})")
    
    return pivot_df

def run_automated_tests():
    """Run automated testing across all system prompts"""
    if not loaded_indices or not loaded_text_chunks:
        print("No indices or text chunks available. Please check your embeddings directory.")
        return
    
    # Select first available index and text chunks as default
    default_index = list(loaded_indices.keys())[0]
    default_text_chunks = list(loaded_text_chunks.keys())[0]
    
    # Create results directory if it doesn't exist
    results_dir = ""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Prepare results storage
    results = []
    
    # Test all system prompts
    print("\n--- Testing different system prompts ---")
    for prompt_id, prompt_text in SYSTEM_PROMPTS.items():
        print(f"\nTesting System Prompt {prompt_id}:")
        prompt_scores = []
        
        for query_item in tqdm(TEST_QUERIES):
            query = query_item["query"]
            query_type = query_item["type"]
            
            response = process_query(
                query=query,
                system_prompt_id=prompt_id,
                index_name=default_index,
                text_chunks_name=default_text_chunks
            )
            
            # Get top context chunks for evaluation
            query_embedding = get_embedding(query).reshape(1, -1)
            embeddings = loaded_indices[default_index]
            text_chunks = loaded_text_chunks[default_text_chunks]
            k = min(BEST_PARAMETERS['chunk_count'], len(text_chunks))
            _, indices_result = embeddings.search(query_embedding, k)
            context_chunks = "\n\n".join([text_chunks[idx] for idx in indices_result[0] if idx < len(text_chunks)])
            
            # Print query and response
            print(f"\nQuery ({query_type}): {query}")
            print(f"Response:\n{response}\n")
            
            # Evaluate response
            evaluation = evaluate_response(query, response, context_chunks)
            scores = extract_scores(evaluation)
            total_score = scores['Total Score']
            prompt_scores.append(total_score)
            
            print(f"Evaluation:\n{evaluation}\n")
            print(f"Score: {total_score}/100\n")
            
            # Store result
            results.append({
                'query': query,
                'query_type': query_type,
                'prompt_id': prompt_id,
                'chunk_count': BEST_PARAMETERS['chunk_count'],
                'vector_weight': BEST_PARAMETERS['vector_weight'],
                'keyword_weight': BEST_PARAMETERS['keyword_weight'],
                'response': response,
                'evaluation': evaluation,
                'accuracy': scores['Accuracy'],
                'relevance': scores['Relevance'],
                'entailment': scores['Entailment'],
                'clarity': scores['Clarity'],
                'score': total_score
            })
        
        # Print average score for this prompt
        avg_score = sum(prompt_scores) / len(prompt_scores)
        print(f"Average score for System Prompt {prompt_id}: {avg_score:.2f}/100")
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Save results to CSV in the dedicated directory
    timestamp = time.strftime("%d-%H%M")
    csv_path = os.path.join(results_dir, f'Test_Results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nTest results saved to {csv_path}")
    
    # Perform analysis by query type
    query_type_analysis = analyze_performance_by_query_type(df)
    
    # Save the query type analysis
    analysis_path = os.path.join(results_dir, f'Query_Type_Analysis_{timestamp}.csv')
    query_type_analysis.to_csv(analysis_path)
    print(f"Query type analysis saved to {analysis_path}")
    
    # Summarize best parameters
    print("\n=== SUMMARY OF BEST PARAMETERS ===")
    
    # Best system prompt
    prompt_avg_scores = df.groupby('prompt_id')['score'].mean()
    best_prompt_id = prompt_avg_scores.idxmax()
    print(f"Best System Prompt: {best_prompt_id} (Avg Score: {prompt_avg_scores[best_prompt_id]:.2f})")
    print(f"Prompt text: {SYSTEM_PROMPTS[best_prompt_id][:100]}...")
    
    # Show metrics performance for best prompt
    best_prompt_metrics = df[df['prompt_id'] == best_prompt_id].mean()
    print("\nMetrics for best prompt:")
    print(f"Accuracy: {best_prompt_metrics['accuracy']:.2f}")
    print(f"Relevance: {best_prompt_metrics['relevance']:.2f}")
    print(f"Entailment: {best_prompt_metrics['entailment']:.2f}")
    print(f"Clarity: {best_prompt_metrics['clarity']:.2f}")
    
    # Note about fixed parameter values
    print("\nUsing fixed parameters:")
    print(f"Chunk count: {BEST_PARAMETERS['chunk_count']}")
    print(f"Vector weight: {BEST_PARAMETERS['vector_weight']}")
    print(f"Keyword weight: {BEST_PARAMETERS['keyword_weight']}")
    
    return df

if __name__ == "__main__":
    # Run the automated tests
    results_df = run_automated_tests()