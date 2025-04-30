from flask import Flask, render_template, request, jsonify
import numpy as np
import faiss
import openai
from openai import OpenAIError
import os
import glob
import time
import hashlib
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__, template_folder='frontend')

gpt_model="gpt-4o"

# Display names for the prompt styles
PROMPT_NAMES = {
    '1': "Basic",
    '2': "Advanced"
}

# System prompts dictionary
SYSTEM_PROMPTS = {
    '1': "You are an informed, approachable assistant for Edinburgh Infectious Diseases, tasked with educating the public about EID’s research.\
    Base all answers solely on the provided context, synthesizing information where appropriate into clear, story-like explanations.\
    Use plain language and avoid scientific jargon to ensure accessibility.\
    Reference specific studies (Study title, 2021) to give credibility to your points.\
    Keep your tone professional yet genuinely enthusiastic about the research.\
    If information is missing or uncertain, ask for clarification.",
    
    '2': "You are an articulate, well-informed assistant for Edinburgh Infectious Diseases. \
    You speak to scientists familiar with infectious disease research who want quick, clear insights rooted in local context.\
    Draw all information from the provided research context. \
    When appropriate, synthesize multiple findings to give a narrative-style answer — but keep responses digestible and structured.\
    Always cite study titles (e.g., Study title, 2021) to back up your points. Ask for clarification if needed.\
    Stay professional, but don't be afraid to show enthusiasm for the science being done — it's impressive work.",
}

# Cache for embeddings and responses (TTL = 1 hour)
CACHE_TTL = 3600
EMBEDDING_CACHE = {}
RESPONSE_CACHE = {}

# Debug info, prints path to python backend location
print(f"Current working directory: {os.getcwd()}")

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Raise an error if the API key is not found
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Get embedding directory from environment variable or use default
embedding_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
print(f"Embeddings directory: {embedding_dir}")

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

# Print available datasets
print(f"Available indices: {list(loaded_indices.keys())}")
print(f"Available text chunks: {list(loaded_text_chunks.keys())}")

# Helper function for vector operations
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

# Route to render the frontend
@app.route('/')
def index():
    # Pass the available datasets to the frontend
    return render_template('index.html', 
                          indices=list(loaded_indices.keys()), 
                          text_chunks=list(loaded_text_chunks.keys()),
                          prompt_names=PROMPT_NAMES)

# API endpoint to get available datasets
@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    return jsonify({
        "indices": list(loaded_indices.keys()),
        "text_chunks": list(loaded_text_chunks.keys()),
        "prompt_styles": PROMPT_NAMES
    })

# Route to handle chat API requests
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Parse JSON data from the POST request
        data = request.json
        user_query = data.get('message', '').strip()
        prompt_style = data.get('file')  # Now this is just the prompt style ID
        history = data.get('history', [])
        
        # Get dataset names from request
        index_name = data.get('index_name')
        text_chunks_name = data.get('text_chunks_name')
        
        # If no specific datasets are provided, use the first available ones
        if not index_name and loaded_indices:
            index_name = list(loaded_indices.keys())[0]
        
        if not text_chunks_name and loaded_text_chunks:
            text_chunks_name = list(loaded_text_chunks.keys())[0]

        # Validate the user query and prompt style
        if not user_query or not prompt_style:
            return jsonify({"error": "Empty query or style not selected"}), 400

        # Make sure we have a valid prompt style
        if prompt_style not in SYSTEM_PROMPTS:
            return jsonify({"error": "Invalid style selected"}), 400
            
        # Check if the requested index and text chunks exist
        if index_name not in loaded_indices:
            return jsonify({"error": f"Index '{index_name}' not found"}), 404
            
        if text_chunks_name not in loaded_text_chunks:
            return jsonify({"error": f"Text chunks '{text_chunks_name}' not found"}), 404
        
        # Create a cache key for this exact query combination
        cache_key = f"{user_query}_{prompt_style}_{index_name}_{text_chunks_name}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Check response cache
        if cache_hash in RESPONSE_CACHE:
            cache_entry = RESPONSE_CACHE[cache_hash]
            if time.time() - cache_entry['timestamp'] < CACHE_TTL:
                print(f"Using cached response for query: {user_query}")
                return jsonify({
                    "response": cache_entry['response'],
                    "used_index": index_name,
                    "used_text_chunks": text_chunks_name,
                    "cached": True
                })
            
        # Get the selected index and text chunks
        embeddings = loaded_indices[index_name]
        text_chunks = loaded_text_chunks[text_chunks_name]

        # Step 1: Embed the user's query
        query_embedding = get_embedding(user_query).reshape(1, -1)
        
        # Step 2: Search for relevant text chunks using FAISS
        k = min(10, len(text_chunks))  # Retrieve top k chunks
        distances, indices_result = embeddings.search(query_embedding, k)

        # Check if any relevant text chunks were found
        if len(indices_result[0]) == 0:
            return jsonify({"error": "No relevant text found"}), 404

        # Gather the most relevant text chunks and their sources
        relevant_texts = []
        sources = []
        
        for i, idx in enumerate(indices_result[0]):
            if idx < len(text_chunks):
                relevant_texts.append(text_chunks[idx])
                source = text_chunks[idx].split('\n')[0].replace('Source: ', '')
                sources.append(source)

        # Combine the relevant texts into a single context
        context = "\n\n".join(relevant_texts[:5])  # Limit to top 5 chunks
        source_info = "\n".join([f"Source: {source}" for source in sources[:5]])

        # Step 3: Get the system prompt based on the selected style
        system_prompt = SYSTEM_PROMPTS[prompt_style]
        
        # Prepare the conversation for GPT-4o
        messages = history + [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context: {context}\n{source_info}"},
            {"role": "user", "content": user_query},
        ]
            
        # Call GPT-4o with the conversation history and context
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

        # Return the response to the frontend
        return jsonify({
            "response": gpt_response,
            "used_index": index_name,
            "used_text_chunks": text_chunks_name
        })

    except OpenAIError as e:
        return jsonify({"error": f"Error communicating with {gpt_model}: {str(e)}"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))