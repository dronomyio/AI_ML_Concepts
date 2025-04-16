#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import math
from collections import defaultdict

app = Flask(__name__, static_folder='static')

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fox is a wild animal from the canine family",
    "Dogs are popular pets known for their loyalty",
    "The lazy cat sleeps all day long",
    "Brown bears and red foxes are wild animals"
]

# SPLADE helper functions
def splade_activation(value):
    """Simulate SPLADE's ReLU(log(1+x)) activation."""
    return math.log(1 + max(0, value)) if value > 0 else 0

# Define a simplified term expansion dictionary to simulate BERT-like expansion
# In a real SPLADE model, these expansions would come from neural network activations
term_expansion = {
    'dog': {'canine': 0.7, 'pet': 0.6, 'animal': 0.5, 'loyal': 0.3},
    'fox': {'wild': 0.6, 'animal': 0.5, 'canine': 0.4, 'red': 0.3},
    'cat': {'pet': 0.6, 'animal': 0.5, 'lazy': 0.3, 'sleep': 0.3},
    'lazy': {'sleep': 0.5, 'slow': 0.4, 'rest': 0.3},
    'wild': {'animal': 0.7, 'nature': 0.5, 'forest': 0.4},
    'animal': {'wild': 0.5, 'pet': 0.4, 'species': 0.6, 'nature': 0.4},
    'quick': {'fast': 0.8, 'speed': 0.6, 'swift': 0.7},
    'brown': {'color': 0.6, 'red': 0.3, 'dark': 0.4},
    'jump': {'leap': 0.8, 'hop': 0.7, 'movement': 0.5},
    'popular': {'common': 0.7, 'liked': 0.8, 'famous': 0.6},
    'loyal': {'faithful': 0.8, 'devoted': 0.7, 'trust': 0.6},
}

def create_vectors(query=None):
    # Create a vocabulary from all documents
    all_tokens = []
    for doc in documents:
        tokens = [token.lower() for token in doc.split()]
        all_tokens.extend(tokens)
    
    # Add query tokens to vocabulary if provided
    if query:
        query_tokens = [token.lower() for token in query.split()]
        all_tokens.extend(query_tokens)
    
    # Create vocabulary (unique tokens)
    vocabulary = sorted(set(all_tokens))
    vocab_size = len(vocabulary)
    token_to_id = {token: idx for idx, token in enumerate(vocabulary)}
    
    # Create traditional sparse vectors (bag-of-words)
    traditional_sparse_vectors = []
    for doc in documents:
        vector = np.zeros(vocab_size)
        tokens = [token.lower() for token in doc.split()]
        for token in tokens:
            if token in token_to_id:
                vector[token_to_id[token]] += 1
        traditional_sparse_vectors.append(vector)
    
    # Create SPLADE-like sparse vectors with term expansion
    splade_vectors = []
    for doc in documents:
        # Initialize vector with zeros
        vector = np.zeros(vocab_size)
        
        # Tokenize document
        tokens = [token.lower() for token in doc.split()]
        
        # First pass: count terms (like traditional BOW)
        for token in tokens:
            if token in token_to_id:
                vector[token_to_id[token]] += 1
        
        # Second pass: expand terms using our simulated neural expansion
        expanded_vector = vector.copy()
        for token in tokens:
            if token in term_expansion:
                # Get expansion terms and weights
                for exp_term, weight in term_expansion[token].items():
                    if exp_term in token_to_id:
                        # Apply expansion with a weight
                        expanded_vector[token_to_id[exp_term]] += weight * 0.5
        
        # Apply SPLADE-like activation (ReLU + log)
        for i in range(len(expanded_vector)):
            expanded_vector[i] = splade_activation(expanded_vector[i])
        
        splade_vectors.append(expanded_vector)
    
    result = {
        'vocabulary': vocabulary,
        'token_to_id': token_to_id,
        'traditional_vectors': [v.tolist() for v in traditional_sparse_vectors],
        'splade_vectors': [v.tolist() for v in splade_vectors],
    }
    
    # If query is provided, add query vectors
    if query:
        # Create traditional query vector
        traditional_query_vector = np.zeros(vocab_size)
        query_tokens = [token.lower() for token in query.split()]
        for token in query_tokens:
            if token in token_to_id:
                traditional_query_vector[token_to_id[token]] += 1
        
        # Create SPLADE-like query vector with expansion
        splade_query_vector = np.zeros(vocab_size)
        for token in query_tokens:
            if token in token_to_id:
                splade_query_vector[token_to_id[token]] += 1
            
            # Apply expansion
            if token in term_expansion:
                for exp_term, weight in term_expansion[token].items():
                    if exp_term in token_to_id:
                        splade_query_vector[token_to_id[exp_term]] += weight * 0.5
        
        # Apply SPLADE activation to query vector
        for i in range(len(splade_query_vector)):
            splade_query_vector[i] = splade_activation(splade_query_vector[i])
        
        # Calculate similarity scores
        traditional_scores = np.array([np.dot(traditional_query_vector, np.array(doc_vec)) 
                                     for doc_vec in result['traditional_vectors']])
        splade_scores = np.array([np.dot(splade_query_vector, np.array(doc_vec)) 
                                for doc_vec in result['splade_vectors']])
        
        # Normalize scores for fair comparison
        if np.sum(traditional_scores) > 0:
            traditional_scores = traditional_scores / np.sum(traditional_scores)
        if np.sum(splade_scores) > 0:
            splade_scores = splade_scores / np.sum(splade_scores)
        
        # Rank the documents
        trad_ranked = np.argsort(-traditional_scores)
        splade_ranked = np.argsort(-splade_scores)
        
        traditional_ranking = [
            {
                'rank': i+1,
                'doc_id': int(idx)+1,
                'score': float(traditional_scores[idx]),
                'text': documents[idx]
            } for i, idx in enumerate(trad_ranked)
        ]
        
        splade_ranking = [
            {
                'rank': i+1,
                'doc_id': int(idx)+1,
                'score': float(splade_scores[idx]),
                'text': documents[idx]
            } for i, idx in enumerate(splade_ranked)
        ]
        
        result['query'] = query
        result['traditional_query_vector'] = traditional_query_vector.tolist()
        result['splade_query_vector'] = splade_query_vector.tolist()
        result['traditional_scores'] = traditional_scores.tolist()
        result['splade_scores'] = splade_scores.tolist()
        result['traditional_ranking'] = traditional_ranking
        result['splade_ranking'] = splade_ranking
    
    return result

def create_visualization(data):
    """Generate visualization of vectors and return as base64 image."""
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Get data
    traditional_vectors = np.array(data['traditional_vectors'])
    splade_vectors = np.array(data['splade_vectors'])
    vocabulary = data['vocabulary']
    
    # Plot traditional sparse vectors
    im1 = ax1.imshow(traditional_vectors, aspect='auto', cmap='Blues')
    plt.colorbar(im1, ax=ax1, label='Term Frequency')
    ax1.set_title('Traditional Sparse Vectors (Bag-of-Words)')
    ax1.set_xlabel('Terms (Vocabulary)')
    ax1.set_ylabel('Documents')
    ax1.set_xticks(range(len(vocabulary)))
    ax1.set_xticklabels(vocabulary, rotation=90)
    ax1.set_yticks(range(len(documents)))
    ax1.set_yticklabels([f"Doc {i+1}" for i in range(len(documents))])
    
    # Add text annotations for non-zero values
    for i in range(len(documents)):
        for j in range(len(vocabulary)):
            if traditional_vectors[i][j] > 0:
                ax1.text(j, i, f"{traditional_vectors[i][j]:.1f}", 
                        ha="center", va="center", color="black", fontsize=8)
    
    # Plot SPLADE-like sparse vectors
    im2 = ax2.imshow(splade_vectors, aspect='auto', cmap='Greens')
    plt.colorbar(im2, ax=ax2, label='Activation Value')
    ax2.set_title('SPLADE-like Neural Sparse Vectors (with Term Expansion)')
    ax2.set_xlabel('Terms (Vocabulary)')
    ax2.set_ylabel('Documents')
    ax2.set_xticks(range(len(vocabulary)))
    ax2.set_xticklabels(vocabulary, rotation=90)
    ax2.set_yticks(range(len(documents)))
    ax2.set_yticklabels([f"Doc {i+1}" for i in range(len(documents))])
    
    # Add text annotations for non-zero values
    for i in range(len(documents)):
        for j in range(len(vocabulary)):
            if splade_vectors[i][j] > 0:
                ax2.text(j, i, f"{splade_vectors[i][j]:.1f}", 
                        ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return img_str

@app.route('/')
def index():
    data = create_vectors()
    img_str = create_visualization(data)
    return render_template('index.html', 
                           documents=documents,
                           vocabulary=data['vocabulary'],
                           visualization=img_str)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    data = create_vectors(query)
    img_str = create_visualization(data)
    
    return jsonify({
        'query': query,
        'traditional_ranking': data['traditional_ranking'],
        'splade_ranking': data['splade_ranking'],
        'visualization': img_str
    })

@app.route('/api/data')
def get_data():
    query = request.args.get('query')
    data = create_vectors(query)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)