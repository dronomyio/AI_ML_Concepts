#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json.encoder = NumpyEncoder

# Sample documents
DEFAULT_DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog",
    "A fox is a wild animal from the canine family",
    "Dogs are popular pets known for their loyalty",
    "The lazy cat sleeps all day long",
    "Brown bears and red foxes are wild animals"
]

def generate_visualization(documents, count_matrix, vocabulary):
    """Generate visualization for sparse vectors"""
    # Convert sparse matrix to dense for visualization
    dense_matrix = count_matrix.toarray()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(dense_matrix, aspect='auto', cmap='Greens')
    plt.colorbar(label='Term Frequency')
    plt.xlabel('Terms (Vocabulary)')
    plt.ylabel('Documents')
    plt.title('Sparse Vector Representation of Documents')
    
    # Add x-axis labels (terms)
    plt.xticks(range(len(vocabulary)), vocabulary, rotation=90)
    plt.yticks(range(len(documents)), [f"Doc {i+1}" for i in range(len(documents))])
    
    # Add text annotations
    for i in range(dense_matrix.shape[0]):
        for j in range(dense_matrix.shape[1]):
            if dense_matrix[i, j] > 0:
                plt.text(j, i, int(dense_matrix[i, j]), 
                        ha="center", va="center", color="black", fontweight="bold")
    
    plt.tight_layout()
    
    # Save to bytes buffer instead of file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def process_documents(documents, query=None):
    """Process documents and return results"""
    # Create CountVectorizer
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(documents)
    vocabulary = count_vectorizer.get_feature_names_out()
    
    # Basic stats
    vocab_size = len(vocabulary)
    total_elements = count_matrix.shape[0] * count_matrix.shape[1]
    non_zero = count_matrix.nnz
    sparsity = non_zero / total_elements
    
    # Create TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    # Generate visualization
    visualization = generate_visualization(documents, count_matrix, vocabulary)
    
    # Create dictionary representation
    sparse_dicts = []
    for i in range(tfidf_matrix.shape[0]):
        doc_vector = {}
        for j in tfidf_matrix[i].nonzero()[1]:
            term = vocabulary[j]
            value = tfidf_matrix[i, j]
            doc_vector[term] = float(value)  # Convert numpy float to Python float for JSON
        sparse_dicts.append(doc_vector)
    
    # Process query if provided
    search_results = []
    if query:
        query_vector = tfidf_vectorizer.transform([query])
        similarity_scores = (tfidf_matrix @ query_vector.T).toarray().flatten()
        ranked_indices = np.argsort(-similarity_scores)
        
        for rank, idx in enumerate(ranked_indices):
            search_results.append({
                'rank': int(rank + 1),
                'document_id': int(idx + 1),
                'score': float(similarity_scores[idx]),
                'text': documents[idx]
            })
    
    return {
        'vocabulary_size': int(vocab_size),
        'vocabulary': vocabulary.tolist(),
        'total_elements': int(total_elements),
        'non_zero_elements': int(non_zero),
        'sparsity': float(sparsity),
        'visualization': visualization,
        'sparse_dicts': sparse_dicts,
        'search_results': search_results,
        'documents': documents
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.json
    documents = data.get('documents', DEFAULT_DOCUMENTS)
    query = data.get('query', None)
    
    results = process_documents(documents, query)
    return jsonify(results)

@app.route('/visualization.png')
def get_visualization():
    results = process_documents(DEFAULT_DOCUMENTS)
    img_data = base64.b64decode(results['visualization'])
    return send_file(io.BytesIO(img_data), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
