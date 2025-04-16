#!/usr/bin/env python3
"""
SPLADE-like Neural Sparse Embedding Demonstration

This script demonstrates the concept of neural network-based sparse embeddings like SPLADE
(SParse Lexical AnD Expansion model) without requiring the actual neural network libraries.
It simulates how SPLADE works to create more sophisticated sparse representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {text} ".center(80, "-"))
    print("="*80 + "\n")

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fox is a wild animal from the canine family",
    "Dogs are popular pets known for their loyalty",
    "The lazy cat sleeps all day long",
    "Brown bears and red foxes are wild animals"
]

print_header("NEURAL SPARSE EMBEDDING DEMONSTRATION (SPLADE-like)")
print("Sample Documents:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")

# Step 1: Create a vocabulary from the documents
print_header("STEP 1: VOCABULARY CREATION")

# Tokenize documents (simplified)
all_tokens = []
for doc in documents:
    tokens = [token.lower() for token in doc.split()]
    all_tokens.extend(tokens)

# Create vocabulary (unique tokens)
vocabulary = sorted(set(all_tokens))
vocab_size = len(vocabulary)
token_to_id = {token: idx for idx, token in enumerate(vocabulary)}

print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary: {', '.join(vocabulary)}")

# Step 2: Traditional Sparse Vectors (for comparison)
print_header("STEP 2: TRADITIONAL SPARSE VECTORS (BAG-OF-WORDS)")

# Create traditional sparse vectors (bag-of-words)
traditional_sparse_vectors = []
for doc in documents:
    vector = np.zeros(vocab_size)
    tokens = [token.lower() for token in doc.split()]
    for token in tokens:
        vector[token_to_id[token]] += 1
    traditional_sparse_vectors.append(vector)

# Calculate sparsity
total_elements = len(documents) * vocab_size
nonzero_elements = sum(np.count_nonzero(vec) for vec in traditional_sparse_vectors)
sparsity = nonzero_elements / total_elements

print(f"Traditional Sparse Vectors:")
print(f"Shape: {len(documents)} × {vocab_size} (documents × terms)")
print(f"Total elements: {total_elements}")
print(f"Non-zero elements: {nonzero_elements}")
print(f"Sparsity: {sparsity:.2%}")

# Step 3: Simulate SPLADE's neural expansion
print_header("STEP 3: SIMULATING SPLADE'S NEURAL TERM EXPANSION")

print("SPLADE uses neural networks (like BERT) to expand terms with related concepts.")
print("For example, 'dog' might activate 'canine', 'pet', 'animal', etc.")
print("This simulation demonstrates this expansion behavior without using actual neural networks.\n")

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

# Function to simulate SPLADE's ReLU and log operations
def splade_activation(value):
    """Simulate SPLADE's ReLU(log(1+x)) activation."""
    return math.log(1 + max(0, value)) if value > 0 else 0

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

# Calculate sparsity for SPLADE vectors
splade_nonzero = sum(np.count_nonzero(vec) for vec in splade_vectors)
splade_sparsity = splade_nonzero / total_elements

print(f"SPLADE-like Sparse Vectors:")
print(f"Shape: {len(documents)} × {vocab_size} (documents × terms)")
print(f"Total elements: {total_elements}")
print(f"Non-zero elements: {splade_nonzero}")
print(f"Sparsity: {splade_sparsity:.2%}")

# Step 4: Visualize the difference between traditional and SPLADE-like sparse vectors
print_header("STEP 4: VISUALIZING TRADITIONAL VS NEURAL SPARSE VECTORS")

# Create a figure with two subplots
plt.figure(figsize=(15, 10))

# Plot traditional sparse vectors
plt.subplot(2, 1, 1)
plt.imshow(traditional_sparse_vectors, aspect='auto', cmap='Blues')
plt.colorbar(label='Term Frequency')
plt.title('Traditional Sparse Vectors (Bag-of-Words)')
plt.xlabel('Terms (Vocabulary)')
plt.ylabel('Documents')
plt.xticks(range(len(vocabulary)), vocabulary, rotation=90)
plt.yticks(range(len(documents)), [f"Doc {i+1}" for i in range(len(documents))])

# Add text annotations for non-zero values
for i in range(len(documents)):
    for j in range(vocab_size):
        if traditional_sparse_vectors[i][j] > 0:
            plt.text(j, i, f"{traditional_sparse_vectors[i][j]:.1f}", 
                    ha="center", va="center", color="black", fontsize=8)

# Plot SPLADE-like sparse vectors
plt.subplot(2, 1, 2)
plt.imshow(splade_vectors, aspect='auto', cmap='Greens')
plt.colorbar(label='Activation Value')
plt.title('SPLADE-like Neural Sparse Vectors (with Term Expansion)')
plt.xlabel('Terms (Vocabulary)')
plt.ylabel('Documents')
plt.xticks(range(len(vocabulary)), vocabulary, rotation=90)
plt.yticks(range(len(documents)), [f"Doc {i+1}" for i in range(len(documents))])

# Add text annotations for non-zero values
for i in range(len(documents)):
    for j in range(vocab_size):
        if splade_vectors[i][j] > 0:
            plt.text(j, i, f"{splade_vectors[i][j]:.1f}", 
                    ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('neural_sparse_vectors.png')
print("Visualization saved as 'neural_sparse_vectors.png'")

# Step 5: Demonstrate search with both approaches
print_header("STEP 5: SEARCH DEMONSTRATION")

# Create a simple query
query = "wild animal"
print(f"Query: '{query}'")

# Tokenize query
query_tokens = [token.lower() for token in query.split()]

# Create traditional query vector
traditional_query_vector = np.zeros(vocab_size)
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

# Calculate similarity scores for traditional approach
traditional_scores = np.array([np.dot(traditional_query_vector, doc_vec) for doc_vec in traditional_sparse_vectors])

# Calculate similarity scores for SPLADE-like approach
splade_scores = np.array([np.dot(splade_query_vector, doc_vec) for doc_vec in splade_vectors])

# Normalize scores for fair comparison
if np.sum(traditional_scores) > 0:
    traditional_scores = traditional_scores / np.sum(traditional_scores)
if np.sum(splade_scores) > 0:
    splade_scores = splade_scores / np.sum(splade_scores)

# Print results
print("\nTraditional Sparse Vector Search Results:")
for i, score in enumerate(traditional_scores):
    print(f"Document {i+1}: {score:.4f} - {documents[i]}")

print("\nSPLADE-like Neural Sparse Vector Search Results:")
for i, score in enumerate(splade_scores):
    print(f"Document {i+1}: {score:.4f} - {documents[i]}")

# Rank the documents for both approaches
trad_ranked = np.argsort(-traditional_scores)
splade_ranked = np.argsort(-splade_scores)

print("\nTraditional Ranking:")
for rank, idx in enumerate(trad_ranked):
    print(f"{rank+1}. Document {idx+1} (Score: {traditional_scores[idx]:.4f})")
    print(f"   {documents[idx]}")

print("\nSPLADE-like Neural Ranking:")
for rank, idx in enumerate(splade_ranked):
    print(f"{rank+1}. Document {idx+1} (Score: {splade_scores[idx]:.4f})")
    print(f"   {documents[idx]}")

# Step 6: Explain how real SPLADE works
print_header("STEP 6: HOW REAL SPLADE WORKS")

print("""
In a real SPLADE implementation:

1. Neural Network Foundation:
   - SPLADE is built on pre-trained language models like BERT or DISTILBERT
   - It fine-tunes these models for the specific task of creating sparse representations

2. Architecture:
   - The model processes text through transformer layers
   - For each token, it computes activations across the entire vocabulary
   - It applies a ReLU(log(1+x)) activation function to promote sparsity

3. Training Process:
   - SPLADE is trained with specialized loss functions that balance:
     a) Relevance: Ensuring similar documents have similar representations
     b) Sparsity: Encouraging most dimensions to be zero
     c) Term expansion: Learning which terms are semantically related

4. Key Advantages:
   - Interpretability: Each dimension corresponds to a specific term
   - Efficiency: Sparse representations are memory-efficient
   - Expressiveness: Neural expansion captures semantic relationships
   - Compatibility: Works with traditional inverted index infrastructure

5. Implementation Details:
   - Uses attention mechanisms to capture context
   - Employs regularization to control sparsity levels
   - Can be trained with different pooling strategies (max pooling is common)
   - Supports separate document and query encoders

This simulation demonstrates the core concept of SPLADE - neural term expansion
that creates more expressive sparse vectors - but real implementations leverage
the full power of transformer-based language models.
""")

print_header("CONCLUSION")
print("""
This demonstration shows how neural network-based methods like SPLADE create
more sophisticated sparse embeddings compared to traditional approaches:

1. Traditional sparse vectors (like bag-of-words) only activate terms that
   explicitly appear in the text.

2. Neural sparse vectors (like SPLADE) expand beyond explicit terms to include
   semantically related concepts, capturing more meaning.

3. The neural approach improves search by:
   - Finding relevant documents even when they use different terminology
   - Maintaining the interpretability and efficiency of sparse representations
   - Leveraging the semantic understanding of neural language models

4. Real SPLADE implementations use transformer architectures like BERT to
   learn optimal term expansions from data, rather than using predefined
   expansion rules as in this simulation.
""")

if __name__ == "__main__":
    print("\nVisualization saved. Run this script to see the full demonstration.")
