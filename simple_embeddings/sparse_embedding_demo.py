#!/usr/bin/env python3
"""
Simple Sparse Embedding Demonstration

This script demonstrates the concept of sparse embeddings in a simple, self-contained way.
It shows how documents can be represented as sparse vectors and how they can be used for search.
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

print_header("SPARSE EMBEDDING DEMONSTRATION")
print("Sample Documents:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")

# Step 1: Create a vocabulary and document-term matrix (sparse representation)
print_header("STEP 1: CREATING SPARSE VECTORS")
print("Creating sparse vectors using bag-of-words approach...")

# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the documents
count_matrix = count_vectorizer.fit_transform(documents)

# Get the vocabulary
vocabulary = count_vectorizer.get_feature_names_out()

print(f"\nVocabulary size: {len(vocabulary)}")
print(f"Vocabulary: {', '.join(vocabulary)}")

print("\nDocument-Term Matrix (Sparse Representation):")
print(f"Shape: {count_matrix.shape} (documents × terms)")
print(f"Total elements: {count_matrix.shape[0] * count_matrix.shape[1]}")
print(f"Non-zero elements: {count_matrix.nnz}")
print(f"Sparsity: {count_matrix.nnz / (count_matrix.shape[0] * count_matrix.shape[1]):.2%}")

# Step 2: Visualize the sparse vectors
print_header("STEP 2: VISUALIZING SPARSE VECTORS")

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
plt.savefig('sparse_vectors_visualization.png')
print("Sparse vectors visualization saved as 'sparse_vectors_visualization.png'")

# Step 3: Improve with TF-IDF weighting
print_header("STEP 3: TF-IDF WEIGHTED SPARSE VECTORS")
print("Creating TF-IDF weighted sparse vectors...")

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\nTF-IDF Matrix (Sparse Representation):")
print(f"Shape: {tfidf_matrix.shape} (documents × terms)")
print(f"Non-zero elements: {tfidf_matrix.nnz}")
print(f"Sparsity: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")

# Step 4: Simple search demonstration
print_header("STEP 4: SEARCH DEMONSTRATION")

# Create a simple query
query = "wild fox"
print(f"Query: '{query}'")

# Transform the query using the same vectorizer
query_vector = tfidf_vectorizer.transform([query])

# Calculate similarity (dot product)
similarity_scores = (tfidf_matrix @ query_vector.T).toarray().flatten()

print("\nSimilarity Scores (using sparse vector dot product):")
for i, score in enumerate(similarity_scores):
    print(f"Document {i+1}: {score:.4f} - {documents[i]}")

# Rank the documents
ranked_indices = np.argsort(-similarity_scores)
print("\nRanked Results:")
for rank, idx in enumerate(ranked_indices):
    print(f"{rank+1}. Document {idx+1} (Score: {similarity_scores[idx]:.4f})")
    print(f"   {documents[idx]}")

# Step 5: Demonstrate dictionary-based sparse representation
print_header("STEP 5: DICTIONARY-BASED SPARSE REPRESENTATION")
print("A more explicit representation of sparse vectors as dictionaries:")

# Convert sparse vectors to dictionaries for clearer representation
sparse_dicts = []
for i in range(tfidf_matrix.shape[0]):
    doc_vector = {}
    for j in tfidf_matrix[i].nonzero()[1]:
        term = vocabulary[j]
        value = tfidf_matrix[i, j]
        doc_vector[term] = value
    sparse_dicts.append(doc_vector)

# Print the dictionary representation
for i, doc_dict in enumerate(sparse_dicts):
    print(f"\nDocument {i+1} sparse vector:")
    print("{" + ", ".join([f"'{term}': {value:.4f}" for term, value in doc_dict.items()]) + "}")

print_header("CONCLUSION")
print("This demonstration shows how documents can be represented as sparse vectors,")
print("where most elements are zero and only a few have non-zero values.")
print("\nKey points about sparse embeddings:")
print("1. They are memory-efficient for high-dimensional data")
print("2. Each dimension corresponds to a specific term (interpretable)")
print("3. They can be weighted (e.g., using TF-IDF) to improve relevance")
print("4. They enable efficient search through sparse vector operations")
print("\nMore advanced sparse embedding methods like SPLADE use neural networks")
print("to create more sophisticated sparse representations.")

if __name__ == "__main__":
    print("\nVisualization saved. Run this script to see the full demonstration.")
