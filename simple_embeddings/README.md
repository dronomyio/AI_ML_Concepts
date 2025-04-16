# Sparse Embeddings Demo

An interactive web application demonstrating sparse embeddings for text documents.

## Features

- Create and visualize sparse vector representations of text documents
- Compare bag-of-words and TF-IDF weighting approaches
- Perform document searches using sparse vector similarity
- Interactive UI for adding/removing documents and running queries

## Getting Started

### Running with Docker

1. Make sure you have Docker and Docker Compose installed

2. Build and start the application:
   ```
   docker-compose up
   ```

3. Access the application at http://localhost:5000

### Running Locally

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Access the application at http://localhost:5000

## How It Works

The application demonstrates:

1. **Sparse Vectors Creation**: Converting text documents into sparse vectors
2. **Vector Visualization**: Visual representation of the document-term matrix
3. **TF-IDF Weighting**: Improving relevance with term frequency-inverse document frequency
4. **Search Functionality**: Finding relevant documents based on query similarity

## Technologies Used

- Python with Flask for the web server
- scikit-learn for vectorization and TF-IDF
- Matplotlib for visualization
- Bootstrap and JavaScript for the frontend
- Docker for containerization