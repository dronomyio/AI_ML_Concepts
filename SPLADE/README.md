# Neural Sparse Embedding Demo (SPLADE-like)

This application demonstrates the concept of neural network-based sparse embeddings like SPLADE (SParse Lexical AnD Expansion model) with an interactive web interface.

## Features

- Visualization of traditional sparse vectors vs neural sparse vectors
- Interactive search demo to compare both approaches
- Educational explanations about how SPLADE works
- Dockerized for easy deployment

## Running the Application

### Using Docker Compose (Recommended)

1. Make sure you have Docker and Docker Compose installed
2. Run the following command in the project directory:

```bash
docker-compose up
```

3. Access the application at http://localhost:5000

### Running Locally

1. Install Python 3.9 or later
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Access the application at http://localhost:5000

## How It Works

This demo shows:

1. **Traditional Sparse Vectors**: Simple bag-of-words approach where only terms that appear in the document are represented
2. **Neural Sparse Vectors**: SPLADE-like approach that uses neural term expansion to include semantically related terms

The demo includes:
- Visual comparison of both vector types
- Search functionality to see how each approach ranks documents
- Educational content explaining the benefits of neural sparse embeddings

## Technical Details

- The application is built with Flask
- Visualizations are created with Matplotlib
- The demo uses a simulated version of SPLADE's neural expansion for educational purposes
- In a real implementation, SPLADE would use transformer-based language models like BERT

## License

MIT