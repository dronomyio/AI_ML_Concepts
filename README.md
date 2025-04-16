# Embeddings Demos

This project contains demonstrations of different embedding techniques for document representation and search:

1. **Simple Sparse Embeddings**: Traditional sparse representations using Bag-of-Words and TF-IDF
2. **Neural Sparse Embeddings**: SPLADE-inspired embeddings with neural term expansion

## Getting Started

### Running with Docker Compose

The easiest way to run all three applications is using Docker Compose:

```bash
docker-compose up
```

This will start three services:
- Main application at http://localhost:5000
- Simple Embeddings at http://localhost:5001
- SPLADE Embeddings at http://localhost:5002

### Running Individually

To run the applications individually:

#### Main App
```bash
python app.py
```

#### Simple Embeddings
```bash
cd simple_embeddings
python app.py
```

#### SPLADE Embeddings
```bash
cd SPLADE
python app.py
```

## Services and Ports

- **Main App**: http://localhost:5000 - Entry point to both embedding demos
- **Simple Embeddings**: http://localhost:5001 - Traditional sparse embeddings demo
- **SPLADE Embeddings**: http://localhost:5002 - Neural sparse embeddings demo

## Project Structure

```
.
├── app.py                # Main entry point application
├── docker-compose.yml    # Docker compose configuration
├── Dockerfile            # Docker configuration for main app
├── requirements.txt      # Dependencies for main app
├── templates/            # Templates for main app
│   └── index.html        # Landing page
├── simple_embeddings/    # Simple sparse embeddings demo
└── SPLADE/               # Neural sparse embeddings demo
```

Each subdirectory contains its own Flask application, templates, and static files.