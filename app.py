#!/usr/bin/env python3
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import os
import io
import base64

# Create the main Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')

# Set the port to an environment variable or default to 5000
port = int(os.environ.get("PORT", 5000))

@app.route('/')
def index():
    """Main landing page that redirects to each embedding demo"""
    return render_template('index.html')

@app.route('/simple')
def simple_embeddings():
    """Redirect to the simple embeddings app running on port 5001"""
    return redirect('http://localhost:5001/')

@app.route('/splade')
def splade_embeddings():
    """Redirect to the SPLADE embeddings app running on port 5002"""
    return redirect('http://localhost:5002/')

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve images from static/images directory"""
    return send_file(os.path.join('static/images', filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)