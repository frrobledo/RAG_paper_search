from flask import Flask, request, jsonify, send_file
from annoy import AnnoyIndex
import numpy as np
import os

app = Flask(__name__)

# Load pre-computed embeddings and document IDs
embeddings = np.load('embeddings.npy')  # numpy array of embeddings
doc_ids = np.load('doc_ids.npy')        # numpy array of document IDs
dimension = embeddings.shape[1]         # dimensionality of the embeddings

# Initialize the Annoy index
index = AnnoyIndex(dimension, 'angular')  # Ensure 'angular' is used
index.load('annoy_index.ann')

# Define the search endpoint
@app.route('/search', methods=['POST'])
def search():
    query_embedding = request.json.get('embedding')
    num_results = request.json.get('num_results', 5)

    if query_embedding is None:
        return jsonify({'error': 'embedding is required'}), 400

    # Retrieve indices and distances
    indices, distances = index.get_nns_by_vector(
        query_embedding,
        num_results,
        include_distances=True
    )

    # Convert distances to similarities
    similarities = [1 - (d ** 2) / 2 for d in distances]

    # Create results with doc_ids and similarities
    results = []
    for idx, sim in zip(indices, similarities):
        results.append({
            'doc_id': doc_ids[idx],
            'similarity': sim
        })

    return jsonify(results)

# Define the document retrieval endpoint
@app.route('/document', methods=['POST'])
def document():
    doc_id = request.json.get('doc_id')
    if not doc_id:
        return jsonify({'error': 'doc_id is required'}), 400

    # Security check: prevent directory traversal attacks
    doc_id = os.path.basename(doc_id)

    # Paths to directories containing PDFs and text files
    pdf_dir = '/home/pi/documents/PDF'
    text_dir = '/home/pi/documents/texts'

    # Construct file paths
    base_filename = os.path.splitext(doc_id)[0]
    pdf_filename = f"{base_filename}.pdf"
    text_filename = f"{base_filename}.txt"

    pdf_path = os.path.join(pdf_dir, pdf_filename)
    text_path = os.path.join(text_dir, text_filename)

    # Send the PDF if it exists; otherwise, send the text file
    if os.path.isfile(pdf_path):
        return send_file(pdf_path, mimetype='application/pdf')
    elif os.path.isfile(text_path):
        return send_file(text_path, mimetype='text/plain')
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
