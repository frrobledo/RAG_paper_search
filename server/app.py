from flask import Flask, request, jsonify, send_file
from annoy import AnnoyIndex
import numpy as np
import os

app = Flask(__name__)

# Load pre-computed embeddings and document IDs from files
embeddings = np.load('embeddings.npy')  # numpy array of embeddings
doc_ids = np.load('doc_ids.npy')        # numpy array of document IDs
dimension = embeddings.shape[1]         # dimensionality of the embeddings
model_name = 'hkunlp/instructor-large'        # name of the Hugging Face model
# model_name = 'all-mpnet-base-v2'        # name of the Hugging Face model


# Create an Annoy index with the specified dimensionality and metric (angular distance)
index = AnnoyIndex(dimension, 'angular')
# Load the pre-built Annoy index from file
index.load('annoy_index.ann')

# Define a route for the search endpoint
@app.route('/search', methods=['POST'])
def search():
    """
    Perform similarity search using the Annoy index.
    """
    # Get the query embedding and number of results from the request body
    query_embedding = request.json.get('embedding')
    num_results = request.json.get('num_results', 5)

    if query_embedding is None:
        return jsonify({'error': 'embedding is required'}), 400

    # Use the Annoy index to find the nearest neighbors to the query embedding
    indices = index.get_nns_by_vector(query_embedding, num_results, include_distances=False)

    # Create a list of document IDs corresponding to the nearest neighbors
    results = [{'doc_id': doc_ids[i]} for i in indices]

    # Return the results as JSON
    return jsonify(results)

@app.route('/document', methods=['POST'])
def document():
    """
    Return the PDF or text file corresponding to the given doc_id.
    """
    doc_id = request.json.get('doc_id')
    if not doc_id:
        return jsonify({'error': 'doc_id is required'}), 400

    # Security check: prevent directory traversal attacks
    doc_id = os.path.basename(doc_id)

    # Paths to the directories containing PDFs and text files
    pdf_dir = '/home/pi/documents/PDF'
    text_dir = '/home/pi/documents/texts'

    # Construct the PDF and text file paths
    base_filename = os.path.splitext(doc_id)[0]
    pdf_filename = f"{base_filename}.pdf"
    text_filename = f"{base_filename}.txt"

    pdf_path = os.path.join(pdf_dir, pdf_filename)
    text_path = os.path.join(text_dir, text_filename)

    # Check if the PDF file exists
    if os.path.isfile(pdf_path):
        try:
            return send_file(pdf_path, mimetype='application/pdf')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    # If PDF doesn't exist, check for text file
    elif os.path.isfile(text_path):
        try:
            return send_file(text_path, mimetype='text/plain')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
