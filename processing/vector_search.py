from annoy import AnnoyIndex
import numpy as np

# Load precomputed embeddings and document IDs from disk
embeddings = np.load('embeddings.npy')
doc_ids = np.load('doc_ids.npy')

# Get the dimension of the embeddings
dimension = embeddings.shape[1]

# Create an AnnoyIndex object with the dimension and the 'angular' metric
index = AnnoyIndex(dimension, 'angular')

# Add each embedding vector to the index with its corresponding document ID
for i, vector in enumerate(embeddings):
    index.add_item(i, vector)

# Build the index with 10 trees
index.build(10)  # Number of trees

# Save the index to disk
index.save('annoy_index.ann')