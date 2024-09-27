from sentence_transformers import SentenceTransformer
import requests
import os

# Server IP address
server_ip = '<raspberry_pi_ip>'

# Initialize the model
model = SentenceTransformer('hkunlp/instructor-large')

# User input
query = input("Your query here: ")
query_input = [["Represent the question for retrieving supporting scientific papers", query]]
query_embedding = model.encode(query_input)[0]

# Prepare the search request
headers = {'Authorization': 'your_secret_token'}  # Update if you have authentication
data = {
    'embedding': query_embedding.tolist(),
    'num_results': 5
}

# Send the search request
response = requests.post(f'http://{server_ip}:5000/search', 
                         json=data, 
                         headers=headers)

results = response.json()

# Process the search results
for result in results:
    doc_id = result['doc_id']
    print(f"Document ID: {doc_id}")

    # Prepare the document retrieval request
    data = {'doc_id': doc_id}
    file_response = requests.post(f'http://{server_ip}:5000/document', json=data, headers=headers)

    if file_response.status_code == 200:
        # Determine file type
        content_type = file_response.headers.get('Content-Type')
        if content_type == 'application/pdf':
            extension = '.pdf'
        elif content_type == 'text/plain':
            extension = '.txt'
        else:
            extension = ''

        # Save the file
        base_filename = os.path.splitext(doc_id)[0]
        filename = f"{base_filename}{extension}"
        with open(filename, 'wb') as f:
            f.write(file_response.content)
        print(f"File saved as {filename}")
    else:
        error_message = file_response.json().get('error', 'Unknown error')
        print(f"Error retrieving file: {error_message}")

    print('--' * 20)
