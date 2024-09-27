from sentence_transformers import SentenceTransformer
import numpy as np
import paramiko
import os
from io import StringIO

# SSH connection details
hostname = 'IP_ADDRESS'
username = 'USER'  # Replace with your actual username
password = 'PASSWORD'  # Replace with your actual password, or use key-based authentication

model_name = 'hkunlp/instructor-large'  # Works with pairs of intruction/text
# model_name = 'all-mpnet-base-v2'

texts_path = '/home/pi/documents/texts/'

# Initialize SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Connect to the remote system
    ssh.connect(hostname, username=username, password=password)

    # Initialize SFTP client
    sftp = ssh.open_sftp()

    # Change to the remote directory
    sftp.chdir(texts_path)  # Adjust path as needed

    # List files in the remote directory
    text_file_list = sftp.listdir()

    model = SentenceTransformer(model_name)
    texts = []
    doc_ids = []

    for filename in text_file_list:
        # Read file content
        with sftp.open(filename, 'r') as file:
            text = file.read()
        texts.append(text)
        doc_ids.append(filename)  # Using filename as identifier
        print(f'Loaded file: {filename}')

    if model_name != 'hkunlp/instructor-large':
        embeddings = model.encode(texts, show_progress_bar=True)
    else:
        inputs = [["Represent the scientific paper for retrieval", text] for text in texts]
        embeddings = model.encode(inputs, batch_size=8, show_progress_bar=True)


    # Save embeddings and doc_ids locally
    np.save('embeddings.npy', embeddings)
    np.save('doc_ids.npy', doc_ids)

finally:
    # Close SFTP and SSH connections
    if 'sftp' in locals():
        sftp.close()
    ssh.close()

print("Embeddings generated and saved successfully.")