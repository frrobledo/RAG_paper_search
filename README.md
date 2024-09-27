# **Dual RAG Paper System**

A Retrieval-Augmented Generation (RAG) system for documents, texts, and articles using a local Large Language Model (LLM).

## **Overview**

This project implements a RAG system that stores documents on a Raspberry Pi (PC A, meant to be low performance but also capable of storing all the documents) and performs computations on a high-performance PC (PC B) equipped with a GPU. The system allows for semantic search and retrieval of documents based on user queries, utilizing the `hkunlp/instructor-large` model for generating embeddings.

## **Features**

- **Document Storage**: Store PDFs and text files on a Raspberry Pi (PC A).
- **Semantic Search**: Perform similarity searches using embeddings.
- **Document Retrieval**: Retrieve and download relevant documents (PDF or text) based on queries.
- **Local Processing**: All computations are performed locally without relying on external services.

## **Project Structure**

```
Dual RAG Paper System/
├── server/
│   ├── app.py
│   └── transform_pdf_text.sh
├── processing/
│   ├── embedding_generation.py
│   ├── vector_search.py
│   └── call_server.py
└── README.md
```

## **Prerequisites**

### **PC A (Raspberry Pi)**

- Python 3
- Flask (`pip install flask`)
- NumPy (`pip install numpy`)
- Annoy (`pip install annoy`)
- Poppler Utils (`sudo apt-get install poppler-utils`)

### **PC B (High-Performance PC)**

- Python 3
- PyTorch (`pip install torch`)
- Sentence Transformers (`pip install -U sentence-transformers`)
- NumPy (`pip install numpy`)
- Annoy (`pip install annoy`)
- Requests (`pip install requests`)

## **Setup Instructions**

### **1. Setting Up PC A (Raspberry Pi)**

#### **a. Clone the Repository**

```bash
git clone https://github.com/your-username/your-project-name.git
```

#### **b. Install Dependencies**

```bash
cd your-project-name/server
pip install flask numpy annoy
sudo apt-get install poppler-utils
```

#### **c. Prepare Documents**

- Place your PDF files in `~/documents/PDF/`.
- Run the script to convert PDFs to text:

  ```bash
  ./transform_pdf_text.sh
  ```

#### **d. Start the Flask Server**

```bash
python app.py
```

### **2. Setting Up PC B (High-Performance PC)**

#### **a. Clone the Repository**

```bash
git clone https://github.com/your-username/your-project-name.git
```

#### **b. Install Dependencies**

```bash
cd your-project-name/processing
pip install -U sentence-transformers numpy annoy requests
```

#### **c. Generate Embeddings and Build Index**

1. **Generate Embeddings:**

   ```bash
   python embedding_generation.py
   ```

2. **Build Annoy Index:**

   ```bash
   python vector_search.py
   ```

3. **Transfer Files to PC A:**

   Copy `embeddings.npy`, `doc_ids.npy`, and `annoy_index.ann` to the Raspberry Pi.

   ```bash
   scp embeddings.npy doc_ids.npy annoy_index.ann pi@<raspberry_pi_ip>:/home/pi/
   ```

#### **d. Run the Client Script**

```bash
python call_server.py
```

## **Usage**

1. **Start the Flask Server on PC A:**

   ```bash
   cd server
   python app.py
   ```

2. **Run the Client on PC B:**

   ```bash
   cd processing
   python call_server.py
   ```

3. **Enter Your Query:**

   When prompted, input your search query. The script will retrieve and save the most relevant documents.

## **Customization**

- **Adjusting the Number of Results:**

  In `call_server.py`, you can change `'num_results': 5` to the desired number of documents to retrieve.

- **Changing the Instruction for Embeddings:**

  In `embedding_generation.py` and `call_server.py`, you can modify the instruction given to the model to better suit your documents.

## **Troubleshooting**

- **Out of Memory Errors:**

  If you encounter memory errors on PC B, reduce the `batch_size` in `embedding_generation.py`.

- **Connection Issues:**

  Ensure that both PCs are on the same network and that the IP addresses are correctly specified.

- **File Not Found Errors:**

  Verify that the documents exist in the specified directories on the Raspberry Pi.

## **Contributing**

Contributions are welcome! Please open an issue or submit a pull request.

