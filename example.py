from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

EMBEDDING_MODEL_NAME = "nomic-embed-text"

# Simple texts
texts = [
    "Apple is releasing a new iPhone this year.",
    "Amazon stock price is rising steadily.",
    "Bananas are yellow and tasty.",
    "Orange juice is refreshing.",
    "Microsoft announces new Surface laptop."
]

# Wrap texts as Documents
docs = [Document(page_content=text) for text in texts]

# Create embedding instance
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# Create FAISS vector store from docs
vectorstore = FAISS.from_documents(docs, embeddings)

# Helper for cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Embeddings for documents (first 8 dimensions):")
doc_vectors = []
for i, doc in enumerate(docs):
    vector = embeddings.embed_query(doc.page_content)
    doc_vectors.append(vector)
    print(f"Doc {i+1}: {doc.page_content}")
    print(f"Vector: {vector[:8]}\n")

# Query
query = "Tell me about Amazon stock"
query_vector = embeddings.embed_query(query)
print(f"Query vector (first 8 dimensions): {query_vector[:8]}")

# Similarity search
results = vectorstore.similarity_search(query, k=3)

print("\nTop 3 similar documents to query:")
for i, res in enumerate(results):
    print(f"Result {i+1}:")
    print("Text:", res.page_content)
    print("Metadata:", res.metadata)
    # Find cosine similarity for printout
    idx = texts.index(res.page_content)
    sim_score = cosine_sim(query_vector, doc_vectors[idx])
    print(f"Cosine similarity: {sim_score:.4f}\n")
