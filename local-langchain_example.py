from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Local LLM and Embeddings setup
MODEL = "deepseek-r1:7b"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)


import pandas as pd
# Load the CSV data
WSB_DATA_2_SMALL = "small.csv"
field_names = [
    "id_col", "date", "title", "author", "url",
    "body", "sentiment", "rationale", "tickers_str"
]
loader = pd.read_csv(WSB_DATA_2_SMALL, names=field_names)
# Extract the 'title' + 'body' + '|' + 'sentiment' + 'reasoning' column as text data
texts = loader.apply(
    lambda row: f"{row['title']} {row['body']} | {row['sentiment']} {row['rationale']}",
    axis=1
).tolist()

# Wrap text in LangChain Document objects
documents = [Document(page_content=text) for text in texts]
print(documents[:2])  # Print first two documents to verify

# Build vector index
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")
# Load the vector store from local storage
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# Search for something similar
query = "What does WallStreetBets think about AMZN or Amazon?"
results = vectorstore.similarity_search(query, k=1)

# Print the top matches
for i, doc in enumerate(results):
    print(f"\nResult {i + 1}")
    print(doc.page_content)
    
