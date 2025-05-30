from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

MODEL = "deepseek-r1:7b" 
EMBEDDING_MODEL_NAME = "nomic-embed-text"
# Initialize the LLM & Embeddings
llm = OllamaLLM(model=MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)



# Your query
query = "What does WallStreetBets think about AMZN or Amazon?"

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
# Get top 1 relevant document from your vectorstore index
results = vectorstore.similarity_search(query, k=1)

if results:
    context = results[0].page_content
    print("Context from vectorstore:")
    print(context)

    # Construct prompt including context + question
    prompt = f"""
    Given the following Reddit post from WallStreetBets:

    \"\"\"{context}\"\"\"

    Please answer the question: {query}
    """

    # Generate answer using the LLM
    response = llm(prompt)
    print("\nLLM response:")
    print(response)
else:
    print("No relevant documents found.")
