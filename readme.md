# Local LLM Journey 

> To explore local LLMs, RAG, and other tools.

Using Ollama, we can grab the latest open source LLMs that are available. See [ollama.com](https://ollama.com)
> Using a Macbook Pro M4, 24GB RAM so I went with DeepSeek 7b.


**Purpose**: Private LLM processing for local use, no need for API keys or internet access. **This is all done locally** except for downloading a LLM via Ollama.



## LangChain Local LLM w/ RAG:

**LangChain**: A framework for developing applications powered by Large Language Models (LLMs)-- simplifying the stages of the application life cycle [source](https://python.langchain.com/docs/introduction/).

**Retrieval-Augmented Generation (RAG)**: "Generate an answer, but first go fetch the most relevant nuggets from a knowledge store that the model hasn’t memorized." Allows for more accurate and up-to-date responses geared towards specific **knowledge bases**.  *RAG basically puts in context before a user's prompt*.

Sample code **derived from** [LangChain Ollama Example](https://python.langchain.com/docs/how_to/local_llms/).

> I used LangChain community edition, open-source info, and docs to create and **understand** a simple search system that can retrieve relevant data from a knowledge base: (my **wallstreetbets data** set with sentiment). 

## Explanation

---
```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
```

* Imports necessary libraries for embeddings and vector stores. See [FAISS (explained further down)](https://github.com/facebookresearch/faiss)

---
```python
# Local LLM and Embeddings setup
MODEL = "deepseek-r1:7b"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
```
* Grabs local LLM (ollama pull deepseek-r1:7b) and sets up the embeddings model (nomic-embed-text).
* Creates embeddings.
> *nomic-embed-text* is trained on massive amounts of text. Through this training, it learns how words are used in context. It then assigns a list of numbers (a vector) to each piece of text.

```python
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
```
* Loads a CSV file containing WallStreetBets (data from my other project github.com/yobed/wsb/) data, extracting relevant text data from the 'title', 'body', 'sentiment', and 'rationale' columns.
* The `texts` variable is a list of strings, each containing the combined text from these columns.
---
```python
# Wrap text in LangChain Document objects
documents = [Document(page_content=text) for text in texts]
print(documents[:2])  # Print first two documents to verify
```
* Wraps the extracted text data in LangChain `Document` objects, which are used to represent the text in a structured way. Like:
```
# From Actual Document code
document = Document(
            page_content="Hello, world!",
            metadata={"source": "https://example.com"}
        )
```
---
```python
# Build vector index
vectorstore = FAISS.from_documents(documents, embeddings)
```
* This line creates a vector store using the FAISS library, which is designed for efficient similarity search and clustering of dense vectors. It takes the list of `documents` and the `embeddings` model to create a searchable index.
> More will show this later on; explaining.
---
```python
# Search for something similar
query = "What does WallStreetBets think about Nvidia or NVDA?"
results = vectorstore.similarity_search(query, k=1)
```
* This line performs a similarity search on the vector store using the provided `query`. It retrieves the top `k` most similar documents to the query based on their embeddings.
> Note I am doing k=1 to get the top result only on a small subset of data (*small.csv*).

### Results

The results are as shown for the above query:
```
Result 1
Big day folks! Good start to 2017. (NVDA and HMNY) https://i.redd.it/i6rm7yajfq701.jpg | Positive The post expresses excitement about a 'big day' and a 'good start' to the year, indicating a positive sentiment towards the mentioned stocks.
```

Let's change the query to find about Amazon from small.csv:
```python
query = "What does WallStreetBets think about AMZN or Amazon?"
```
Results:
```
Result 1
United States Post Office being sold off to Amazon! $AMZN https://i.redd.it/3ize8vud1o701.jpg | Negative The phrase 'being sold off' implies a loss or negative change regarding the United States Post Office, suggesting concern or disapproval.
```
## Pluging into LLM

Testing out on a little script *local-langchain-llm.py*:
```python
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


```

Results:
```
CONTEXT FROM VECTORSTORE:
Context from vectorstore:
United States Post Office being sold off to Amazon! $AMZN https://i.redd.it/3ize8vud1o701.jpg | Negative The phrase 'being sold off' implies a loss or negative change regarding the United States Post Office, suggesting concern or disapproval.

LLM RESPONSE:
WallstreetBets' reaction to the Reddit post suggests a level of concern or disapproval regarding Amazon (AMZN) due to the hypothetical scenario of selling off the United States Post Office. This could imply skepticism about Amazon's involvement in public services, possibly viewing it as controversial or not aligning with certain values. While the exact sentiment is not explicitly clear, the negative annotation suggests some dissatisfaction with the action rather than a direct critique of AMZN itself.
```

# Conclusion
Glad I went through it, as I learned a lot, but it also took a long time just to get here. Local LLMs are powerful, and incorporating RAG can enhance someone's capabilities significantly.

### Deeper Dive

The formula for comparing vectors is cosine similarity-- a simple explanation is (via Gemini):

Imagine two people pointing.
* Pointing in the exact same direction: The angle between their arms is 0 degrees. The cosine of 0 degrees is 1. This means they are perfectly similar in terms of direction.
* Pointing in completely opposite directions: The angle is 180 degrees. The cosine of 180 degrees is -1. This means they are perfectly dissimilar (opposite) in direction.
* Pointing at a right angle (90 degrees) to each other: They are not pointing in similar or opposite ways; their directions are unrelated (orthogonal). The cosine of 90 degrees is 0.


Check out the output from example.py that helped me better understand how it all works with the query *Tell me about Amazon stock*:

```
Embeddings for documents (first 8 dimensions):
Doc 1: Apple is releasing a new iPhone this year.
Vector: [-0.02742733, 0.07799537, -0.13070494, 0.061305147, 0.013402329, 0.024989907, -0.010277455, -0.036339115]

Doc 2: Amazon stock price is rising steadily.
Vector: [0.022809342, 0.10147492, -0.20086195, 0.04495343, 0.04608117, 0.02288313, -0.021616293, -0.003229385]

Doc 3: Bananas are yellow and tasty.
Vector: [0.02226448, 0.05847136, -0.16288817, -0.013965524, 0.05342938, 0.043206885, -0.04346327, -0.039610162]

Doc 4: Orange juice is refreshing.
Vector: [0.021066997, 0.042747572, -0.1726614, 0.020668926, 0.010601804, -0.016618805, -0.034935135, 0.011382953]

Doc 5: Microsoft announces new Surface laptop.
Vector: [0.026107596, 0.044353634, -0.13780388, 0.03633491, 0.008605899, 0.028567221, 0.00323064, -0.0038514647]

Query vector (first 8 dimensions): [-0.050112747, 0.07679666, -0.16160788, 0.03369418, 0.06466097, 0.042440627, 0.0153988, -0.026556918]

Top 3 similar documents to query:
Result 1:
Text: Amazon stock price is rising steadily.
Metadata: {}
Cosine similarity: 0.7341

Result 2:
Text: Bananas are yellow and tasty.
Metadata: {}
Cosine similarity: 0.4105

Result 3:
Text: Apple is releasing a new iPhone this year.
Metadata: {}
Cosine similarity: 0.4065
```

And we can see that result 1 is the most similar to the query, closest to 1. **People pointing in the same*ish* direction.**
















