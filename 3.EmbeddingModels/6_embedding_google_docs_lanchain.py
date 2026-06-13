from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=32
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# # ✅ Wrap each doc in a single-item list → forces one API call per document
# all_embeddings = [
#     embeddings_model.embed_documents([doc])[0]  # [doc] = single-item list, [0] = extract vector
#     for doc in documents
# ]

# # Verify you got separate vectors
# print(f"Total embeddings: {len(all_embeddings)}")   # should be 3
# print(f"Each vector dim : {len(all_embeddings[0])}") # e.g. 3072

# for i, emb in enumerate(all_embeddings):
#     print(f"\nEmbedding {i + 1} for '{documents[i]}':")
#     print(f"  First 5 values : {emb}")
#     print(f"  Last  5 values : {emb}")
#     print("-" * 40)

# The batch embed_documents API call might be returning a single vector for all 3 sentences.
# To force it to return an individual vector for each sentence using embed_documents,
# we wrap each document in a single-item list:
results = [
    embeddings_model.embed_documents([doc])[0] 
    for doc in documents
]

for i, doc in enumerate(documents):
    print(f"Vector dimension for '{doc}': {results[i]}")
