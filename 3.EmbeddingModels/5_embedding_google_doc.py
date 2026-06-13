from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# To get multiple vectors for multiple queries, we can iterate over the documents.
# Passing a list directly into `contents` concatenates them into a single document.
all_embeddings = []
for doc in documents:
    response = client.models.embed_content(
        model='gemini-embedding-2',
        contents=doc,
        config={
            'output_dimensionality': 64
        },
    )
    # Each response.embeddings is a list containing the vector for the doc
    all_embeddings.extend(response.embeddings)

print(f"Generated {len(all_embeddings)} embeddings:")
for i, emb in enumerate(all_embeddings):
    print(f"Embedding {i + 1} for '{documents[i]}':")
    print(emb.values)
    print("-" * 20)