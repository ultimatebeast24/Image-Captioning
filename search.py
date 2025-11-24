from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chromadb_store")
collection = client.get_or_create_collection("images")

def search(query, top_k=1):
    embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    return results
