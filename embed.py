from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chromadb_store")
collection = client.get_or_create_collection("images")

def embed_and_store(caption, image_path):
    embedding = model.encode([caption])[0].tolist()
    collection.add(documents=[caption], embeddings=[embedding], ids=[image_path], metadatas=[{"path": image_path}])

