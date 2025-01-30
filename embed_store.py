import faiss
import numpy as np
from data_preprocessing import chunk_mapping, chunk_embeddings, model

# Initialize the FAISS vector store (FlatL2 index)
vector_dim = 384  # Dimension of the embeddings (for 'all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(vector_dim)  # Simple, flat index for L2 distance (good for small datasets)

# Function to add embeddings to the FAISS index
def add_to_index(chunk_embeddings):
    vectors = []
    chunk_ids = []
    for chunk_id, embedding in chunk_embeddings:
        vectors.append(embedding.numpy())  # Convert tensor to numpy array
        chunk_ids.append(chunk_id)
    
    # Convert list of vectors to a numpy array
    vectors = np.vstack(vectors)
    
    print(f"Adding {len(vectors)} vectors to FAISS index with shape {vectors.shape}")
    
    # Add vectors to the FAISS index
    index.add(vectors)
    
    return chunk_ids

# Function to retrieve top-k relevant chunks for a query
def retrieve_chunks(query, k=5):
    query_vector = model.encode(query, convert_to_tensor=True).numpy()  # Generate embedding for the query
    distances, indices = index.search(np.array([query_vector]), k)  # Perform search
    
    # Debug: Print the indices and distances
    print(f"Indices: {indices}")
    print(f"Distances: {distances}")
    
    # Map indices back to chunk IDs
    retrieved_chunks = [chunk_mapping.get(chunk_id, "Unknown") for chunk_id in indices[0] if chunk_id < len(chunk_mapping)]
    
    return retrieved_chunks

chunk_ids = add_to_index(chunk_embeddings)  

# Optional: Save and load index from disk (for persistence between script executions)
def save_index(path="faiss_index.index"):
    faiss.write_index(index, path)

def load_index(path="faiss_index.index"):
    global index
    index = faiss.read_index(path)

print(f"Embedding for first chunk: {chunk_embeddings[0]}")
print(f"Chunk mapping sample: {list(chunk_mapping.items())[:5]}")
