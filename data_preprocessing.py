from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess and split text corpus
def preprocess_and_split(corpus):
    chunks = []
    for doc_id, text in enumerate(corpus):
        # Split text into smaller chunks (~200-300 words each)
        words = text.split()
        for i in range(0, len(words), 200):
            chunk = " ".join(words[i:i+200])
            chunks.append((f"doc_{doc_id}_chunk_{i//200}", chunk))
    return chunks

# Example corpus (replace with your own text data)
corpus = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
    "Machine learning is a subset of artificial intelligence that involves the use of algorithms and statistical models to perform tasks without explicit instructions."
]

# Preprocess corpus into chunks
corpus_chunks = preprocess_and_split(corpus)

# Save chunk mapping for embedding
chunk_mapping = {chunk_id: text for chunk_id, text in corpus_chunks}

# Generate embeddings
def generate_embeddings(chunk_mapping):
    embeddings = []
    for chunk_id, text in chunk_mapping.items():
        embedding = model.encode(text, convert_to_tensor=True)
        embeddings.append((chunk_id, embedding))
    return embeddings

# Generate embeddings for the corpus chunks
chunk_embeddings = generate_embeddings(chunk_mapping)
