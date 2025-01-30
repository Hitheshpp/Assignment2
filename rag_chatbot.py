import faiss
import numpy as np
from flask import Flask, request, jsonify
from data_preprocessing import chunk_mapping, chunk_embeddings, model
import mysql.connector
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# MySQL Database setup
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "database": "chatbot_db"
}

# Connect to MySQL database
def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

# Create table for chat history
def setup_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME,
            role VARCHAR(10),
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

setup_database()

# Initialize vector store
vector_dim = 384
index = faiss.IndexFlatL2(vector_dim)

# Add embeddings to the index
def add_to_index(chunk_embeddings):
    vectors = []
    chunk_ids = []
    for chunk_id, embedding in chunk_embeddings:
        vectors.append(embedding.numpy())
        chunk_ids.append(chunk_id)
    vectors = np.vstack(vectors)
    index.add(vectors)
    return chunk_ids

# Retrieve top-k chunks from the index
def retrieve_chunks(query, k=5):
    query_vector = model.encode(query, convert_to_tensor=True).numpy()
    distances, indices = index.search(np.array([query_vector]), k)
    retrieved_chunks = [chunk_mapping.get(chunk_id, "Unknown") for chunk_id in indices[0] if chunk_id < len(chunk_mapping)]
    return retrieved_chunks

# Simple answer generation function (using retrieved chunks)
def generate_answer(query, retrieved_chunks):
    # Concatenate retrieved chunks and query for answer construction
    context = " ".join(retrieved_chunks)
    return f"Answer to '{query}': {context}"

# Store chat history in MySQL
def store_chat_history(role, content):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_history (timestamp, role, content)
        VALUES (%s, %s, %s)
    ''', (datetime.now(), role, content))
    conn.commit()
    conn.close()

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    if request.content_type != "application/json":
        return jsonify({"error": "Unsupported Media Type. Use application/json"}), 415

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON request"}), 400

    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    # Retrieve relevant chunks from FAISS index
    relevant_chunks = retrieve_chunks(query, k=3)

    # Generate answer using the retrieved chunks
    answer = generate_answer(query, relevant_chunks)

    # Store user query and system response in chat history
    store_chat_history("user", query)
    store_chat_history("system", answer)

    return jsonify({
        "query": query,
        "answer": answer,
        "retrieved_chunks": relevant_chunks
    })

# History endpoint
@app.route('/history', methods=['GET'])
def history():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM chat_history ORDER BY timestamp ASC')
    history = cursor.fetchall()
    conn.close()
    return jsonify(history)

# Main function to run Flask app
if __name__ == '__main__':
    app.run(debug=True)
