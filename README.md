# Assignment2
RAG Chatbot
README - RAG Chatbot

1. How to Install and Run the System Locally

Prerequisites
- Python 3.x installed
- MySQL installed and running
- `pip` for package management

Installation Steps
1. **Clone the repository (if applicable)**
   bash
   git clone <repository_url>
   cd AiMl_assignment_2


2. **Set up a virtual environment** (recommended)
   bash
   python -m venv AiMl_assignment_2
   source AiMl_assignment_2/bin/activate  # For Linux/Mac
   AiMl_assignment_2\Scripts\activate  # For Windows
   ```

3. **Install required dependencies**
   bash
   pip install -r requirements.txt

4. **Run the Flask application**
   bash
   python rag_chatbot.py
   
   The server will start on `http://127.0.0.1:5000`

 2. How to Set Up MySQL and Create Required Tables

### MySQL Setup
1. **Login to MySQL:**
   bash
   mysql -u root -p
   
2. **Create the chatbot database:**
   sql
   CREATE DATABASE chatbot_db;
   
3. **Update the `DB_CONFIG` in `rag_chatbot.py` if necessary:**
   python
   DB_CONFIG = {
       "host": "localhost",
       "port": 3306,
       "user": "root",
       "password": "your_password",  # Add if required
       "database": "chatbot_db"
   }
   
4. **Run the application to create the `chat_history` table automatically.**
   The table is created in `setup_database()` inside `rag_chatbot.py`.

## 3. How to Test the `/chat` and `/history` Endpoints

### Test `/chat` Endpoint
Send a POST request to `/chat`:
bash
curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d "{\"query\": \"Hello\"}"

Expected response:
json
{
  "query": "Hello",
  "answer": "(Generated response)",
  "top_chunks": ["retrieved chunk 1", "retrieved chunk 2"]
}


### Test `/history` Endpoint
Retrieve chat history:
bash
curl -X GET http://127.0.0.1:5000/history

Expected response:
json
[
  {"id": 1, "timestamp": "2025-01-30 12:00:00", "role": "user", "content": "Hello"},
  {"id": 2, "timestamp": "2025-01-30 12:00:01", "role": "system", "content": "(Generated response)"}
]


## 4. Required Environment Variables
Set the following environment variables if needed:
bash
export FLASK_APP=rag_chatbot.py
export FLASK_ENV=development
export MYSQL_USER=root
export MYSQL_PASSWORD=Hidu@9497
export MYSQL_DB=chatbot_db

For Windows (Command Prompt):
cmd
set FLASK_APP=rag_chatbot.py
set FLASK_ENV=development
set MYSQL_USER=root
set MYSQL_PASSWORD=Hidu@9497
set MYSQL_DB=chatbot_db


Now your RAG chatbot is ready to use!

