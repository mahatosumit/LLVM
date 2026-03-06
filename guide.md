# Uni-Doc-Intel Hybrid RAG System

### Installation & Usage Guide (Windows)

---

# 1. Overview

Uni-Doc-Intel is a **local Retrieval Augmented Generation (RAG) system** designed to answer questions using a structured knowledge base.

The system combines:

* **Vector Search (FAISS)** for semantic similarity
* **BM25 Search** for keyword retrieval
* **Cross-Encoder Re-Ranking** for precision
* **Ollama LLM** for answer generation
* **Citation enforcement** to show the source of answers

Everything runs **fully offline**.

---

# 2. System Architecture

Pipeline used in this system:

```
User Query
    ↓
Vector Search (FAISS)
    ↓
BM25 Keyword Search
    ↓
Hybrid Retrieval
    ↓
Cross-Encoder Re-Ranking
    ↓
Context Selection
    ↓
Ollama LLM
    ↓
Answer + Citations
```

Knowledge is stored in a **JSON catalogue database**.

---

# 3. System Requirements

Recommended minimum:

| Component | Requirement             |
| --------- | ----------------------- |
| OS        | Windows 10 / Windows 11 |
| RAM       | 8 GB                    |
| Storage   | 5 GB free               |
| Python    | 3.10 – 3.12             |
| CPU       | Any modern CPU          |

Optional:

GPU can improve embedding speed but is **not required**.

---

# 4. Installing Ollama (Windows)

### Step 1 — Download Ollama

Go to:

```
https://ollama.com/download
```

Download the **Windows installer**.

---

### Step 2 — Install

Run the installer and follow the steps.

After installation verify it works.

Open **PowerShell** and run:

```
ollama --version
```

---

### Step 3 — Download the LLM Model

Pull the model used by the system:

```
ollama pull phi3
```

or

```
ollama pull gemma3:1b
```

You can test the model:

```
ollama run phi3
```

Example prompt:

```
Explain what a starter motor does
```

If the model responds, Ollama is working correctly.

---

# 5. Installing Python

Download Python from:

```
https://www.python.org/downloads/
```

Install **Python 3.11 or 3.12**.

During installation enable:

```
Add Python to PATH
```

Verify installation:

```
python --version
```

---

# 6. Project Setup

Create the project directory.

Example:

```
RAG/
```

Inside create the following structure:

```
RAG
│
├── database
│     └── products.json
│
├── vector_store
│
├── rag.py
└── requirements.txt
```

---

# 7. Create Virtual Environment

Navigate to the project folder.

```
cd RAG
```

Create a virtual environment.

```
python -m venv .venv
```

Activate it.

PowerShell:

```
.venv\Scripts\activate
```

Command prompt:

```
.venv\Scripts\activate.bat
```

You should now see:

```
(.venv)
```

in the terminal.

---

# 8. Install Required Libraries

Install all dependencies.

```
pip install faiss-cpu
pip install sentence-transformers
pip install rank_bm25
pip install numpy
pip install ollama
```

Optional (recommended):

```
pip install streamlit
```

---

# 9. Library Explanation

| Library               | Purpose                  |
| --------------------- | ------------------------ |
| sentence-transformers | Creates embeddings       |
| faiss                 | Vector similarity search |
| rank_bm25             | Keyword search           |
| CrossEncoder          | Re-ranking results       |
| numpy                 | Numerical computation    |
| ollama                | Local LLM communication  |
| streamlit             | Web interface            |

---

# 10. Knowledge Base Format

The knowledge base must be stored as **JSON**.

Example file:

```
database/products.json
```

Example structure:

```json
{
  "products": [
    {
      "category": "Starter Motor",
      "product_code": "STMR-PLSR-DK09",
      "sap_no": "A45002400",
      "description": "Starter Motor",
      "model": "Pulsar 200"
    }
  ]
}
```

Each product becomes a **retrievable knowledge chunk**.

---

# 11. Building the Vector Database

When the system runs for the first time:

```
python rag.py
```

The system will automatically:

1. Load JSON catalogue
2. Convert entries into text chunks
3. Generate embeddings
4. Create a FAISS index

Files generated:

```
vector_store/index.faiss
vector_store/docs.json
```

---

# 12. Running the System

Start the Ollama model first:

```
ollama run phi3
```

Open a second terminal and run:

```
python rag.py
```

The system will start.

Example output:

```
Hybrid RAG system ready.
Ask:
```

---

# 13. Example Queries

Examples of valid queries:

```
Which starter motor works for Pulsar 200?

Show CDI units for Pulsar 180.

What regulator rectifier works for Discover 125?
```

Example answer:

```
The starter motor used for Pulsar 200 is STMR-PLSR-DK09. [1]
```

Sources will also be displayed.

---

# 14. Citation System

Each context chunk is labeled.

Example:

```
[1] Starter Motor Pulsar 200
[2] CDI Pulsar 180
```

The LLM is instructed to cite these sources.

Example response:

```
The starter motor for Pulsar 200 is STMR-PLSR-DK09. [1]
```

This ensures **traceable answers**.

---

# 15. Hybrid Retrieval System

The system uses **two search methods**.

### Vector Search

Uses semantic similarity.

Good for queries like:

```
motor for pulsar bike
```

---

### BM25 Search

Uses keyword matching.

Good for queries like:

```
STMR-PLSR-DK09
```

---

### Hybrid Retrieval

Both methods are combined to improve recall.

---

# 16. Re-Ranking

A cross-encoder model evaluates:

```
Query + Document
```

This improves relevance before sending context to the LLM.

Model used:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

# 17. Troubleshooting

### Ollama not detected

Restart terminal and check:

```
ollama --version
```

---

### Model not found

Download again:

```
ollama pull phi3
```

---

### FAISS error

Reinstall:

```
pip install faiss-cpu
```

---

### Slow performance

Possible improvements:

* use smaller embedding model
* reduce TOP_K
* use GPU for embeddings

---

# 18. Future Improvements

Possible enhancements:

* metadata filtering
* automatic PDF → JSON extraction
* multi-catalogue search
* product recommendation system
* dashboard interface

---

# 19. Summary

Uni-Doc-Intel is a **fully offline hybrid RAG system** that provides:

* semantic search
* keyword search
* re-ranking
* citation-based answers
* local LLM reasoning

All components run locally, making the system **secure, fast, and enterprise-ready**.

---
