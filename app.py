import os
import re
import json
import time
import logging
import threading
from typing import List, Dict, Any, Tuple

# === ENFORCE 100% OFFLINE MODE ===
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import numpy as np
import faiss
import streamlit as st
import ollama
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# =====================================================
# CONFIGURATION & ENTERPRISE LOGGING
# =====================================================
DATA_DIR = "data_products"
VECTOR_DIR = "vector_store"
VECTOR_INDEX = os.path.join(VECTOR_DIR, "index.faiss")
DOC_STORE = os.path.join(VECTOR_DIR, "docs.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnterpriseRAG")

# =====================================================
# CORE ENGINE CLASS (Singleton Pattern)
# =====================================================
class RAGEngine:
    def __init__(self):
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm_model = "phi3:mini"
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self):
        """Loads or builds the index and BM25"""
        with self.lock:
            if os.path.exists(VECTOR_INDEX) and os.path.exists(DOC_STORE):
                self.index = faiss.read_index(VECTOR_INDEX)
                with open(DOC_STORE, "r") as f:
                    self.docs = json.load(f)
            else:
                self.docs = []
                self.index = None
            
            if self.docs:
                tokenized = [re.findall(r"[a-z0-9]+", d["text"].lower()) for d in self.docs]
                self.bm25 = BM25Okapi(tokenized)
            else:
                self.bm25 = None

    def add_document(self, file_path):
        """Incremental Update Logic"""
        time.sleep(1.5) # Wait for the OS to finish transferring the file
        
        try:
            with open(file_path, "r") as f:
                new_data = json.load(f)
            
            records = new_data if isinstance(new_data, list) else [new_data]
            new_docs = []
            
            for r in records:
                text = f"Category: {r.get('category')} | Model: {r.get('model')} | SAP: {r.get('sap_no')} | Desc: {r.get('description')}"
                new_docs.append({"text": text, "meta": r})

            with self.lock:
                embeddings = self.embed_model.encode([d["text"] for d in new_docs], normalize_embeddings=True)
                if self.index is None:
                    self.index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
                
                self.index.add(np.array(embeddings).astype("float32"))
                self.docs.extend(new_docs)
                
                faiss.write_index(self.index, VECTOR_INDEX)
                with open(DOC_STORE, "w") as f:
                    json.dump(self.docs, f)
                
                tokenized = [re.findall(r"[a-z0-9]+", d["text"].lower()) for d in self.docs]
                self.bm25 = BM25Okapi(tokenized)
                
            logger.info(f"Successfully indexed {len(new_docs)} records from {file_path}")
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")

# =====================================================
# WATCHDOG (Automatic DB Update)
# =====================================================
class DataWatcher(FileSystemEventHandler):
    def __init__(self, engine):
        self.engine = engine
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            logger.info(f"Detected new file: {event.src_path}")
            self.engine.add_document(event.src_path)

if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if 'observer' not in st.session_state:
    observer = Observer()
    observer.schedule(DataWatcher(st.session_state.rag_engine), DATA_DIR, recursive=False)
    observer.start()
    st.session_state.observer = observer

# =====================================================
# QUERY REFORMULATOR
# =====================================================
def reformulate_query(query: str, chat_history: list, engine: RAGEngine) -> str:
    """Uses the LLM to rewrite contextual follow-up questions into standalone queries."""
    if not chat_history:
        return query
    
    # Grab the last 4 messages (2 user/assistant turns) for context
    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]])
    
    prompt = f"""
    Given the following conversation history and the user's latest question, 
    rewrite the latest question so it is a standalone search query that contains all necessary context.
    Do not answer the question, JUST return the rewritten query. 
    If the question is already standalone, return it exactly as is.

    History:
    {history_text}

    Latest Question: {query}
    Rewritten Query:"""

    try:
        # We use temperature 0 for deterministic, factual rewriting
        res = ollama.chat(
            model=engine.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0} 
        )
        reformulated = res["message"]["content"].strip()
        
        # Strip out conversational filler if the LLM gets chatty
        reformulated = reformulated.replace("Rewritten Query:", "").replace('"', '').strip()
        return reformulated if reformulated else query
    except Exception as e:
        logger.error(f"Reformulation failed: {e}")
        return query

# =====================================================
# SMART ROUTER & SEARCH
# =====================================================
def enterprise_search(query, engine):
    q = query.lower()
    
    is_agg = any(word in q for word in ["how many", "list all", "total", "count"])
    
    if is_agg and engine.docs:
        results = [d for d in engine.docs if any(word in d['text'].lower() for word in q.split() if len(word) > 3)]
        return results[:50], "AGGREGATION"

    if not engine.index: return [], "NONE"
    
    emb = engine.embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, I = engine.index.search(emb, 15)
    
    tokens = re.findall(r"[a-z0-9]+", q)
    bm_scores = engine.bm25.get_scores(tokens)
    bm_ids = np.argsort(bm_scores)[::-1][:15]
    
    combined_ids = set(list(I[0]) + list(bm_ids))
    candidates = [engine.docs[int(i)] for i in combined_ids if i >= 0]
    
    if not candidates:
        return [], "NONE"
        
    pairs = [[query, c["text"]] for c in candidates]
    scores = engine.reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    return [c for c, _ in ranked[:6]], "HYBRID"

# =====================================================
# GENERATOR WITH CONVERSATIONAL MEMORY
# =====================================================
def generate_stream(context_text, mode, engine, chat_history):
    """Yields chunks of text directly from the Ollama stream, with memory."""
    
    system_msg = f"""You are a professional catalog assistant. 
    Mode: {mode}
    Task: Answer the user's latest question accurately using ONLY the provided context. 
    If the user asks for a count, be precise. 
    Context:
    {context_text}
    """
    
    messages = [{"role": "system", "content": system_msg}]
    
    # Inject recent history (excluding the system prompt)
    for m in chat_history[-5:]:
        messages.append({"role": m["role"], "content": m["content"]})
        
    stream = ollama.chat(
        model=engine.llm_model, 
        messages=messages, 
        stream=True 
    )
    for chunk in stream:
        yield chunk["message"]["content"]

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🛡️ Enterprise Uni-Doc-Intel")
st.caption("Auto-Syncing Search Engine | Real-time Catalog Intelligence")

st.sidebar.metric("Database Size", len(st.session_state.rag_engine.docs))

if st.sidebar.button("Hard Rebuild"):
    if os.path.exists(VECTOR_DIR):
        import shutil
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    st.session_state.rag_engine.refresh()
    st.rerun()

if "messages" not in st.session_state: 
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

query = st.chat_input("Ask about products or 'List all products in category X'...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context & formulating search..."):
            engine = st.session_state.rag_engine
            
            # Step 1: Reformulate query based on chat history (excluding current query)
            search_query = reformulate_query(query, st.session_state.messages[:-1], engine)
            
            # Provide UI feedback if the query was altered
            if search_query.lower() != query.lower():
                st.caption(f"🔄 *Searching catalog for: \"{search_query}\"*")
                
            # Step 2: Search using the clean, standalone query
            results, mode = enterprise_search(search_query, engine)
            
        if not results and mode != "AGGREGATION":
            ans = "I couldn't find any relevant data in the catalog."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        else:
            context_text = "\n".join([f"- {r['text']}" for r in results]) if results else "No direct matches found."
            
            # Step 3: Stream the response with full context and memory
            full_response = st.write_stream(
                generate_stream(context_text, mode, engine, st.session_state.messages)
            )
            
            if results:
                with st.expander("View Source Data"):
                    st.table([r['meta'] for r in results])

            st.session_state.messages.append({"role": "assistant", "content": full_response})
