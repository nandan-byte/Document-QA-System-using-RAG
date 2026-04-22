# 📚 Document QA System using RAG

A Streamlit web application that lets you upload any PDF document and ask natural language questions about its contents. Powered by **Retrieval-Augmented Generation (RAG)** — it finds the most relevant chunks from your document and uses an LLM to generate accurate, context-grounded answers.

---

## How It Works

The system follows a classic RAG pipeline:

1. **Upload** — You upload a PDF via the Streamlit UI.
2. **Chunk** — The document is split into overlapping text chunks (800 tokens, 150 overlap) using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embed** — Each chunk is converted to a vector embedding using `MistralAIEmbeddings`.
4. **Store** — Embeddings are persisted in a **ChromaDB** vector store, keyed per uploaded file.
5. **Retrieve** — On each question, the top-4 most semantically similar chunks are retrieved.
6. **Generate** — The retrieved context is passed to `ChatMistralAI` (`mistral-small-2506`) with a strict prompt that prevents hallucination — the model only answers from the provided context.

---

## Features

- Upload any PDF and build a searchable vector database with one click
- Ask free-form natural language questions
- Answers are grounded strictly in document content — no hallucinated prior knowledge
- Expandable debug panel shows exactly which chunks (with page numbers) were used to generate each answer
- Per-file vector databases — switching documents rebuilds the DB cleanly

---

## Project Structure

```
Document-QA-System-using-RAG/
├── app.py                  # Main Streamlit application
├── main.py                 # Standalone CLI entry point
├── create_database.py      # Script to build ChromaDB from a PDF
├── document_loaders/       # Custom or extended document loader utilities
├── retrivers/              # Retriever configuration and experiments
├── Vector Store/           # Persisted ChromaDB vector store data
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- Python 3.9+
- A [Mistral AI API key](https://console.mistral.ai/)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nandan-byte/Document-QA-System-using-RAG.git
cd Document-QA-System-using-RAG

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

The app loads this automatically via `python-dotenv`.

---

## Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Usage:**
1. Upload a PDF using the file uploader.
2. Click **"Create Vector Database"** and wait for processing to complete.
3. Type a question in the text input and press Enter.
4. View the AI-generated answer. Expand **"🔍 Retrieved Context"** to see the source chunks.

---

## Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| LLM | Mistral AI (`mistral-small-2506`) |
| Embeddings | `langchain-mistralai` MistralAI Embeddings |
| Vector Store | ChromaDB |
| PDF Loading | LangChain `PyPDFLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Orchestration | LangChain Core / Community |
| Env Config | `python-dotenv` |

---

## Notes

- The vector database is stored locally under `chroma_db/<filename>/`. Each uploaded file gets its own isolated store.
- Re-uploading the same file and clicking "Create Vector Database" will delete and rebuild the store cleanly.
- The LLM prompt enforces strict document-only answers. If your question falls outside the document's content, the model will say so rather than guess.
