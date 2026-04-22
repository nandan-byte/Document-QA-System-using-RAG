import streamlit as st
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

if "MISTRAL_API_KEY" in st.secrets:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
else:
    st.error(" MISTRAL_API_KEY not found in Streamlit secrets")
    st.stop()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="RAG PDF Assistant")
st.title("📚 RAG PDF Assistant")
st.write("Upload a PDF and ask questions from the document")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Safe filename (IMPORTANT)
    safe_name = uploaded_file.name.replace(" ", "_").replace("/", "_")

    # Temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Unique DB per file
    db_path = f"chroma_db/{safe_name}"

    # -----------------------------
    # Create Vector DB
    # -----------------------------
    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            try:
                # Clean previous DB
                if os.path.exists(db_path):
                    shutil.rmtree(db_path)

                # Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                st.info(f"📄 Loaded {len(docs)} pages")

                # Split
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(docs)
                st.info(f"✂️ Created {len(chunks)} chunks")

                # Embeddings
                embeddings = MistralAIEmbeddings()

                # Create vector DB
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=db_path
                )

                vectorstore.persist()

                st.success("✅ Vector DB created successfully!")

            except Exception as e:
                st.error(f" Error creating vector DB: {e}")

            finally:
                # Cleanup temp file
                if os.path.exists(file_path):
                    os.remove(file_path)

# -----------------------------
# Query Section
# -----------------------------
if uploaded_file:

    db_path = f"chroma_db/{uploaded_file.name.replace(' ', '_').replace('/', '_')}"

    if os.path.exists(db_path):

        embeddings = MistralAIEmbeddings()

        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        llm = ChatMistralAI(model="mistral-small-2506")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a strict document QA assistant.

STRICT RULES:
- Use ONLY the provided context
- Do NOT use prior knowledge
- If context is irrelevant, say: "Context mismatch"
- If answer not found, say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}

Answer:"""
            )
        ])

        st.divider()
        st.subheader("💬 Ask Questions From the Document")

        query = st.text_input("Enter your question")

        if query:

            with st.spinner("🔍 Searching..."):

                try:
                    docs = retriever.invoke(query)

                    if not docs:
                        st.warning("No relevant context found.")
                        st.stop()

                    # -----------------------------
                    # DEBUG VIEW (IMPORTANT)
                    # -----------------------------
                    with st.expander("🔍 Retrieved Context"):
                        for i, doc in enumerate(docs, 1):
                            st.markdown(f"**Chunk {i} (Page {doc.metadata.get('page','N/A')}):**")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()

                    # Prepare context
                    context = "\n\n".join([
                        f"[Page {doc.metadata.get('page','N/A')}]: {doc.page_content}"
                        for doc in docs
                    ])

                    # Generate answer
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    response = llm.invoke(final_prompt)

                    st.markdown("### 🤖 AI Answer")
                    st.write(response.content)

                except Exception as e:
                    st.error(f" Error during query: {e}")

    else:
        st.warning("⚠️ Please create the vector database first.")
