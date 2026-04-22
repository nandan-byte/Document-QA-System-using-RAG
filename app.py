# import streamlit as st
# from dotenv import load_dotenv
# import tempfile
# import os

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_mistralai import MistralAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate


# load_dotenv()

# st.set_page_config(page_title="RAG Book Assistant")

# st.title("📚 RAG Book Assistant")
# st.write("Upload a PDF and ask questions from the document")

# uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


# if uploaded_file:

#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         file_path = tmp_file.name

#     st.success("PDF uploaded successfully!")

#     if st.button("Create Vector Database"):

#         with st.spinner("Processing document..."):

#             loader = PyPDFLoader(file_path)
#             docs = loader.load()

#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200
#             )

#             chunks = splitter.split_documents(docs)

#             embeddings = MistralAIEmbeddings()

#             vectorstore = Chroma.from_documents(
#                 documents=chunks,
#                 embedding=embeddings,
#                 persist_directory="chroma_db"
#             )

#             vectorstore.persist()

#         st.success("Vector database created!")



# if os.path.exists("chroma_db"):

#     embeddings = MistralAIEmbeddings()

#     vectorstore = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     retriever = vectorstore.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 4
#     }
#     )

#     llm = ChatMistralAI(model="mistral-small-2506")

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """You are a helpful AI assistant.

# Use ONLY the provided context to answer the question.

# If the answer is not present in the context,
# say: "I could not find the answer in the document."
# """
#             ),
#             (
#                 "human",
#                 """Context:
# {context}

# Question:
# {question}
# """
#             )
#         ]
#     )

#     st.divider()
#     st.subheader("Ask Questions From the Book")

#     query = st.text_input("Enter your question")

#     if query:

#         docs = retriever.invoke(query)

#         context = "\n\n".join(
#             [doc.page_content for doc in docs]
#         )

#         final_prompt = prompt.invoke({
#             "context": context,
#             "question": query
#         })

#         response = llm.invoke(final_prompt)

#         st.write("### AI Answer")
#         st.write(response.content)

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")
st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:

    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Unique DB per file (IMPORTANT FIX)
    db_path = f"chroma_db/{uploaded_file.name}"

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            # Remove old DB for same file (clean rebuild)
            if os.path.exists(db_path):
                shutil.rmtree(db_path)

            # Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            st.info(f"Loaded {len(docs)} pages")

            # Split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )
            chunks = splitter.split_documents(docs)
            st.info(f"Created {len(chunks)} chunks")

            # Embeddings
            embeddings = MistralAIEmbeddings()

            # Create DB
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_path
            )

            vectorstore.persist()

        st.success("✅ Vector DB created successfully!")

# -----------------------------
# Query Section
# -----------------------------
if uploaded_file:

    db_path = f"chroma_db/{uploaded_file.name}"

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

            with st.spinner("Searching..."):

                docs = retriever.invoke(query)

                # -----------------------------
                # DEBUG VIEW (CRITICAL)
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

                # Output
                st.markdown("### 🤖 AI Answer")
                st.write(response.content)

    else:
        st.warning("Please create the vector database first.")