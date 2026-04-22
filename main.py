# import os
# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# load_dotenv()

# # data = TextLoader(r"D:\ML\Genai\Part2_RAG\document_loaders\notes.txt", encoding='utf-8')
# # docs = data.load()
# # data = PyPDFLoader(r"document_loaders/GRU.pdf")
# # docs = data.load()
# data = PyPDFLoader("deep-learning-material-dept-ece-ase-blr-1.pdf")

# docs = data.load()

# # url = "https://www.apple.com/in/macbook-pro/"
# # data = WebBaseLoader(url)

# # docs = data.load()
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_documents(docs)


# template = ChatPromptTemplate.from_messages([
#     ("system", 
#      """You are a precise and reliable summarization assistant.

# Your task is to generate a clear, concise, and accurate summary strictly based on the provided context.

# Rules:
# 1. Use ONLY the given context. Do not add external knowledge.
# 2. If the context is insufficient, say: "Insufficient information in the provided context."
# 3. Keep the summary concise but complete (no unnecessary fluff).
# 4. Preserve key facts, numbers, and technical terms.
# 5. Do not hallucinate or assume missing details.
# 6. If multiple topics exist, organize the summary into bullet points.
# 7. Maintain a neutral and professional tone.

# Output format:
# - Short paragraph summary
# - Bullet points for key insights (if applicable)
# """),
#     ("human", "{data}")
# ])


# model = ChatMistralAI(model="mistral-small", api_key=os.getenv("MISTRAL_API_KEY"), max_retries=5)

# prompt = template.format_messages(data=docs[0].page_content)

# result = model.invoke(prompt)

# print(result.content)


from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":4,
                   "fetch_k":10,
                   "lambda_mult":0.5}
)

llm = ChatMistralAI(model="mistral-small-latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a precise and reliable summarization assistant.

Your task is to generate a clear, concise, and accurate summary strictly based on the provided context.

Rules:
1. Use ONLY the given context. Do not add external knowledge.
2. If the context is insufficient, say: "Insufficient information in the provided context."
3. Keep the summary concise but complete (no unnecessary fluff).
4. Preserve key facts, numbers, and technical terms.
5. Do not hallucinate or assume missing details.
6. If multiple topics exist, organize the summary into bullet points.
7. Maintain a neutral and professional tone.

Output format:
- Short paragraph summary
- Bullet points for key insights (if applicable)
"""),
    ("human", 
     """Context:
     {context}
     
Question:
{question}
""")
])

print("Rag System Creatted")
print("Press 0 to exit ")

while True:
    query = input("You: ")
    if query == "0":
        break
    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })
    
    response  = llm.invoke(final_prompt)
    
    print(f"\nBot: {response.content}")