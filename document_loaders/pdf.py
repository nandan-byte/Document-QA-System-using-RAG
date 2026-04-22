from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("GRU.pdf")

docs = data.load()
# splitter = TokenTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=10
# )
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

token = splitter.split_documents(docs)
print(token[0].page_content)
