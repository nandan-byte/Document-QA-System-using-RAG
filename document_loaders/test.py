from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

data = TextLoader(r"D:\ML\Genai\Part2_RAG\document_loaders\note.txt", encoding='utf-8')

docs = data.load()
chunks = splitter.split_documents(docs)
# print(chunks)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()