from langchain_community.retrievers import ArxivRetriever

retriver = ArxivRetriever(
    load_max_docs=3,
    load_all_available_meta=True
)

docs = retriver.invoke("Large language models")

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Published:", doc.metadata.get("Published"))
    print("Summary:", doc.page_content)