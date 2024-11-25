from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def load_documents(path):
    loader = loader = PyPDFLoader(path)
    document = loader.load()
    return document

def split_documents(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(document)

# documents = load_documents("data/maintaining-open-source-projects.pdf")
# chunks = split_documents(documents)
# print(chunks[100])
