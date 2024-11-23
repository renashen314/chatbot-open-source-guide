from ollama_text_embedding import get_embedding_function 
from langchain_community.llms.ollama import Ollama

PROMPT_TEMPLATE = """

"""
def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    model = Ollama(model="")
    response_text = model.invoke(prompt)
    source = [doc.metadata.get("id, None") for doc, _score in results]
    print(response_text)

