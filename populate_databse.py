from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

def add_to_chroma(chunks: list[Document]):
    db=Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    db.add_documents(new_chunks, ids=new_chunk_ids)
    db.persist()

def update_db(db):
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids: 
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()

def tag_chunks(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
    chunk_id = f"{current_page_id}:{current_chunk_index}"
    chunk.metadata["id"] = chunk_id
    return chunk_id