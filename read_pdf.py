import pymupdf4llm
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid 
    
chunk_size = 1000
chunk_overlap = 20
chromadb_path = Path('./chroma-db')
data_path = Path('./data/')

splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
)

test_client = chromadb.EphemeralClient
client = chromadb.PersistentClient(path=chromadb_path)
collection = client.get_or_create_collection(name='article-collection')


for file_path in data_path.iterdir():
    text = pymupdf4llm.to_markdown(file_path)
    chunks = splitter.split_text(str(text))
    
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    metadata = [{"source": file_path.name, "path":str(file_path), "index":i} for i in range(len(chunks))]
    
    for i in range(len(chunks)):
        metadata = {"source": file_path.name, "path":str(file_path), "index":i}
        collection.add(
            ids = ids[i],
            metadatas=metadata,
            documents=chunks[i],
        )
        
        
    
