import pymupdf4llm
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import uuid 
import chromadb.utils.embedding_functions as embedding_fct


class CustomSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'title'), ('##', 'subtitle'), ('###', 'subsubtitle')])
        self.default_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    
    def split(self, md_text):
        md_chunks = self.md_splitter.split_text(md_text)
        chunks = self.default_splitter.split_documents(md_chunks)
                
        return chunks

class VectorDB:
    def __init__(self, path: str, collection_name: str) -> None:
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
    
    def add(self, documents, ids, metadatas):
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
    
    def query(self, question: str, n_results: int = 5):
        return self.collection.query(query_texts=question, n_results=n_results)
        
class RAGpipeline:
    def __init__(self, db_path: str, collection_name: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self.db_path = db_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = CustomSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.db = VectorDB(path=self.db_path, collection_name=self.collection_name)
    
    def index_pdf(self, pdf_path: Path):
        md_text = pymupdf4llm.to_markdown(pdf_path)
        chunks = self.splitter.split(md_text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        for i in range(len(chunks)):
            metadata = {"source":pdf_path.name, "path":str(pdf_path), "index": i}
            metadata.update(chunks[i].metadata)
            self.db.add(documents=chunks[i].page_content, ids=ids[i], metadatas=metadata)
    
    def index_folder(self, folder_path: Path):
        for pdf in folder_path.glob("*.pdf"):
            self.index_pdf(pdf)
            
    def query(self, question: str, n_results: int = 5):
        return self.db.query(question=question, n_results=n_results)
    
rag = RAGpipeline(
    db_path='./chroma-db/',
    collection_name='test-collection',
    chunk_size=1000,
    chunk_overlap=150,
)

quest = "What is ringnerf ?"

# rag.index_folder(folder_path=Path('./data/'))
response = rag.query(
    question=quest,
    n_results=5
)

print(quest)

if response['documents'] and response['distances'] and response['metadatas'] is not None:
    for doc, dist, metadatas in zip(response['documents'][0], response['distances'][0], response['metadatas'][0]):
        print('--------------------------------------')
        print(f'metadatas: {metadatas}, dist: {dist}')
        print('\n')
        print(doc)
        print('\n\n')