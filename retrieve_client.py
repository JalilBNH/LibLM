import chromadb
from typing import Optional

client = chromadb.PersistentClient('./chroma-db/')

collection = client.get_collection(name='article-collection')

query = 'How does NeRF works ?'

results = collection.query(
    query_texts=query,
    n_results=5,
)

print('\n',query, '\n')

if results["documents"] and results["metadatas"] and results["distances"] is not None:
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f'meta: {meta}, dist: {dist}\n')
        print(doc, '\n', '-------------------------------------------------------------------')
    
    
    
