import chromadb
from typing import Optional
from chromadb.config import Settings

client = chromadb.PersistentClient('./chroma-db/', settings=Settings(allow_reset=True))

collections = client.get_collection(name='test-collection')



print(client.heartbeat())
client.reset()
