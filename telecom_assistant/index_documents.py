import chromadb
client = chromadb.PersistentClient(path='data/chromadb')
for col in client.list_collections():
    print(f"Collection: {col.name}, Count: {col.count()}")