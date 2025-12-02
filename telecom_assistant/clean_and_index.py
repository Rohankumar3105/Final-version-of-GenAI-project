"""
Clean ChromaDB and Index Only .txt Documents
This script will:
1. Delete the entire ChromaDB directory
2. Load ONLY the 5 .txt files (ignoring .md files)
3. Create a fresh vector index with OpenAI embeddings
4. Verify the indexing succeeded
"""
import sys
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "data" / "chromadb"
DOCS_DIR = BASE_DIR / "data" / "documents"

print("=" * 80)
print("CHROMADB CLEANUP AND REINDEXING")
print("=" * 80)

# STEP 1: Delete entire ChromaDB directory
print("\n" + "-" * 80)
print("STEP 1: Deleting ChromaDB Directory")
print("-" * 80)

if CHROMA_DIR.exists():
    print(f"Deleting: {CHROMA_DIR}")
    try:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        print(" ChromaDB directory deleted")
    except Exception as e:
        print(f"  Warning during deletion: {e}")
        print("Continuing anyway...")
else:
    print(f"ChromaDB directory doesn't exist: {CHROMA_DIR}")

# Create fresh directory
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
print(f" Created fresh ChromaDB directory")

# STEP 2: Import required libraries
print("\n" + "-" * 80)
print("STEP 2: Importing Libraries")
print("-" * 80)

try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    import chromadb
    from config.config import OPENAI_API_KEY, LLM_MODEL
    print(" All libraries imported successfully")
except ImportError as e:
    print(f" FAIL: Import error: {e}")
    print("Please install required packages:")
    print("  pip install llama-index chromadb llama-index-vector-stores-chroma llama-index-llms-openai llama-index-embeddings-openai")
    sys.exit(1)

# STEP 3: Configure LlamaIndex Settings
print("\n" + "-" * 80)
print("STEP 3: Configuring LlamaIndex Settings")
print("-" * 80)

try:
    Settings.llm = LlamaOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    print(f" LLM: {LLM_MODEL}")
    print(f" Embedding Model: text-embedding-3-small")
except Exception as e:
    print(f" FAIL: Configuration error: {e}")
    sys.exit(1)

# STEP 4: Load ONLY .txt documents
print("\n" + "-" * 80)
print("STEP 4: Loading .txt Documents ONLY")
print("-" * 80)

try:
    print(f"Documents directory: {DOCS_DIR}")
    
    documents = SimpleDirectoryReader(
        str(DOCS_DIR),
        required_exts=[".txt"],
        recursive=False
    ).load_data()
    
    print(f" Loaded {len(documents)} documents")
    
    for doc in documents:
        filename = doc.metadata.get('file_name', 'Unknown')
        char_count = len(doc.text)
        print(f"   - {filename}: {char_count:,} characters")
    
    if len(documents) == 0:
        print(f" FAIL: No .txt documents found in {DOCS_DIR}")
        sys.exit(1)
    
    md_files = list(DOCS_DIR.glob("*.md"))
    if md_files:
        print(f"\n Ignoring {len(md_files)} .md files (as requested):")
        for f in md_files:
            print(f"   - {f.name}")
    
except Exception as e:
    print(f" FAIL: Document loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 5: Create ChromaDB collection
print("\n" + "-" * 80)
print("STEP 5: Creating ChromaDB Collection")
print("-" * 80)

try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    collection_name = "telecom_knowledge_base"
    
    try:
        chroma_client.delete_collection(collection_name)
        print(f"  Deleted old collection: {collection_name}")
    except:
        pass
    
    chroma_collection = chroma_client.create_collection(collection_name)
    print(f" Created collection: {collection_name}")
    
except Exception as e:
    print(f" FAIL: ChromaDB collection error: {e}")
    sys.exit(1)

# STEP 6: Create vector index
print("\n" + "-" * 80)
print("STEP 6: Creating Vector Index (this may take a minute...)")
print("-" * 80)

try:
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(" Indexing documents with OpenAI embeddings...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(" Vector index created successfully")
    
except Exception as e:
    print(f" FAIL: Indexing error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 7: Verify indexing
print("\n" + "-" * 80)
print("STEP 7: Verifying Index")
print("-" * 80)

try:
    vector_count = chroma_collection.count()
    print(f" Total vectors in collection: {vector_count}")
    
    if vector_count == 0:
        print(f" FAIL: No vectors were created!")
        sys.exit(1)
    
    print(f"\n Indexing Statistics:")
    print(f"   Documents indexed: {len(documents)}")
    print(f"   Vectors created: {vector_count}")
    print(f"   Collection name: {collection_name}")
    print(f"   Embedding model: text-embedding-3-small")
    
except Exception as e:
    print(f" FAIL: Verification error: {e}")
    sys.exit(1)

# STEP 8: Test queries
print("\n" + "-" * 80)
print("STEP 8: Testing with Sample Queries")
print("-" * 80)

test_queries = [
    "How do I set up VoLTE on my phone?",
    "What are the billing charges for international calls?",
    "How to troubleshoot network connectivity issues?"
]

try:
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n Test Query {i}: {query}")
        
        response = query_engine.query(query)
        response_text = str(response).strip()
        
        print(f"   Response length: {len(response_text)} characters")
        print(f"   Preview: {response_text[:150]}...")
        
        if hasattr(response, 'source_nodes'):
            print(f"   Sources: {len(response.source_nodes)} documents")
    
    print(f"\n All test queries executed successfully")
    
except Exception as e:
    print(f"  Test query error: {e}")
    print("(Index was created successfully, but query test failed)")

# FINAL SUMMARY
print("\n" + "=" * 80)
print(" INDEXING COMPLETE!")
print("=" * 80)
print(f"\n Summary:")
print(f"    ChromaDB cleaned and rebuilt")
print(f"    {len(documents)} .txt files indexed")
print(f"    {vector_count} vectors created")
print(f"    Collection: {collection_name}")
print(f"\n Next Steps:")
print(f"   1. Run: python test_knowledge_agents.py")
print(f"   2. Start app: streamlit run ui/streamlit_app.py")
print(f"   3. Test query: 'How do I set up VoLTE on my phone?'")
print("=" * 80)
