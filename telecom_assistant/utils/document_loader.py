# Document loader utilities
"""
Document Loader for Telecom Assistant
Handles loading and processing of documents using LlamaIndex
"""

from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import OPENAI_API_KEY, LLM_MODEL


def load_documents_from_directory(directory_path: str):
    """
    Load documents from a directory using SimpleDirectoryReader.
    
    Args:
        directory_path: Path to the directory containing documents
        
    Returns:
        List of loaded documents
    """
    reader = SimpleDirectoryReader(
        directory_path,
        required_exts=[".pdf", ".md", ".txt"],
        recursive=False
    )
    documents = reader.load_data()
    return documents


def initialize_llama_settings():
    """Initialize LlamaIndex settings with OpenAI models"""
    Settings.llm = LlamaOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )


def create_vector_store_index(documents, chroma_collection):
    """
    Create a vector store index from documents using ChromaDB.
    
    Args:
        documents: List of documents to index
        chroma_collection: ChromaDB collection
        
    Returns:
        VectorStoreIndex
    """
    from llama_index.core import StorageContext
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create and return index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    return index
