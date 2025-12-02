"""
Document Manager for Admin Dashboard
Handles document upload, processing, and indexing into ChromaDB
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from config.config import OPENAI_API_KEY, LLM_MODEL


# Constants
DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"


def initialize_settings():
    """Initialize LlamaIndex settings for document processing"""
    Settings.llm = LlamaOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def get_existing_documents() -> List[Dict[str, str]]:
    """
    Get list of all existing documents in the knowledge base.
    
    Returns:
        List of dictionaries containing document metadata
    """
    if not DOCS_DIR.exists():
        return []
    
    documents = []
    
    # Supported file types
    supported_extensions = {'.pdf', '.md', '.txt'}
    
    for file_path in DOCS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            # Get file metadata
            stat = file_path.stat()
            
            # Determine file type
            file_type = "PDF" if file_path.suffix.lower() == '.pdf' else \
                       "Markdown" if file_path.suffix.lower() == '.md' else \
                       "Text"
            
            documents.append({
                'name': file_path.name,
                'type': file_type,
                'upload_date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d'),
                'size': stat.st_size,
                'path': str(file_path)
            })
    
    # Sort by upload date (newest first)
    documents.sort(key=lambda x: x['upload_date'], reverse=True)
    
    return documents


def save_uploaded_file(uploaded_file) -> Tuple[bool, str, str]:
    """
    Save an uploaded file to the documents directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (success: bool, message: str, file_path: str)
    """
    try:
        # Ensure documents directory exists
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create file path
        file_path = DOCS_DIR / uploaded_file.name
        
        # Check if file already exists
        if file_path.exists():
            return False, f"File '{uploaded_file.name}' already exists. Please rename or delete the existing file first.", ""
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return True, f"Successfully saved '{uploaded_file.name}'", str(file_path)
        
    except Exception as e:
        return False, f"Error saving file: {str(e)}", ""


def process_and_index_documents(progress_callback=None) -> Tuple[bool, str, Dict]:
    """
    Process all documents and index them into ChromaDB.
    
    Args:
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (success: bool, message: str, stats: dict)
    """
    try:
        stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'processing_time': 0
        }
        
        import time
        start_time = time.time()
        
        # Initialize settings
        if progress_callback:
            progress_callback(0.1, "Initializing AI models...")
        
        initialize_settings()
        
        # Load documents
        if progress_callback:
            progress_callback(0.2, "Loading documents...")
        
        # Support PDF, Markdown, and Text files
        documents = SimpleDirectoryReader(
            str(DOCS_DIR),
            required_exts=[".pdf", ".md", ".txt"],
            recursive=False
        ).load_data()
        
        stats['total_documents'] = len(documents)
        
        if len(documents) == 0:
            return False, "No documents found to process", stats
        
        if progress_callback:
            progress_callback(0.4, f"Loaded {len(documents)} documents. Creating embeddings...")
        
        # Clean and recreate ChromaDB directory
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback(0.5, "Setting up vector database...")
        
        # Create ChromaDB collection
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection_name = "telecom_knowledge_base"
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        chroma_collection = chroma_client.create_collection(name=collection_name)
        
        if progress_callback:
            progress_callback(0.6, "Creating vector store...")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        if progress_callback:
            progress_callback(0.7, "Generating embeddings and indexing...")
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=False
        )
        
        # Calculate stats
        stats['total_chunks'] = len(chroma_collection.get()['ids'])
        stats['processing_time'] = round(time.time() - start_time, 2)
        
        if progress_callback:
            progress_callback(1.0, "Indexing complete!")
        
        message = f"""
Successfully indexed {stats['total_documents']} documents into {stats['total_chunks']} chunks.
Processing time: {stats['processing_time']} seconds.
"""
        
        return True, message, stats
        
    except Exception as e:
        import traceback
        error_msg = f"Error during indexing: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, stats


def delete_document(file_name: str) -> Tuple[bool, str]:
    """
    Delete a document from the knowledge base.
    
    Args:
        file_name: Name of the file to delete
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        file_path = DOCS_DIR / file_name
        
        if not file_path.exists():
            return False, f"File '{file_name}' not found"
        
        file_path.unlink()
        return True, f"Successfully deleted '{file_name}'"
        
    except Exception as e:
        return False, f"Error deleting file: {str(e)}"


def get_knowledge_base_stats() -> Dict:
    """
    Get statistics about the current knowledge base.
    
    Returns:
        Dictionary containing knowledge base statistics
    """
    stats = {
        'total_documents': 0,
        'total_size_mb': 0,
        'total_chunks': 0,
        'last_indexed': 'Never'
    }
    
    try:
        # Count documents
        if DOCS_DIR.exists():
            supported_extensions = {'.pdf', '.md', '.txt'}
            total_size = 0
            
            for file_path in DOCS_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    stats['total_documents'] += 1
                    total_size += file_path.stat().st_size
            
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        # Get ChromaDB stats
        if CHROMA_DIR.exists():
            try:
                chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                collection = chroma_client.get_collection(name="telecom_knowledge_base")
                stats['total_chunks'] = collection.count()
                
                # Get last modified time of ChromaDB
                chroma_files = list(CHROMA_DIR.rglob("*"))
                if chroma_files:
                    latest_mod = max(f.stat().st_mtime for f in chroma_files if f.is_file())
                    stats['last_indexed'] = datetime.fromtimestamp(latest_mod).strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
    
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    return stats
