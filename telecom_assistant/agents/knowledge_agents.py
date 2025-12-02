# LlamaIndex knowledge agents using existing ChromaDB vectors
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

# Ensure project root on path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

# LlamaIndex / Chroma imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

# Paths
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chromadb"

# Globals (lazy singletons)
_chroma_client: Optional[chromadb.PersistentClient] = None
_vector_store: Optional[ChromaVectorStore] = None
_index: Optional[VectorStoreIndex] = None
_query_engine = None
_selected_collection_name: Optional[str] = None


def _init_openai_settings() -> None:
    """Configure LlamaIndex global settings for OpenAI."""
    # Ensure env var is set for OpenAI
    if OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    Settings.llm = LlamaOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


def _auto_select_collection(client: chromadb.PersistentClient):
    """Auto-detect a Chroma collection to use."""
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(f"No Chroma collections found in {CHROMA_DIR}")

    preferred_keywords = ["telecom", "doc", "kb", "knowledge"]

    def score(col_name: str) -> int:
        name = (col_name or "").lower()
        return sum(1 for k in preferred_keywords if k in name)

    # Choose collection with best score, then by name
    best = sorted(
        collections,
        key=lambda c: (-score(getattr(c, "name", "")), getattr(c, "name", "")),
    )[0]
    return best


def _ensure_query_engine():
    """Lazy-initialize the Chroma-backed LlamaIndex query engine."""
    global _chroma_client, _vector_store, _index, _query_engine, _selected_collection_name

    if _query_engine is not None:
        return

    _init_openai_settings()

    if not CHROMA_DIR.exists():
        raise RuntimeError(f"ChromaDB path not found: {CHROMA_DIR}")

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = _auto_select_collection(_chroma_client)
    _selected_collection_name = getattr(collection, "name", None) or "<unknown>"

    _vector_store = ChromaVectorStore(chroma_collection=collection)
    _index = VectorStoreIndex.from_vector_store(_vector_store)
    _query_engine = _index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )


def _format_sources(source_nodes) -> str:
    """Format source nodes (not currently used but kept for future use)."""
    if not source_nodes:
        return "- No sources available"
    lines: List[str] = []
    for sn in source_nodes:
        try:
            node = getattr(sn, "node", None) or sn
            score = getattr(sn, "score", None)
            md = getattr(node, "metadata", {}) or {}
            name = (
                md.get("file_name")
                or md.get("filename")
                or md.get("source")
                or md.get("doc_id")
                or "Document"
            )
            if score is not None:
                lines.append(f"- {name} (score: {score:.2f})")
            else:
                lines.append(f"- {name}")
        except Exception:
            lines.append("- Document")
    return "\n".join(lines)


def process_knowledge_query(
    query: str, customer_info: Optional[Dict[str, Any]] = None
) -> str:
    """Answer a knowledge query from local telecom docs (Chroma-backed)."""
    try:
        _ensure_query_engine()

        # Guardrail + formatting instructions
        guardrail = (
            "You are a helpful telecom customer service assistant. "
            "Using ONLY the provided context from our documentation, answer the user's question clearly and naturally. "
            "\n\n"
            "Formatting Guidelines:\n"
            "- Use a conversational tone with clear paragraphs for explanations\n"
            "- Use bullet points or numbered lists for step-by-step instructions\n"
            "- Use bold text (**text**) for emphasis on important terms\n"
            "- Keep the response well-organized and easy to read\n"
            "- Do NOT mention source files or document names\n"
            "- If the context doesn't contain the answer, politely say you don't have that information\n"
        )

        # ✅ FIXED: this line was previously broken and caused a syntax error
        composed_query = f"{guardrail}\n\nUser question: {query}"

        resp = _query_engine.query(composed_query)
        answer_text = getattr(resp, "response", None) or str(resp)

        if answer_text:
            answer_text = answer_text.strip()

        return (
            answer_text
            if answer_text
            else (
                "I apologize, but I couldn't find relevant information in our "
                "documentation to answer your question. Please contact customer "
                "support at 198 for assistance."
            )
        )

    except Exception:
        # Don't leak internal errors to user
        return (
            "I apologize, but I'm currently unable to access our knowledge base. "
            "Please try again in a moment, or contact customer support at 198 "
            "for immediate assistance."
        )
