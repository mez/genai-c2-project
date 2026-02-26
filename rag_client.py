import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    search_roots = [Path("."), Path("./chroma_db")]
    chroma_dirs = []
    for root in search_roots:
        chroma_dirs.extend([p.parent for p in root.rglob("chroma.sqlite3")])
    seen = set()
    for chroma_dir in chroma_dirs:
        dir_str = str(chroma_dir.resolve())
        if dir_str in seen:
            continue
        seen.add(dir_str)
        try:
            client = chromadb.PersistentClient(path=dir_str, settings=Settings(
                allow_reset=False,
                anonymized_telemetry=False
            ))
            collections = client.list_collections()
            for col in collections:
                col_name = col.name if hasattr(col, 'name') else str(col)
                # Try to get document count, fallback to 'unknown'
                try:
                    count = col.count() if hasattr(col, 'count') else 'unknown'
                except Exception:
                    count = 'unknown'
                display_name = f"{chroma_dir.name}/{col_name} ({count} docs)"
                key = f"{dir_str}:{col_name}"
                backends[key] = {
                    "directory": dir_str,
                    "collection_name": col_name,
                    "display_name": display_name,
                    "doc_count": count
                }
        except Exception as e:
            # If directory can't be accessed, add an error entry
            err = str(e)
            if len(err) > 60:
                err = err[:57] + "..."
            display_name = f"{chroma_dir.name}/ERROR: {err}"
            key = f"{dir_str}:ERROR"
            backends[key] = {
                "directory": dir_str,
                "collection_name": "ERROR",
                "display_name": display_name,
                "doc_count": 0
            }
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(allow_reset=False, anonymized_telemetry=False)
        )
        collection = client.get_collection(collection_name)
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable
    filter_dict = None
    if mission_filter and mission_filter.lower() not in ("all", "none", ""):
        filter_dict = {"mission": mission_filter}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        return results
    except Exception as e:
        return {"error": str(e)}

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["Context Documents:"]
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        mission = meta.get("mission", "Unknown Mission").replace("_", " ").title()
        source = meta.get("source", "Unknown Source")
        category = meta.get("category", "General").replace("_", " ").title()
        # Header for this document
        header = f"\n---\nDocument {idx} | Mission: {mission} | Source: {source} | Category: {category}"
        context_parts.append(header)
        # Truncate document if very long
        max_len = 1200
        doc_content = doc[:max_len] + ("..." if len(doc) > max_len else "")
        context_parts.append(doc_content)
    return "\n".join(context_parts)