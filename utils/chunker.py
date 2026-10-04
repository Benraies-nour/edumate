# utils/chunker.py
import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)

def chunk_documents(documents: List[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Transforme les documents en chunks pour le VectorStore.
    
    Args:
        documents: Liste de dicts, chacun avec 'page_content' et 'metadata'
        chunk_size: Taille maximale d'un chunk (nombre de caractères)
        chunk_overlap: Chevauchement entre deux chunks successifs

    Returns:
        Liste de dicts contenant 'content' et 'metadata'
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
            
            text = doc.get("page_content") or doc.get("content") or ""
            metadata = doc.get("metadata", {})

            if not text.strip():
                continue

            chunks = splitter.split_text(text)

            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": f"{metadata.get('filename', 'doc')}_{chunk_idx}",
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks)
                    }
                })
        
    return all_chunks