"""Configuration centralisée d'EduMate.

Chaque paramètre est lu dans cet ordre : variable d'environnement (ou .env),
puis secrets Streamlit (déploiement Streamlit Cloud), puis valeur par défaut.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # Pas de secrets.toml ou exécution hors Streamlit
        pass
    return default


# --- Modèles ---
GROQ_API_KEY = get_setting("GROQ_API_KEY")
LLM_MODEL = get_setting("LLM_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- Chemins ---
LOGO_PATH = str(BASE_DIR / "img" / "logoedumate.png")

# --- Chunking ---
RAG_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 120

# --- RAG ---
RAG_TOP_K = 5
# Plancher de similarité cosine : élimine seulement les extraits sans rapport.
# La pertinence fine est jugée par le LLM.
RAG_MIN_SCORE = float(get_setting("RAG_MIN_SCORE", "0.15"))
# Début de chaque document joint au contexte (questions globales : "de quoi parle le cours ?")
RAG_OVERVIEW_CHARS = 1500
