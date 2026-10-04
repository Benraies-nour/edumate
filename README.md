<p align="center">
  <img src="img/logoedumate.png" alt="EduMate" width="220">
</p>

<h1 align="center">EduMate</h1>

<p align="center">
  Assistant pédagogique IA : résumés, quiz et chatbot RAG à partir de vos propres cours.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B">
  <img alt="Groq" src="https://img.shields.io/badge/LLM-Groq-orange">
  <img alt="FAISS" src="https://img.shields.io/badge/Vector%20DB-FAISS-green">
</p>

## Fonctionnalités

| Module | Ce qu'il fait |
|---|---|
| **Résumés** | Résumé court, moyen ou détaillé de chaque document, téléchargeable en `.txt`. |
| **Quiz** | QCM générés à partir du cours, niveau de difficulté et nombre de questions réglables, correction avec explication. |
| **Chatbot** | Questions-réponses sur les cours importés, avec citation des extraits utilisés `[1]`. Quand la réponse ne figure pas dans les cours, il le signale avant de répondre avec ses connaissances générales. |

Formats acceptés : PDF, DOCX, TXT (CSV et Excel en plus pour les résumés).

## Fonctionnement du chatbot

1. **Ingestion** : extraction du texte, nettoyage qui conserve paragraphes et symboles, découpage en extraits de 800 caractères.
2. **Indexation** : embeddings multilingues calculés en local (fastembed), index FAISS en similarité cosine, mis en cache pour la session.
3. **Recherche** : les 5 extraits les plus proches de la question ; une question de suivi est enrichie avec la question précédente.
4. **Génération** : le LLM reçoit les extraits numérotés et l'aperçu de chaque document (titre, introduction, plan), ce qui lui permet de répondre aussi aux questions générales (« de quoi parle le cours ? »). Il cite ses sources, et seules les sources citées sont affichées.

## Architecture

```text
app.py                     interface Streamlit (accueil, résumés, quiz, chatbot)
src/
├── summarizer.py          résumé map-reduce : points clés par extrait, puis synthèse
├── quiz_generator.py      QCM en JSON structuré, validation et relances
└── rag.py                 chatbot : recherche, construction du contexte, citations
utils/
├── llm.py                 accès unique au LLM (Groq), relances, journal des latences et tokens
├── vector_store.py        embeddings fastembed + index FAISS
├── data_loader.py         extraction du texte (PDF, DOCX, TXT, CSV, Excel)
├── preprocessor.py        nettoyage du texte
├── chunker.py             découpage récursif en extraits
└── config.py              configuration (.env ou secrets Streamlit)
tests/                     tests pytest, sans réseau
```

| Composant | Choix |
|---|---|
| Interface | Streamlit |
| LLM | `openai/gpt-oss-120b` via l'API Groq |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (fastembed, exécuté en local) |
| Index vectoriel | FAISS, similarité cosine |

## Installation

Prérequis : Python 3.11 ou plus récent et une clé API [Groq](https://console.groq.com/keys) (gratuite).

```bash
git clone https://github.com/Benraies-nour/edumate.git
cd edumate
python -m venv .venv
```

Activer l'environnement :

```bash
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

Installer les dépendances et configurer la clé :

```bash
pip install -r requirements.txt
copy .env.example .env      # macOS / Linux : cp .env.example .env
```

Renseigner `GROQ_API_KEY` dans `.env`, puis lancer l'application :

```bash
streamlit run app.py
```

Au premier lancement du chatbot, le modèle d'embedding (environ 220 Mo) est téléchargé puis mis en cache.

## Déploiement sur Streamlit Community Cloud

1. Créer une application à partir de ce dépôt, fichier principal `app.py`.
2. Dans **Settings → Secrets**, ajouter la clé :
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

## Configuration

| Variable | Rôle | Défaut |
|---|---|---|
| `GROQ_API_KEY` | Clé API Groq | obligatoire |
| `LLM_MODEL` | Modèle Groq | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | Modèle d'embedding fastembed | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `RAG_MIN_SCORE` | Similarité cosine minimale pour qu'un extrait soit transmis au LLM | `0.15` |

## Tests

```bash
pip install pytest
pytest -q
```

Le LLM et le modèle d'embedding sont remplacés par des doublures  : les tests s'exécutent sans clé API ni réseau.

## Pistes d'évolution

- Recherche hybride (BM25 + vecteurs) et reranking des extraits.
- Numéros de page dans les citations.
- Jeu de questions de référence pour mesurer la qualité des réponses.
