"""Chatbot RAG : recherche dans les cours importés puis réponse via le LLM."""
import logging
import re

from utils import config
from utils.llm import LLMError, chat
from utils.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

OFF_COURSE_PREFIX = "Cette information n'est pas dans les cours fournis, mais d'après mes connaissances :"

SYSTEM_PROMPT = """Tu es EduMate, un assistant pédagogique qui aide un étudiant à réviser ses cours.

Tu disposes de deux sources :
- APERÇU DES DOCUMENTS : le début de chaque document importé (titre, introduction, plan).
  Utilise-le pour les questions générales ("de quoi parle le cours ?", "quel est le plan ?").
- EXTRAITS PERTINENTS : les passages les plus proches de la question, numérotés.

Consignes :
1. Réponds à partir de ces sources dès qu'elles contiennent l'information, même partiellement.
2. Cite les extraits utilisés avec leur numéro entre crochets droits, par exemple [1] ou [2][3].
3. Seulement si aucune source ne permet de répondre, commence ta réponse exactement par :
   "{prefix}" puis réponds avec tes connaissances générales, sans citation.
4. Respecte les contraintes de format demandées (longueur, liste...). Sois précis et pédagogique.
5. Réponds en français.

APERÇU DES DOCUMENTS :
{overview}

EXTRAITS PERTINENTS :
{context}"""

_CITATION_VARIANTS = re.compile(r"【\s*(\d+)\s*(?:†[^】]*)?】")
_CITATION = re.compile(r"\[(\d+)\]")


def normalize_citations(answer: str) -> str:
    """Certains modèles citent en 【1】 : on uniformise en [1]."""
    return _CITATION_VARIANTS.sub(r"[\1]", answer)


class RAGSystem:
    """Sans état : l'historique de conversation est géré par l'interface."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        overviews: dict[str, str] | None = None,
        k: int = config.RAG_TOP_K,
        min_score: float = config.RAG_MIN_SCORE,
    ):
        self.vsm = vector_store
        self.overviews = overviews or {}
        self.k = k
        self.min_score = min_score

    def ask(self, query: str, history: list[dict] | None = None) -> dict:
        history = history or []

        # Une question de suivi ("et ses limites ?") est cherchée avec la question précédente
        previous_questions = [m["content"] for m in history if m["role"] == "user"]
        search_query = f"{previous_questions[-1]}\n{query}" if previous_questions else query

        # Seuil plancher bas : c'est le LLM qui juge si les extraits répondent à la question
        results = self.vsm.search(search_query, k=self.k, min_score=self.min_score)
        logger.info("Recherche : %d extraits (scores %s)", len(results), [round(r["score"], 2) for r in results])

        overview = "\n\n".join(f"« {name} » :\n{text}" for name, text in self.overviews.items()) or "(aucun)"
        context = "\n\n".join(
            f"[{i}] (Source : {r['metadata'].get('filename', 'document')})\n{r['content']}"
            for i, r in enumerate(results, start=1)
        ) or "(aucun)"

        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            prefix=OFF_COURSE_PREFIX, overview=overview, context=context)}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        messages.append({"role": "user", "content": query})

        try:
            answer = normalize_citations(chat(messages, temperature=0.0, max_tokens=1200))
        except LLMError as e:
            logger.error("Erreur LLM : %s", e)
            return {"answer": "Le service est momentanément indisponible, réessayez dans un instant.", "sources": []}

        if answer.startswith(OFF_COURSE_PREFIX):
            return {"answer": answer, "sources": []}

        # On n'affiche que les extraits effectivement cités
        cited = {int(n) for n in _CITATION.findall(answer)}
        sources = [
            {
                "id": i,
                "file": r["metadata"].get("filename", "document"),
                "score": round(r["score"], 3),
                "excerpt": r["content"][:300],
            }
            for i, r in enumerate(results, start=1)
            if i in cited
        ]
        return {"answer": answer, "sources": sources}
