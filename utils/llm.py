"""Point d'accès unique au LLM (Groq).

Tous les modules (résumé, quiz, RAG) passent par `chat()` : changer de
fournisseur ou de modèle se fait ici, et les tests n'ont qu'une fonction à mocker.
"""
import logging
import time
from functools import lru_cache

from groq import APIConnectionError, Groq, NotFoundError, RateLimitError

from utils import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Erreur levée quand le LLM est inaccessible après les tentatives."""


@lru_cache(maxsize=1)
def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY manquante (fichier .env ou secrets Streamlit).")
    return Groq(api_key=config.GROQ_API_KEY)


def chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    json_mode: bool = False,
    model: str | None = None,
    retries: int = 3,
) -> str:
    """Envoie une conversation au LLM et renvoie le texte de la réponse."""
    model = model or config.LLM_MODEL
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    if model.startswith("openai/gpt-oss"):
        # Modèle à raisonnement : effort réduit (latence, quota) et marge de tokens
        # pour le raisonnement, qui n'apparaît pas dans la réponse.
        kwargs["reasoning_effort"] = "low"
        kwargs["include_reasoning"] = False
        kwargs["max_completion_tokens"] = max_tokens + 1024
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, retries + 1):
        try:
            start = time.perf_counter()
            completion = _client().chat.completions.create(**kwargs)
            usage = completion.usage
            logger.info(
                "LLM %s | %.2fs | tokens in=%s out=%s",
                kwargs["model"], time.perf_counter() - start,
                usage.prompt_tokens, usage.completion_tokens,
            )
            return completion.choices[0].message.content or ""
        except NotFoundError as e:
            raise LLMError(
                f"Modèle « {model} » introuvable chez Groq (peut-être retiré) : "
                "changez LLM_MODEL dans .env, voir https://console.groq.com/docs/models"
            ) from e
        except (RateLimitError, APIConnectionError) as e:
            if attempt == retries:
                raise LLMError(f"LLM indisponible : {e}") from e
            wait = 2 ** attempt
            logger.warning("Tentative %d échouée (%s), nouvel essai dans %ds", attempt, type(e).__name__, wait)
            time.sleep(wait)
    raise LLMError("Échec inattendu du LLM")
