import re
import unicodedata


def clean_text(text: str) -> str:
    """Nettoie le texte extrait en conservant sa structure.

    Les sauts de paragraphe sont gardés : le splitter s'en sert pour découper
    entre deux idées. Les symboles (%, =, |, ², ...) sont gardés : ils portent
    du sens dans des cours scientifiques.
    """
    if not text:
        return ""

    # Normalisation Unicode (ligatures, espaces insécables, accents)
    text = unicodedata.normalize("NFKC", text)

    # Unification des tirets (K-NN, FP-Growth, ...)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")

    # Recollage des mots coupés en fin de ligne
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Caractères de contrôle (sauf \n et \t)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)

    # Espaces multiples sur une ligne, lignes vides multiples
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
