import json

from utils.chunker import chunk_documents
from utils.preprocessor import clean_text
from utils.vector_store import VectorStoreManager

COURSE = (
    "Le K-means est un algorithme de clustering qui partitionne les données en K groupes.\n\n"
    "FP-Growth extrait les itemsets fréquents grâce à un FP-tree, sans générer de candidats.\n\n"
    "La précision vaut 95% quand presque toutes les prédictions positives sont correctes."
)


def build_store():
    chunks = chunk_documents([{"page_content": COURSE, "metadata": {"filename": "cours.txt"}}], chunk_size=100, chunk_overlap=0)
    store = VectorStoreManager()
    store.build_index(chunks)
    return store


def test_clean_text_keeps_paragraphs_and_symbols():
    out = clean_text("Précision  =  95%\n\n\n\nP(A|B) est  calcu-\nlée")
    assert out == "Précision = 95%\n\nP(A|B) est calculée"


def test_search_returns_most_relevant_chunk_first():
    results = build_store().search("algorithme FP-Growth itemsets fréquents", k=3)
    assert "FP-Growth" in results[0]["content"]
    assert results[0]["score"] >= results[-1]["score"]


def test_rag_shows_only_cited_sources_and_normalizes_citations(monkeypatch):
    import src.rag as rag_module
    monkeypatch.setattr(rag_module, "chat", lambda messages, **kw: "Le K-means forme K groupes 【1】.")
    result = rag_module.RAGSystem(build_store(), min_score=0.0).ask("Que fait le K-means ?")
    assert "[1]" in result["answer"] and "【" not in result["answer"]
    assert [s["id"] for s in result["sources"]] == [1]


def test_rag_sends_document_overview_to_llm(monkeypatch):
    import src.rag as rag_module
    captured = {}

    def fake_chat(messages, **kw):
        captured["system"] = messages[0]["content"]
        return "Le cours porte sur le clustering."

    monkeypatch.setattr(rag_module, "chat", fake_chat)
    rag = rag_module.RAGSystem(build_store(), overviews={"cours.txt": "Chapitre 2 : apprentissage non supervisé"})
    rag.ask("De quoi parle le cours ?")
    assert "Chapitre 2 : apprentissage non supervisé" in captured["system"]


def test_rag_off_course_answer_has_no_sources(monkeypatch):
    import src.rag as rag_module
    answer = rag_module.OFF_COURSE_PREFIX + " Tokyo."
    monkeypatch.setattr(rag_module, "chat", lambda messages, **kw: answer)
    result = rag_module.RAGSystem(build_store(), min_score=0.0).ask("Capitale du Japon ?")
    assert result["answer"].startswith(rag_module.OFF_COURSE_PREFIX)
    assert result["sources"] == []


def test_quiz_parses_json_mode_output(monkeypatch, tmp_path):
    import src.quiz_generator as qg
    (tmp_path / "cours.txt").write_text(COURSE, encoding="utf-8")
    questions = [
        {"question": f"Concept clé numéro {i} du clustering ?",
         "choices": {"A": f"a{i}", "B": f"b{i}", "C": f"c{i}", "D": f"d{i}"},
         "correct": "B", "explanation": "parce que"}
        for i in range(3)
    ]
    monkeypatch.setattr(qg, "chat", lambda messages, **kw: json.dumps({"questions": questions}))
    quiz = qg.QuizGenerator().generate_quiz(str(tmp_path), "moyen", 3)
    assert len(quiz) == 3 and quiz[0]["correct"] == "B" and len(quiz[0]["choices"]) == 4
