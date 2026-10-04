import hashlib
import html
import os
import shutil
import tempfile

import streamlit as st
from PIL import Image

from src.quiz_generator import QuizGenerator
from src.rag import RAGSystem
from src.summarizer import Summarizer
from utils import config
from utils.chunker import chunk_documents
from utils.data_loader import load_and_parse_files
from utils.preprocessor import clean_text
from utils.vector_store import VectorStoreManager


@st.cache_resource(show_spinner=False, max_entries=8)
def build_vector_store(files: tuple[tuple[str, bytes], ...]) -> tuple[VectorStoreManager, dict[str, str]]:
    """Indexe les fichiers importés et renvoie l'index + un aperçu (début) de chaque document.

    Mis en cache : un même lot de fichiers n'est indexé qu'une fois.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files:
            with open(os.path.join(tmp, os.path.basename(name)), "wb") as f:
                f.write(data)
        documents = load_and_parse_files(tmp)
    for doc in documents:
        doc["page_content"] = clean_text(doc["page_content"])
    chunks = chunk_documents(documents, chunk_size=config.RAG_CHUNK_SIZE, chunk_overlap=config.RAG_CHUNK_OVERLAP)
    vsm = VectorStoreManager()
    vsm.build_index(chunks)
    overviews = {
        doc["metadata"]["filename"]: doc["page_content"][: config.RAG_OVERVIEW_CHARS]
        for doc in documents
    }
    return vsm, overviews

# Configuration de la page
st.set_page_config(page_title="EduMate", page_icon="🎓", layout="wide")

# --- CSS AMÉLIORÉ ---
st.markdown("""
<style>
.stApp { 
    background-color: white; 
}

/* Contenu centré, large mais limité sur les très grands écrans */
.block-container {
    max-width: 1100px !important;
    padding-top: 2rem !important;
}

/* Zone de saisie du chatbot alignée sur le contenu */
[data-testid="stBottomBlockContainer"] {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* Boutons centrés dans leur colonne */
div.stButton {
    display: flex !important;
    justify-content: center !important;
}

/* Boutons principaux */
div.stButton { width: 100% !important; }
div.stButton > button {
    width: 100% !important;
    max-width: 230px !important;
    height: 64px !important;
    background-color: black !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    border: none !important;
    margin: 5px !important;
    transition: background-color 0.3s, color 0.3s !important;
}
div.stButton > button p, div.stDownloadButton > button p {
    font-family: 'Garamond' !important;
    font-size: 18px !important;
    font-weight: bold !important;
    color: white !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
div.stButton > button:hover { 
    background-color: #737373 !important; 
}

/* Style spécifique pour le bouton "Générer le résumé" */
.st-key-btn_gen_res button {
    background-color: #4CAF50 !important;
    width: 100% !important;
    height: 60px !important;
}

/* Style spécifique pour le bouton "Générer le quiz" */
.st-key-btn_gen_quiz button {
    background-color: #FF6B6B !important;
    width: auto !important;
    height: 60px !important;
}

/* File uploader */
div[data-testid="stFileUploader"] * {
    color: black !important;
    background-color: #f2f2f2 !important;
    border-radius: 10px !important;
    font-size: 20px !important;
}

/* Labels des widgets */
section[data-testid="stForm"] div[data-baseweb="select"] > div:first-child,
section[data-testid="stForm"] div[data-baseweb="slider"] > div:first-child {
    color: black !important;
    font-weight: bold;
    font-size: 20px !important;
}

div[role="radiogroup"] * {
    color: black !important;
    font-family: Garamond !important;
    font-size: 23px !important;
}

/* Classes pour le quiz */
.quiz-question {
    background-color: #f8f9fa;
    padding: 20px;
    border-left: 4px solid #000000;
    border-radius: 5px;
    margin-bottom: 20px;
    font-weight: bold !important;
    font-family: Garamond !important;
    font-size: 23px !important;
}
.quiz-correct {
    background-color: #d4edda;
    padding: 15px;
    border-left: 4px solid #28a745;
    border-radius: 5px;
    margin-bottom: 15px;
}
.quiz-incorrect {
    background-color: #f8d7da;
    padding: 15px;
    border-left: 4px solid #dc3545;
    border-radius: 5px;
    margin-bottom: 15px;
}

/* Chatbot styles */
.user-message {
    background-color: #007bff;
    color: white !important;
    padding: 18px 18px;
    border-radius: 20px 20px 2px 20px;
    display: inline-block;
    max-width: 80%;
    margin-bottom: 10px;
    font-family: 'Garamond', serif;
    font-size: 23px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.bot-message {
    background-color: #f2f2f2;
    color: black !important;
    padding: 12px 18px;
    border-radius: 20px 20px 20px 2px;
    display: inline-block;
    max-width: 80%;
    margin-bottom: 10px;
    font-family: 'Garamond', serif;
    font-size: 23px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
/* --- DESIGN LARGE DU RÉSUMÉ --- */
.summary-container {
    width: 100%;
    margin-top: 20px;
}

.summary-box {
    font-family: 'Garamond', serif;
    font-size: 18px; /* Un peu plus petit pour mieux lire */
    color: black;
    background-color: #f2f2f2;
    padding: 30px;
    border-radius: 10px;
    width: 100% !important; /* Prendre toute la largeur */
    
    /* CORRECTIONS CRITIQUES POUR L'AFFICHAGE */
    height: auto !important; /* La hauteur s'adapte au texte */
    min-height: 100px;
    overflow-y: visible !important; /* Pas de barre de scroll interne cachée */
    white-space: pre-wrap !important; /* Conserve les sauts de ligne */
    margin-bottom: 20px !important;
    line-height: 1.6;
    border: 1px solid #ddd; /* Petit contour pour la visibilité */
}

div.stButton > button {
    margin: 0 auto;
    display: block;
}

            
</style>
""", unsafe_allow_html=True)

# --- Gestion d'état ---
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "show_quiz" not in st.session_state:
    st.session_state.show_quiz = False
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False
if "generated" not in st.session_state:
    st.session_state.generated = False
if "quiz_generated" not in st.session_state:
    st.session_state.quiz_generated = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summaries_data" not in st.session_state:
    st.session_state.summaries_data = []

# --- BARRE DE NAVIGATION ---
any_mode_active = st.session_state.show_uploader or st.session_state.show_quiz or st.session_state.show_chatbot

if any_mode_active:
    col_back, col_spacer, col_clear = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Retour", key="back_to_menu"):
            st.session_state.show_uploader = False
            st.session_state.show_quiz = False
            st.session_state.show_chatbot = False
            st.session_state.generated = False
            st.session_state.quiz_generated = False
            st.session_state.summaries_data = []
            st.rerun()
    with col_clear:
        if st.session_state.show_chatbot and st.session_state.chat_history:
            if st.button("Effacer l'historique", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

# --- ÉCRAN D'ACCUEIL ---
if not any_mode_active:
    try:
        # Chemin absolu ou relatif selon votre config
        image = Image.open(config.LOGO_PATH)
        image = image.resize((500, 500))
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, width='content')
    except Exception:
        pass

    st.markdown("<h2 style='text-align: center; font-family: \"Garamond\"; color: #000000;'>Comment puis-je vous aider aujourd'hui ?</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Générer des résumés", use_container_width=True):
            st.session_state.show_uploader = True
            st.rerun()
    with c2:
        if st.button("Créer un quiz", use_container_width=True):
            st.session_state.show_quiz = True
            st.rerun()
    with c3:
        if st.button("Poser une question", use_container_width=True):
            st.session_state.show_chatbot = True
            st.rerun()

# --- ZONE RÉSUMÉS ---
if st.session_state.show_uploader:
    uploaded_files = st.file_uploader("Télécharger le fichier à résumer", type=['pdf', 'docx', 'txt', 'csv', 'xls', 'xlsx'], accept_multiple_files=True)

    if uploaded_files:
        st.markdown("<p style='font-family:Calibri; font-size:20px; color:black;'>Choisissez le type de résumé :</p>", unsafe_allow_html=True)
        summary_mode = st.selectbox("", ["court", "moyen", "long"], index=1, label_visibility="collapsed")
        
        # On définit les colonnes JUSTE pour le bouton
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            # Le bouton reste centré
            if st.button("Générer le résumé", key="btn_gen_res", use_container_width=True):
                st.session_state.generated = True
                loading = st.empty()
                loading.info("🔄 Génération du résumé en cours...")
                temp_dir = tempfile.mkdtemp()
                try:
                    for f in uploaded_files:
                        file_path = os.path.join(temp_dir, f.name)
                        with open(file_path, "wb") as o: o.write(f.read())
                    
                    summarizer = Summarizer()
                    st.session_state.summaries_data = summarizer.summarize_documents(temp_dir, length=summary_mode)
                    loading.empty()

                except Exception as e:
                    loading.empty()
                    st.error(f"Erreur : {e}")
                finally:
                    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

  
    if st.session_state.generated and st.session_state.summaries_data:
        st.markdown("<br>", unsafe_allow_html=True)
        
        for s in st.session_state.summaries_data:
            filename = s['metadata']['filename']

            # Rendu Markdown sans HTML brut : la sortie du LLM ne peut pas injecter de balises
            with st.container(border=True):
                st.markdown(f"### Résumé de : {filename}")
                st.markdown(s["summary"])
            
            # Bouton de téléchargement centré
            col_dl_1, col_dl_2, col_dl_3 = st.columns([1,1,1])
            with col_dl_2:
                st.download_button(
                    label="Télécharger le résumé", 
                    data=s['summary'], 
                    file_name=f"{filename}_summary.txt",
                    key=f"dl_{filename}" # Clé unique pour éviter les conflits
                )
# --- ZONE QUIZ ---
if st.session_state.show_quiz:
    if not st.session_state.quiz_generated:
        uploaded_quiz_file = st.file_uploader("Télécharger un fichier pour le quiz", type=['pdf', 'docx', 'txt'])
        if uploaded_quiz_file:
            st.markdown("<p style='font-family:Calibri; font-size:20px; color:black;'>Difficulté :</p>", unsafe_allow_html=True)
            difficulty = st.selectbox("diff", ["facile", "moyen", "difficile"], label_visibility="collapsed")
            st.markdown("<p style='font-family:Calibri; font-size:20px; color:black;'>Nombre de questions :</p>", unsafe_allow_html=True)
            num_questions = st.slider("q_num", 5, 20, 10, label_visibility="collapsed")
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                if st.button("Générer le quiz", key="btn_gen_quiz", use_container_width=True):
                    loading = st.empty()
                    loading.info("🔄 Préparation du quiz...")
                    temp_dir = tempfile.mkdtemp()
                    try:
                        path = os.path.join(temp_dir, uploaded_quiz_file.name)
                        with open(path, "wb") as f: f.write(uploaded_quiz_file.read())
                        generator = QuizGenerator()
                        quiz_data = generator.generate_quiz(temp_dir, difficulty, num_questions)
                        loading.empty()
                        if quiz_data:
                            st.session_state.quiz = quiz_data
                            st.session_state.quiz_generated = True
                            st.session_state.user_answers = {}
                            st.rerun()
                    except Exception as e:
                        loading.empty()
                        st.error(f"Erreur : {e}")
                    finally:
                        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    if st.session_state.get('quiz_generated') and st.session_state.get('quiz'):
        quiz_data = st.session_state.quiz
            
        st.markdown("""
        <p style='text-align:center; font-weight:bold; font-size:33px; color:black; font-family:Garamond;'>
            Votre Quiz est prêt, testez vos connaissances maintenant !
        </p>
        """, unsafe_allow_html=True)

        if len(st.session_state.user_answers) > len(quiz_data):
            st.session_state.user_answers = {}

        for i, question in enumerate(quiz_data):
            if not all(key in question for key in ['question', 'choices', 'correct']):
                continue

            with st.container():
                st.markdown(f"""
                    <div class="quiz-question">
                        <p style='color: #2c3e50; font-weight: bold; font-size: 22px; margin-bottom: 15px;'>
                            Question {i+1} : {html.escape(question['question'])}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                choices = question['choices']
                choice_letters = question.get('choice_letters', ['A', 'B', 'C', 'D'])
                
                options = [f"{letter}. {choice}" for letter, choice in zip(choice_letters, choices)]
                
                user_answer = st.radio(
                    f"Réponse à la question {i+1}",
                    label_visibility="collapsed",
                    options=choice_letters,
                    format_func=lambda x, opts=options, letters=choice_letters: opts[letters.index(x)],
                    key=f"quiz_question_{i}",
                    index=None
                )
                
                if user_answer is not None:
                    st.session_state.user_answers[i] = user_answer
                
                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        answered = sum(1 for i in range(len(quiz_data)) if st.session_state.user_answers.get(i))
        all_answered = answered == len(quiz_data)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            voir_score = st.button(
                "Voir mon score",
                use_container_width=False,
                type="primary",
                disabled=not all_answered,
            )
        if not all_answered:
            st.markdown(
                f"<p style='text-align:center; color:#555;'>Répondez à toutes les questions pour voir votre score "
                f"({answered}/{len(quiz_data)} répondues).</p>",
                unsafe_allow_html=True,
            )

        if voir_score:
            score = 0
            total_questions = len(quiz_data)
            results = []

            for i, question in enumerate(quiz_data):
                user_answer = st.session_state.user_answers.get(i)
                correct_answer = question['correct']
                
                is_correct = user_answer == correct_answer
                if is_correct:
                    score += 1
                
                results.append({
                    'question_number': i + 1,
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'question_text': question['question'],
                    'explanation': question.get('explanation', '')
                })

            st.markdown("""
        <p style='text-align:center; font-weight:bold; margin-top:40px; font-size:33px; color:black; font-family:Garamond;'>
            Résultats du Quiz
        </p>
        """, unsafe_allow_html=True)

            for result in results:
                if result['is_correct']:
                    st.markdown(f"""
                        <div class="quiz-correct">
                            <p style='color: #155724; font-weight: bold; margin: 0;'>
                                ✅ Question {result['question_number']} : Correct !
                            </p>
                            <p style='color: #155724; margin: 5px 0 0 0; font-size: 14px;'>
                                Votre réponse : <strong>{result['user_answer']}</strong>
                            </p>
                            {f"<p style='color: #155724; margin: 5px 0 0 0; font-size: 14px;'>Explication : {html.escape(result['explanation'])}</p>" if result['explanation'] else ""}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="quiz-incorrect">
                            <p style='color: #721c24; font-weight: bold; margin: 0;'>
                                ❌ Question {result['question_number']} : Incorrect
                            </p>
                            <p style='color: #721c24; margin: 5px 0 0 0; font-size: 14px;'>
                                Votre réponse : <strong>{result['user_answer']}</strong> | 
                                Réponse correcte : <strong>{result['correct_answer']}</strong>
                            </p>
                            {f"<p style='color: #721c24; margin: 5px 0 0 0; font-size: 14px;'>Explication : {html.escape(result['explanation'])}</p>" if result['explanation'] else ""}
                        </div>
                    """, unsafe_allow_html=True)

            pourcentage = (score / total_questions) * 100 if total_questions > 0 else 0
            
            if pourcentage >= 80:
                message = "Excellent travail !"
                color = "#28a745"
            elif pourcentage >= 60:
                message = "Bien joué !"
                color = "#ffc107"
            else:
                message = "Continuez vos efforts !"
                color = "#dc3545"

            st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
                            padding: 5px; border-radius: 5px; margin-top: 10px;
                            border: 2px solid {color}; width: 500px; margin-left:auto; margin-right:auto;'>
                    <h2 style='color: {color}; text-align: center; margin: 0; font-size: 18px;'>
                        {message}
                    </h2>
                    <h1 style='color: {color}; text-align: center; margin: 10px 0; font-size: 40px;'>
                        {score} / {total_questions}
                    </h1>
                    <p style='color: {color}; text-align: center; margin: 0; font-size: 25px; font-weight: bold;'>
                        {pourcentage:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Nouveau quiz", use_container_width=True):
                    st.session_state.quiz_generated = False
                    st.session_state.quiz = None
                    st.session_state.user_answers = {}
                    st.rerun()

# --- ZONE CHATBOT ---
if st.session_state.show_chatbot:
    chat_files = st.file_uploader(
        "Importer les cours sur lesquels poser vos questions",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="chat_files",
    )

    if not chat_files:
        st.info("Importez un ou plusieurs cours pour commencer la conversation.")
    else:
        payload = tuple(sorted((f.name, f.getvalue()) for f in chat_files))
        fingerprint = hashlib.sha256(b"".join(name.encode() + data for name, data in payload)).hexdigest()

        # Nouveaux fichiers : nouvelle conversation
        if st.session_state.get("chat_fingerprint") != fingerprint:
            st.session_state.chat_fingerprint = fingerprint
            st.session_state.chat_history = []

        with st.spinner("Indexation des cours (le premier lancement télécharge le modèle d'embedding)..."):
            vector_store, overviews = build_vector_store(payload)

        if not vector_store.document_chunks:
            st.error("Aucun texte n'a pu être extrait de ces fichiers.")
        else:
            rag = RAGSystem(vector_store, overviews=overviews)

            def show_sources(sources, expanded=False):
                if sources:
                    with st.expander("📚 Sources consultées", expanded=expanded):
                        for src in sources:
                            st.markdown(f"**[{src['id']}] {src['file']}** · similarité {src['score']}")
                            st.caption(src["excerpt"] + "…")

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    show_sources(msg.get("sources"))

            if prompt := st.chat_input("Posez votre question..."):
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Recherche dans vos cours..."):
                        result = rag.ask(prompt, history=st.session_state.chat_history)
                    st.markdown(result["answer"])
                    show_sources(result["sources"], expanded=True)

                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )
