import json
import re
from typing import List, Dict
from utils.data_loader import load_and_parse_files
from utils.preprocessor import clean_text
from utils.chunker import chunk_documents
from utils.llm import chat
import logging

logging.basicConfig(level=logging.INFO)

class QuizGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_quiz(
        self, 
        folder_path: str, 
        difficulty: str = "moyen", 
        num_questions: int = 10
    ) -> List[Dict]:
        """
        Génère un quiz à partir de documents.
        Approche robuste avec JSON + fallback regex.
        """
        
        # 1. Charger les documents
        docs = load_and_parse_files(folder_path)
        if not docs:
            self.logger.error("Aucun document chargé")
            return []
        # 2. Nettoyer les documents
        self.logger.info("Nettoyage des documents en cours...")
        for doc in docs:
            # On remplace le texte brut par le texte nettoyé
            doc["page_content"] = clean_text(doc["page_content"])

        # 3. Chunker les documents
        chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=50)
        
        if not chunks:
            self.logger.error("Aucun chunk généré")
            return []

        # 4. Sélectionner  les chunks
        selected_chunks = self._select_best_chunks(chunks, num_questions)
        
        # 5. Préparer le contexte
        max_context_length = 6000
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(selected_chunks):
            chunk_text = f"[Extrait {i+1}]\n{chunk['content']}"
            if current_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n\n---\n\n".join(context_parts)

        self.logger.info(f"Contexte préparé: {len(context)} caractères depuis {len(selected_chunks)} chunks")

        # 6. Générer le quiz avec retry
        return self._generate_with_retry(context, difficulty, num_questions)
    
    
    def _select_best_chunks(self, chunks: List[Dict], num_questions: int) -> List[Dict]:
        """Sélectionne les meilleurs chunks pour générer des questions variées."""
        if not chunks:
            return []
        
        total_chunks = len(chunks)
        num_chunks_needed = min(num_questions * 2, total_chunks)
        
        if total_chunks <= num_chunks_needed:
            return chunks
        
        step = total_chunks / num_chunks_needed
        selected_indices = [int(i * step) for i in range(num_chunks_needed)]
        
        selected_chunks = [chunks[i] for i in selected_indices if i < total_chunks]
        
        self.logger.info(f"Sélectionné {len(selected_chunks)} chunks pour {num_questions} questions")
        return selected_chunks

    def _generate_with_retry(
        self, 
        context: str, 
        difficulty: str, 
        num_questions: int,
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Génère le quiz avec système de retry et complétion automatique.
        """
        
        all_valid_quizzes = []
        
        for attempt in range(max_retries):
            try:
                remaining = num_questions - len(all_valid_quizzes)
                if remaining <= 0:
                    break
                
                self.logger.info(f"Tentative {attempt + 1}/{max_retries} de génération ({remaining} questions à générer)...")
                
                prompt = self._build_json_prompt(context, difficulty, remaining)
                
                response_text = chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=4000,
                    json_mode=True,
                ).strip()
                
                self.logger.info(f"Réponse reçue: {len(response_text)} caractères")
                self.logger.debug(f"Début de la réponse: {response_text[:500]}...")
                
                # Essayer d'abord le parsing JSON
                quizzes = self._parse_json_response(response_text)
                self.logger.info(f"Parsing JSON: {len(quizzes)} questions extraites")
                
                if quizzes:
                    valid_quizzes = self._validate_and_format(quizzes, remaining)
                    
                    # Ajouter les nouvelles questions valides avec gestion des doublons
                    for quiz in valid_quizzes:
                        question_text = quiz['question']
                        is_duplicate = any(
                            q['question'] == question_text 
                            for q in all_valid_quizzes
                        )
                        if not is_duplicate:
                            all_valid_quizzes.append(quiz)
                    
                    self.logger.info(f"Total accumulé: {len(all_valid_quizzes)}/{num_questions} questions")
                    
                    if len(all_valid_quizzes) >= num_questions:
                        self.logger.info(f"✓ {len(all_valid_quizzes)} questions valides générées")
                        return all_valid_quizzes[:num_questions]
                
                # Fallback regex au dernier essai
                if attempt == max_retries - 1 and len(all_valid_quizzes) < num_questions:
                    self.logger.warning("Fallback sur parsing regex...")
                    quizzes = self._parse_regex_response(response_text)
                    self.logger.info(f"Parsing regex: {len(quizzes)} questions extraites")
                    valid_quizzes = self._validate_and_format(quizzes, remaining)
                    
                    for quiz in valid_quizzes:
                        question_text = quiz['question']
                        is_duplicate = any(
                            q['question'] == question_text 
                            for q in all_valid_quizzes
                        )
                        if not is_duplicate:
                            all_valid_quizzes.append(quiz)
                    
                    self.logger.info(f"Validation finale: {len(all_valid_quizzes)} questions valides")
                    return all_valid_quizzes[:num_questions]
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Erreur tentative {attempt + 1}: {error_msg}")
                
                if "GGML_ASSERT" in error_msg or "500" in error_msg:
                    self.logger.warning("Erreur de mémoire détectée, réduction du contexte...")
                    if attempt < max_retries - 1:
                        context = context[:len(context)//2]
                        continue
                
                if attempt == max_retries - 1:
                    self.logger.error("Toutes les tentatives ont échoué")
                    return all_valid_quizzes[:num_questions] if all_valid_quizzes else []
        
        return all_valid_quizzes[:num_questions]

    def _build_json_prompt(self, context: str, difficulty: str, num_questions: int) -> str:
        """Construit le prompt pour génération JSON."""
        return f"""Tu es un expert en pédagogie et création de quiz d'évaluation académique.

    CONTENU À ANALYSER:
    {context}

    INSTRUCTIONS CRITIQUES:
    1. Génère EXACTEMENT {num_questions} questions QCM de qualité pédagogique
    2. Niveau de difficulté: {difficulty}
    3. Chaque question doit tester la COMPRÉHENSION du contenu, pas des détails sans importance

    RÈGLES STRICTES - NE PAS VIOLER:
    INTERDIT de poser des questions sur:
    - Les noms d'auteurs ou de personnes
    - Les dates de publication ou création
    - Les numéros de pages ou sections
    - Les titres de documents
    - Des informations administratives

    OBLIGATOIRE:
    - Questions sur les CONCEPTS clés
    - Questions sur le FONCTIONNEMENT des algorithmes/processus
    - Questions sur les APPLICATIONS pratiques
    - Questions sur les DIFFÉRENCES entre concepts
    - Chaque question doit être UNIQUE (pas de répétition de concept)

    FORMAT DES CHOIX:
    - Les 4 choix doivent être plausibles
    - La bonne réponse doit être claire
    - Les réponses correctes doivent être réparties entre A, B, C et D

    RÉPONSE - un objet JSON avec une clé "questions" contenant la liste :
    {{"questions": [
    {{
        "question": "Comment fonctionne le principe X ?",
        "choices": {{
        "A": "Explication plausible A",
        "B": "Explication plausible B", 
        "C": "Explication plausible C",
        "D": "Explication plausible D"
        }},
        "correct": "B",
        "explanation": "Une phrase expliquant pourquoi B est correct."
    }}
    ]}}

    Génère {num_questions} questions UNIQUES et PERTINENTES."""

    def _parse_json_response(self, response_text: str) -> List[Dict]:
        """Parse la réponse JSON du LLM avec gestion améliorée."""
        try:
            cleaned = self._clean_json_text(response_text)
            data = json.loads(cleaned)
            if isinstance(data, dict) and isinstance(data.get("questions"), list):
                data = data["questions"]

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Erreur parsing JSON: {e}")
            repaired = self._repair_broken_json(response_text)
            if repaired:
                try:
                    data = json.loads(repaired)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
                except json.JSONDecodeError:
                    pass
            return []
        except Exception as e:
            self.logger.error(f"Erreur inattendue parsing JSON: {e}")
            return []

    def _repair_broken_json(self, text: str) -> str:
        """Tente de réparer le JSON cassé."""
        try:
            text = re.sub(r',\s*}', '}', text)
            text = re.sub(r',\s*]', ']', text)
            
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                return text[start:end+1]
            
            bracket_count = 0
            in_json = False
            json_start = 0
            for i, char in enumerate(text):
                if char == '[':
                    bracket_count += 1
                    if not in_json:
                        in_json = True
                        json_start = i
                elif char == ']':
                    bracket_count -= 1
                    if in_json and bracket_count == 0:
                        return text[json_start:i+1]
            
        except Exception as e:
            self.logger.debug(f"Échec réparation JSON: {e}")
        
        return ""

    def _clean_json_text(self, text: str) -> str:
        """Nettoie le texte pour extraire le JSON pur."""
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        
        start = text.find('[')
        end = text.rfind(']')
        
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]
        
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and start < end:
            return '[' + text[start:end+1] + ']'
        
        return text.strip()

    def _parse_regex_response(self, response_text: str) -> List[Dict]:
        """Fallback: parse avec regex si JSON échoue."""
        quizzes = []
        
        blocks = re.split(r'\n\s*Question\s*\d*\s*[:\)]', response_text, flags=re.IGNORECASE)
        
        for block in blocks[1:]:
            block = block.strip()
            if not block:
                continue
            
            question_match = re.match(r'(.*?)(?:\n|$)(?:Choix|A\)|A\.|A:)', block, re.DOTALL | re.IGNORECASE)
            question = question_match.group(1).strip() if question_match else ""
            
            choices_dict = {}
            for letter in ['A', 'B', 'C', 'D']:
                pattern = rf'{letter}[\)\.\:]\s*([^\n]+?)(?=\s*[B-D][\)\.\:]|\n\s*Réponse|\n\s*Correct|$)'
                match = re.search(pattern, block, re.IGNORECASE)
                if match:
                    choices_dict[letter] = match.group(1).strip()
            
            correct_match = re.search(r'(?:Réponse|Correct)[\s:]*([A-D])', block, re.IGNORECASE)
            correct = correct_match.group(1).upper() if correct_match else ""
            
            if question and len(choices_dict) == 4 and correct:
                quizzes.append({
                    "question": question,
                    "choices": choices_dict,
                    "correct": correct
                })
        
        return quizzes

    def _validate_and_format(self, quizzes: List[Dict], num_questions: int) -> List[Dict]:
        """Valide et formate les questions pour Streamlit avec filtrage qualité."""
        valid_quizzes = []
        seen_questions = set()
        
        # Mots-clés interdits dans les questions
        forbidden_keywords = [
            'auteur', 'author', 'date', 'titre', 'title', 'document',
            'extrait', 'page', 'section', 'chapitre', 'paragraphe',
            'prénom', 'écrit par', 'publié', 'année'
        ]
        
        for q in quizzes:
            if not isinstance(q, dict):
                continue
            
            question = str(q.get("question", "")).strip()
            choices = q.get("choices", {})
            correct = str(q.get("correct", "")).strip().upper()  
            # Validation 1: Structure de base
            if not question or not isinstance(choices, dict) or len(choices) != 4:
                continue
            
            if set(choices.keys()) != {'A', 'B', 'C', 'D'} or correct not in ['A', 'B', 'C', 'D']:
                continue
            
            if not all(str(choices[k]).strip() for k in ['A', 'B', 'C', 'D']):
                continue
            
            # Validation 2: Pas de doublons exacts
            if question in seen_questions:
                self.logger.debug(f"Question dupliquée ignorée: {question[:50]}")
                continue
            
            # Validation 3: Filtrer les questions interdites 
            question_lower = question.lower()
            if any(keyword in question_lower for keyword in forbidden_keywords):
                self.logger.debug(f"Question interdite ignorée: {question[:50]}")
                continue
            
            # Validation 4: Les choix doivent être distincts
            choice_texts = [str(choices[k]).lower().strip() for k in ['A', 'B', 'C', 'D']]
            if len(set(choice_texts)) < 4:
                self.logger.debug(f"Choix non distincts ignorés: {question[:50]}")
                continue
            
            valid_quizzes.append({
                "question": question,
                "choices": [choices["A"], choices["B"], choices["C"], choices["D"]],
                "choice_letters": ["A", "B", "C", "D"],
                "correct": correct, 
                "explanation": q.get("explanation", "")
            })
            
            seen_questions.add(question)
            
            if len(valid_quizzes) >= num_questions:
                break
        
        self.logger.info(f"Validation: {len(valid_quizzes)} questions valides sur {len(quizzes)} générées")
        return valid_quizzes

