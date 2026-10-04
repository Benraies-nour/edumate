import logging
import os
from typing import List, Dict
import streamlit as st

from utils.llm import chat

# --- IMPORTS DES MODULES DECLARES ---
# On suppose que vos fichiers sont dans un dossier "utils"
from utils.data_loader import load_and_parse_files
from utils.chunker import chunk_documents
from utils.preprocessor import clean_text

logging.basicConfig(level=logging.INFO)

# --- Classe Summarizer ---

class Summarizer:
    def __init__(self):
        # Gros chunks : moins d'appels LLM (quota Groq) et plus de contexte par extrait
        self.chunk_size = 6000
        self.chunk_overlap = 300
        self.max_merged_length = 12000
       
    def summarize_text_bullets(self, text: str, length="moyen") -> str:
        """Étape 1 : Extraction des points clés"""
        
        detail_map = {
            "court": "Extrayez uniquement les idées maîtresses, soyez très bref.",
            "moyen": "Extrayez les points importants et les définitions clés.",
            "long": "Extrayez TOUS les détails, chiffres, exemples et définitions techniques. Soyez exhaustif."
        }
        
        prompt = f"""
Vous êtes un analyste expert. Analysez le texte technique ci-dessous.
Produisez une liste de points clés (bullet points) en français.

Consignes strictes :
1. Ne faites pas de phrases d'introduction (type "Ce texte parle de..."). Allez droit au but.
2. {detail_map.get(length, detail_map['moyen'])}

Texte : {text}
"""
        try:
            return chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            ).strip()
        except Exception as e:
            st.error(f"Erreur LLM (Bullets): {e}")
            logging.error(f"Erreur LLM (Bullets): {e}")
            return ""

    def summarize_text_paragraph(self, bullets_text: str, length="moyen") -> str:
        """Étape 2 : Résumé final structuré"""
        
        detail_map = {
            "court": "Rédige un paragraphe unique, concis et efficace (environ 100 mots).",
            "moyen": "Rédige un résumé clair et fluide. Organise les idées logiquement.",
            "long": """
            Rédige un **Rapport Complet et Structuré**.
            1. Utilise des **Titres** pour séparer les sections.
            2. Développe chaque concept en profondeur.
            3. ATTENTION : Les notes contiennent des doublons. **FUSIONNE** les informations répétées. Ne te répète jamais.
            4. Fais une introduction et une conclusion claires.
            """
        }
        
        predict_limit = 3000 if length == "long" else 1500
        
        prompt = f"""
Vous êtes un rédacteur professionnel.
Voici des notes brutes extraites d'un document (il peut y avoir des répétitions).
Votre tâche : Rédiger le résumé final en français.

Consignes :
{detail_map.get(length, detail_map['moyen'])}

Notes brutes :
{bullets_text}
"""
        try:
            return chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=predict_limit,
            ).strip()
        except Exception as e:
            st.error(f"Erreur LLM (Final): {e}")
            logging.error(f"Erreur LLM (Final): {e}")
            return bullets_text[:500] + "..."

    def summarize_documents(self, input_dir: str, length="moyen") -> List[Dict]:
        """
        Orchestre le chargement, le nettoyage, le découpage et le résumé.
        Utilise : load_and_parse_files, clean_text, chunk_documents.
        """
        # 1. Utilisation du Data Loader externe
        documents = load_and_parse_files(input_dir)
        
        if not documents:
            st.warning("Aucun document lu ou dossier vide.")
            return []

        summarized_documents = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_docs = len(documents)

        for idx, doc in enumerate(documents):
            filename = doc['metadata']['filename']
            status_text.text(f"Traitement de {filename} ({length})...")
            
            try:
                # 2. Utilisation du Preprocessor (clean_text)
                raw_content = doc.get('page_content', '')
                if not raw_content:
                    continue
                    
                cleaned_content = clean_text(raw_content)
                # On met à jour le contenu du doc avec le texte propre
                doc['page_content'] = cleaned_content

                # 3. Utilisation du Chunker externe
                chunks = chunk_documents(
                    [doc], 
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
                
                if not chunks: 
                    logging.warning(f"Aucun chunk généré pour {filename}")
                    continue

                # 4. Processus de résumé (interne à la classe Summarizer)
                chunk_bullets = []
                for chunk in chunks:
                    text_segment = chunk.get("content", "")
                    if text_segment:
                        res = self.summarize_text_bullets(text_segment, length=length)
                        if res: chunk_bullets.append(res)
                
                if not chunk_bullets: 
                    continue

                merged_bullets = "\n".join(chunk_bullets)
                
                # Gestion de la taille max pour le contexte du LLM final
                if len(merged_bullets) > self.max_merged_length:
                    merged_bullets = merged_bullets[:self.max_merged_length]
                    logging.warning(f"Résumé intermédiaire tronqué pour {filename}")

                final_summary = self.summarize_text_paragraph(merged_bullets, length=length)
                
                summarized_documents.append({
                    "summary": final_summary, 
                    "metadata": doc["metadata"]
                })
                
                # Mise à jour barre de progression
                progress_bar.progress((idx + 1) / total_docs)
                
            except Exception as e:
                st.error(f"Erreur critique sur {filename}: {e}")
                logging.error(f"Erreur critique sur {filename}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        return summarized_documents