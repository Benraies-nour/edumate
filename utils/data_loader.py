# utils/data_loader.py
import os
import requests
import zipfile
import io
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fonctions d'extraction de texte

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """
    Extrait le texte d'un fichier PDF en utilisant pdfplumber 
    pour une meilleure précision sur les documents pédagogiques.
    """
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Si pdfplumber n'a rien trouvé, on tente un dernier recours avec PyPDF2
        if not text.strip():
            logging.warning(f"pdfplumber n'a rien extrait de {file_path}, tentative avec PyPDF2...")
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())

        logging.info(f"Texte extrait de PDF: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction PDF {file_path}: {e}")
        return None

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Word DOCX."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs if para.text)
        logging.info(f"Texte extrait de DOCX: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction DOCX {file_path}: {e}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier texte brut."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        logging.info(f"Texte extrait de TXT: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction TXT {file_path}: {e}")
        return None

def extract_text_from_csv(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier CSV (convertit en string)."""
    try:
        import pandas as pd
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception:
             df = pd.read_csv(file_path, sep=';', encoding='latin1')

        text = df.to_string()
        logging.info(f"Texte extrait de CSV: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction CSV {file_path}: {e}")
        return None

def extract_text_from_excel(file_path: str) -> Optional[str]:
    """Extrait le texte d'un fichier Excel (convertit en string)."""
    try:
        import pandas as pd
        df = pd.read_excel(file_path, sheet_name=None)
        text = ""
        for sheet_name, sheet_df in df.items():
            text += f"--- Feuille: {sheet_name} ---\n{sheet_df.to_string()}\n\n"
        logging.info(f"Texte extrait de Excel: {file_path} ({len(text)} caractères)")
        return text
    except Exception as e:
        logging.error(f"Erreur extraction Excel {file_path}: {e}")
        return None

# --- Fonctions de chargement ---

def load_and_parse_files(input_dir: str) -> List[Dict[str, any]]:
    """Charge et parse récursivement les fichiers d'un répertoire."""
    documents = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logging.error(f"Le répertoire '{input_dir}' n'existe pas.")
        return []

    for file_path in input_path.rglob("*.*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            text = None

            if ext == ".pdf":
                text = extract_text_from_pdf(str(file_path))
            elif ext == ".docx":
                text = extract_text_from_docx(str(file_path))
            elif ext == ".txt":
                text = extract_text_from_txt(str(file_path))
            elif ext == ".csv":
                text = extract_text_from_csv(str(file_path))
            elif ext in [".xlsx", ".xls"]:
                text = extract_text_from_excel(str(file_path))
            
            if text and text.strip():
                documents.append({
                    "page_content": text,
                    "metadata": {
                        "filename": file_path.name,
                        "source": str(file_path.relative_to(input_path))
                    }
                })
    return documents