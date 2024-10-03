import os
from PyPDF2 import PdfReader
import re 

def extract_text(file_path) -> str:
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    elif file_extension == '.pdf':
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def normalize_text(text) -> str:
    return text.lower()

def segment_paragraphs(text) -> list:
    return re.split(r'\n\s*\n', text.strip())

def preprocess_document(file_path)-> list:
    raw_text = extract_text(file_path)
    normalized_text = normalize_text(raw_text)
    paragraphs = segment_paragraphs(normalized_text)
    return paragraphs


