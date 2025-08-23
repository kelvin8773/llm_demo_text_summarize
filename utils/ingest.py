# utils/ingest.py
import docx
from PyPDF2 import PdfReader


def load_document(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = " ".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = " ".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")
    return text
