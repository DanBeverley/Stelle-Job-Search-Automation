import io
import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes."""
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_bytes: bytes) -> str:
    """Extracts text from DOCX bytes."""
    text = ""
    doc = docx.Document(io.BytesIO(docx_bytes))
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text 