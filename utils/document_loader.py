from docx import Document
from email import policy
from email.parser import BytesParser
from utils.pdf_reader import extract_text_from_pdf

def load_text(filepath):
    if filepath.endswith(".pdf"):
        return extract_text_from_pdf(filepath)

    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

    elif filepath.endswith(".eml"):
        with open(filepath, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        body = msg.get_body(preferencelist=('plain',))
        if body:
            return body.get_content()
        return ""

    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()