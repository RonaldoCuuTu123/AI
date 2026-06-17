import re
from pyvi import ViTokenizer

def clean_text(text):
    """
    Làm sạch văn bản cơ bản
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Loại bỏ dấu câu
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Tiền xử lý văn bản tiếng Việt: làm sạch và tách từ bằng pyvi
    """
    cleaned = clean_text(text)
    # Tách từ tiếng Việt
    tokenized = ViTokenizer.tokenize(cleaned)
    return tokenized
