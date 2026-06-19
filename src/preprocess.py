import re
from pyvi import ViTokenizer

SLANG_DICT = {
    r'\bc1\b': 'Champions League',
    r'\bnha\b': 'Ngoại hạng Anh',
    r'\bepl\b': 'Ngoại hạng Anh',
    r'\bmu\b': 'Manchester United',
    r'\bm\.u\b': 'Manchester United',
    r'\bman đỏ\b': 'Manchester United',
    r'\bcr7\b': 'Cristiano Ronaldo',
    r'\btbn\b': 'Tây Ban Nha',
    r'\bck\b': 'chung kết',
    r'\bvđ\b': 'vô địch',
    r'\bhlv\b': 'huấn luyện viên',
    r'\bm10\b': 'Lionel Messi',
    r'\bwc\b': 'World Cup',
}

def normalize_text(text):
    """
    Chuẩn hóa văn bản: chuyển thành chữ thường và thay thế từ lóng/viết tắt
    """
    text = str(text).lower()
    for slang, standard in SLANG_DICT.items():
        text = re.sub(slang, standard.lower(), text)
    return text

def clean_text(text):
    """
    Làm sạch văn bản cơ bản sau khi đã chuẩn hóa
    """
    text = re.sub(r'[^\w\s]', ' ', text)  # Loại bỏ dấu câu
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Tiền xử lý văn bản tiếng Việt: chuẩn hóa từ lóng, làm sạch và tách từ bằng pyvi
    """
    normalized = normalize_text(text)
    cleaned = clean_text(normalized)
    # Tách từ tiếng Việt
    tokenized = ViTokenizer.tokenize(cleaned)
    return tokenized
