import pandas as pd
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv
from src.preprocess import preprocess_text

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)

class RAGPipeline:
    def __init__(self):
        # Load models
        models_dir = 'models'
        if not os.path.exists('models/processed_df.pkl'):
            models_dir = '../models'
            
        try:
            self.df = pd.read_pickle(os.path.join(models_dir, 'processed_df.pkl'))
            with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(models_dir, 'nb_model.pkl'), 'rb') as f:
                self.nb_model = pickle.load(f)
            with open(os.path.join(models_dir, 'knn_model.pkl'), 'rb') as f:
                self.knn_model = pickle.load(f)
            self.is_ready = True
        except FileNotFoundError:
            print("Chưa tìm thấy mô hình. Vui lòng chạy train.py trước.")
            self.is_ready = False

        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.generation_model = None

    def retrieve_context(self, query, top_k=3):
        """
        Bước 1: Truy xuất ngữ cảnh bằng KNN
        """
        if not self.is_ready:
            return "Hệ thống chưa sẵn sàng. Thiếu model.", None
            
        processed_query = preprocess_text(query)
        X_query = self.vectorizer.transform([processed_query])
        
        # Dự đoán Topic
        predicted_topic = self.nb_model.predict(X_query)[0]
        
        # KNN search
        distances, indices = self.knn_model.kneighbors(X_query, n_neighbors=top_k)
        
        contexts = []
        for idx in indices[0]:
            contexts.append(self.df.iloc[idx]['Context_Answer'])
            
        return contexts, predicted_topic

    def generate_answer(self, query):
        """
        Bước 2: Sinh câu trả lời bằng Gemini API dựa trên ngữ cảnh khắt khe
        """
        if self.generation_model is None:
            return {
                "answer": "Lỗi: Chưa cấu hình GEMINI_API_KEY. Vui lòng thêm key vào file .env",
                "retrieved_context": [],
                "detected_topic": "Error"
            }

        contexts, topic = self.retrieve_context(query)
        if not isinstance(contexts, list):
            return {
                "answer": contexts,
                "retrieved_context": [],
                "detected_topic": "Error"
            }
            
        context_str = "\n".join([f"- {c}" for c in contexts])
        
        # Prompt engineering linh hoạt (Ưu tiên RAG, fallback kiến thức chung)
        prompt = f"""
Bạn là một Trợ lý Ảo chuyên nghiệp về Dữ liệu Bóng đá Châu Âu.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác.

NGỮ CẢNH:
{context_str}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

QUY TẮC BẮT BUỘC:
1. ƯU TIÊN hàng đầu sử dụng thông tin trong phần "NGỮ CẢNH" để trả lời câu hỏi.
2. Nếu "NGỮ CẢNH" không chứa thông tin liên quan hoặc không đủ thông tin để trả lời câu hỏi của người dùng, bạn hãy sử dụng hiểu biết và kiến thức bóng đá chung của bản thân để trả lời câu hỏi một cách chính xác, khách quan và chuyên nghiệp. Tuyệt đối không từ chối trả lời nếu câu hỏi liên quan đến bóng đá nói chung.
3. Trả lời bằng tiếng Việt, văn phong tự nhiên, thân thiện và súc tích.
"""
        try:
            response = self.generation_model.generate_content(prompt)
            return {
                "answer": response.text,
                "retrieved_context": contexts,
                "detected_topic": topic
            }
        except Exception as e:
            return {
                "answer": f"Lỗi khi gọi API Gemini: {str(e)}",
                "retrieved_context": contexts,
                "detected_topic": topic
            }
