import pandas as pd
import pickle
import os
from src.preprocess import preprocess_text
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self):
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
                
            print("Đang khởi tạo mô hình AI Ngôn Ngữ (Semantic Search)...")
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.is_ready = True
        except FileNotFoundError:
            print("Chưa tìm thấy mô hình. Vui lòng chạy train.py trước.")
            self.is_ready = False

    def retrieve_context(self, query, top_k=3):
        if not self.is_ready:
            return [], None
            
        # 1. Phân loại Topic bằng TF-IDF + Naive Bayes
        processed_query = preprocess_text(query)
        X_sparse = self.vectorizer.transform([processed_query])
        predicted_topic = self.nb_model.predict(X_sparse)[0]
        
        # 2. Tìm kiếm ngữ nghĩa bằng Dense Embeddings + KNN
        X_dense = self.embedder.encode([query])
        distances, indices = self.knn_model.kneighbors(X_dense, n_neighbors=top_k)
        
        contexts = [self.df.iloc[idx]['Context_Answer'] for idx in indices[0]]
        return contexts, predicted_topic

    def generate_answer(self, query, history=None):
        if not self.is_ready:
            return {"answer": "Lỗi: Hệ thống chưa sẵn sàng. Vui lòng chạy train.py", "retrieved_context": [], "detected_topic": "Error"}

        # 1. Truy xuất dữ liệu nội bộ bằng Semantic Search (Dense + KNN)
        contexts, topic = self.retrieve_context(query)
        
        # 2. Định dạng câu trả lời thô trực tiếp từ CSV
        if not contexts:
            answer = "Hệ thống không tìm thấy dữ liệu nội bộ phù hợp cho câu hỏi này."
        else:
            answer = "Dựa trên mô hình tìm kiếm ngữ nghĩa (Semantic Search), đây là các kết quả gần nhất:\n\n"
            for i, ctx in enumerate(contexts, 1):
                answer += f"{i}. {ctx}\n"

        return {
            "answer": answer.strip(),
            "retrieved_context": contexts,
            "detected_topic": topic or "General"
        }
