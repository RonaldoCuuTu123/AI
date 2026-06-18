import pandas as pd
import pickle
import os
from src.preprocess import preprocess_text

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
            self.is_ready = True
        except FileNotFoundError:
            print("Chưa tìm thấy mô hình. Vui lòng chạy train.py trước.")
            self.is_ready = False

    def retrieve_context(self, query, top_k=3):
        if not self.is_ready:
            return [], None
        processed_query = preprocess_text(query)
        X_query = self.vectorizer.transform([processed_query])
        predicted_topic = self.nb_model.predict(X_query)[0]
        distances, indices = self.knn_model.kneighbors(X_query, n_neighbors=top_k)
        contexts = [self.df.iloc[idx]['Context_Answer'] for idx in indices[0]]
        return contexts, predicted_topic

    def generate_answer(self, query, history=None):
        if not self.is_ready:
            return {"answer": "Lỗi: Hệ thống chưa sẵn sàng.", "retrieved_context": [], "detected_topic": "Error"}

        # 1. Truy xuất dữ liệu nội bộ bằng KNN
        contexts, topic = self.retrieve_context(query)
        
        # 2. Định dạng câu trả lời thô trực tiếp từ CSV (không dùng LLM)
        if not contexts:
            answer = "Hệ thống không tìm thấy dữ liệu nội bộ phù hợp cho câu hỏi này."
        else:
            answer = "Dựa trên dữ liệu nội bộ, đây là các kết quả tìm được:\n\n"
            for i, ctx in enumerate(contexts, 1):
                answer += f"{i}. {ctx}\n"

        return {
            "answer": answer.strip(),
            "retrieved_context": contexts,
            "detected_topic": topic or "General"
        }
