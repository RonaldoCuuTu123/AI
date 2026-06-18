import pandas as pd
import pickle
import os
import re
import datetime
import numpy as np
from dotenv import load_dotenv
from src.preprocess import preprocess_text
from src.football_api import get_fixtures_by_date, get_live_fixtures

load_dotenv()

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
            return [], None, []
        processed_query = preprocess_text(query)
        X_query = self.vectorizer.transform([processed_query])
        predicted_topic = self.nb_model.predict(X_query)[0]
        distances, indices = self.knn_model.kneighbors(X_query, n_neighbors=top_k)
        contexts = [self.df.iloc[idx]['Context_Answer'] for idx in indices[0]]
        return contexts, predicted_topic, distances[0]

    def rewrite_query(self, query, history):
        return query

    def _extract_date_intent(self, query_lower):
        now = datetime.datetime.now()
        if any(kw in query_lower for kw in ['đang đá', 'đang diễn ra', 'live', 'trực tiếp', 'đang thi đấu']):
            return 'live', None
        date_match = re.search(r'(\d{1,2})/(\d{1,2})(?:/(\d{4}))?', query_lower)
        if date_match:
            d, m = int(date_match.group(1)), int(date_match.group(2))
            y = int(date_match.group(3)) if date_match.group(3) else now.year
            try:
                target = datetime.datetime(y, m, d)
                return 'fixture', target.strftime("%Y-%m-%d")
            except:
                pass
        n_match = re.search(r'(\d+)\s*ngày\s*trước', query_lower)
        if n_match:
            n = int(n_match.group(1))
            return 'fixture', (now - datetime.timedelta(days=n)).strftime("%Y-%m-%d")
        if any(kw in query_lower for kw in ['ngày mai', 'tomorrow']):
            return 'fixture', (now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if any(kw in query_lower for kw in ['hôm qua', 'yesterday', 'hôm trước']):
            return 'fixture', (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        if any(kw in query_lower for kw in ['hôm nay', 'today', 'lịch thi đấu', 'kết quả', 'trận đấu']):
            return 'fixture', now.strftime("%Y-%m-%d")
        return None, None

    def _get_api_football_data(self, intent, date_iso):
        if intent == 'live':
            result = get_live_fixtures()
            if not result["success"]:
                return f"[API Football] Lỗi: {result['error']}", False
            wc = result.get("wc_live", [])
            major = result.get("major_live", [])
            if wc:
                return "🔴 LIVE World Cup 2026:\n" + "\n".join(wc), True
            elif major:
                return "🔴 LIVE (Giải lớn):\n" + "\n".join(major), True
            else:
                return "[API Football] Hiện không có trận đấu lớn nào đang diễn ra.", True
        elif intent == 'fixture':
            result = get_fixtures_by_date(date_iso)
            if not result["success"]:
                error_msg = result.get('error', '')
                if "Free plans do not have access" in error_msg:
                    return f"[API Football] Bị chặn do gói Miễn phí (Free plan) không cho phép xem lịch ngày {date_iso}.", False
                return f"[API Football] Lỗi: {error_msg}", False
            wc = result.get("wc_fixtures", [])
            major = result.get("major_fixtures", [])
            date_vn = datetime.datetime.strptime(date_iso, "%Y-%m-%d").strftime("%d/%m/%Y")
            if wc:
                return f"🏆 World Cup 2026 ngày {date_vn}:\n" + "\n".join(wc), True
            elif major:
                return f"📅 Các giải lớn ngày {date_vn}:\n" + "\n".join(major), True
            else:
                return f"[API Football] Ngày {date_vn}: Không tìm thấy trận đấu lớn nào.", True
        return "", False

    def generate_answer(self, query, history=None):
        if history is None:
            history = []
        query_lower = query.lower()

        # 1. API Football (Lịch/Kết quả)
        intent, date_iso = self._extract_date_intent(query_lower)
        if intent:
            api_data, api_success = self._get_api_football_data(intent, date_iso)
            if api_success:
                return {
                    "answer": api_data,
                    "retrieved_context": [],
                    "detected_topic": "Football API",
                    "confidence": 100.0
                }

        # 2. RAG context
        contexts, topic, distances = self.retrieve_context(query)
        
        if contexts:
            # Tính độ tin cậy dựa trên Cosine Distance (1 - distance)
            # Giới hạn trong khoảng 0-100
            dist = distances[0]
            confidence = max(0, min(100, (1 - dist) * 100))
            
            return {
                "answer": contexts[0],
                "retrieved_context": contexts,
                "detected_topic": topic or "General",
                "confidence": round(confidence, 1)
            }
        
        return {
            "answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong hệ thống dữ liệu.",
            "retrieved_context": [],
            "detected_topic": "None",
            "confidence": 0.0
        }
