import pandas as pd
import pickle
import os
import re
import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.preprocess import preprocess_text
from src.football_api import get_fixtures_by_date, get_live_fixtures

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = None

    def retrieve_context(self, query, top_k=3):
        if not self.is_ready:
            return [], None
        processed_query = preprocess_text(query)
        X_query = self.vectorizer.transform([processed_query])
        predicted_topic = self.nb_model.predict(X_query)[0]
        distances, indices = self.knn_model.kneighbors(X_query, n_neighbors=top_k)
        contexts = [self.df.iloc[idx]['Context_Answer'] for idx in indices[0]]
        return contexts, predicted_topic

    def rewrite_query(self, query, history):
        if not history or self.client is None:
            return query
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        prompt = f"""Viết lại câu hỏi mới thành một câu hỏi độc lập đầy đủ ý nghĩa (giải quyết đại từ 'anh ấy', 'họ', 'đội đó').
Nếu đã rõ ràng, giữ nguyên. Chỉ trả về câu hỏi viết lại.

Lịch sử:
{history_str}

Câu hỏi mới: {query}
Câu hỏi viết lại:"""
        try:
            response = self.client.models.generate_content(
                model='gemini-flash-lite-latest', contents=prompt)
            return response.text.strip() or query
        except:
            return query

    def _extract_date_intent(self, query_lower):
        """
        Phân tích câu hỏi và trả về (intent, target_date_iso).
        intent: 'live' | 'fixture' | None
        """
        now = datetime.datetime.now()

        # Đang diễn ra
        if any(kw in query_lower for kw in ['đang đá', 'đang diễn ra', 'live', 'trực tiếp', 'đang thi đấu']):
            return 'live', None

        # Ngày cụ thể dạng DD/MM hoặc DD/MM/YYYY
        date_match = re.search(r'(\d{1,2})/(\d{1,2})(?:/(\d{4}))?', query_lower)
        if date_match:
            d, m = int(date_match.group(1)), int(date_match.group(2))
            y = int(date_match.group(3)) if date_match.group(3) else now.year
            try:
                target = datetime.datetime(y, m, d)
                return 'fixture', target.strftime("%Y-%m-%d")
            except:
                pass

        # N ngày trước
        n_match = re.search(r'(\d+)\s*ngày\s*trước', query_lower)
        if n_match:
            n = int(n_match.group(1))
            return 'fixture', (now - datetime.timedelta(days=n)).strftime("%Y-%m-%d")

        # Ngày mai
        if any(kw in query_lower for kw in ['ngày mai', 'tomorrow']):
            return 'fixture', (now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        # Hôm qua
        if any(kw in query_lower for kw in ['hôm qua', 'yesterday', 'hôm trước']):
            return 'fixture', (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        # Hôm nay / lịch / kết quả / trận đấu
        if any(kw in query_lower for kw in ['hôm nay', 'today', 'lịch', 'kết quả', 'trận']):
            return 'fixture', now.strftime("%Y-%m-%d")

        return None, None

    def _get_api_football_data(self, intent, date_iso):
        """
        Luôn ưu tiên API Football cho mọi câu hỏi về lịch/kết quả.
        Trả về chuỗi mô tả dữ liệu thực tế (hoặc thông báo lỗi rõ ràng).
        """
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
                # API Football không có dữ liệu cho ngày này (quá cũ hoặc quá xa)
                return f"[API Football] Không có dữ liệu cho ngày {date_iso}: {result.get('error', 'Unknown error')}", False
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
        if self.client is None:
            return {"answer": "Lỗi: Chưa cấu hình GEMINI_API_KEY.", "retrieved_context": [], "detected_topic": "Error"}

        if history is None:
            history = []

        rewritten_query = self.rewrite_query(query, history)
        query_lower = rewritten_query.lower()

        # 1. Luôn thử API Football trước cho mọi câu hỏi về lịch/kết quả
        intent, date_iso = self._extract_date_intent(query_lower)
        api_data = ""
        api_success = False

        if intent:
            api_data, api_success = self._get_api_football_data(intent, date_iso)
            print(f"[Pipeline] intent='{intent}', date='{date_iso}', api_success={api_success}")

        # 2. RAG context cho kiến thức lịch sử / background
        contexts, topic = self.retrieve_context(rewritten_query)
        context_str = "\n".join([f"- {c}" for c in contexts]) if contexts else ""
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        current_date = datetime.datetime.now().strftime("%d/%m/%Y")

        # 3. Xây dựng prompt theo tình huống
        if api_success and api_data and "[API Football] Không có dữ liệu" not in api_data and "Lỗi" not in api_data:
            # Có dữ liệu thực từ API → Gemini chỉ đóng vai trình bày và phân tích
            system_role = """Bạn là AI Agent Bóng đá chuyên nghiệp. Nhiệm vụ của bạn là trình bày dữ liệu thực tế 
một cách đẹp mắt, rõ ràng, và có thể thêm phân tích/bình luận chuyên sâu dựa trên thông tin đã cung cấp.
KHÔNG được bịa thêm tên đội hay tỷ số nào ngoài dữ liệu API Football đã cung cấp."""
        elif intent and not api_success:
            # API Football không có dữ liệu (ngày quá cũ/mới) → Gemini dùng kiến thức lịch sử + RAG, KHÔNG bịa số liệu
            system_role = """Bạn là AI Agent Bóng đá chuyên nghiệp. API Football không có dữ liệu cho yêu cầu này.
Hãy trả lời dựa trên kiến thức lịch sử bóng đá và ngữ cảnh nội bộ của bạn.
Nếu thực sự không có thông tin, hãy thành thật nói rằng không có dữ liệu cho ngày đó và gợi ý câu hỏi khác."""
        else:
            # Câu hỏi không liên quan đến lịch/kết quả → Gemini trả lời tự do từ kiến thức + RAG
            system_role = """Bạn là AI Agent Bóng đá chuyên nghiệp với kiến thức sâu rộng về lịch sử bóng đá,
cầu thủ, chiến thuật, và các giải đấu. Hãy trả lời câu hỏi một cách chi tiết, chính xác và hấp dẫn."""

        prompt = f"""{system_role}

Hôm nay: {current_date} (Năm 2026 - World Cup 2026 đang diễn ra)

LỊCH SỬ TRÒ CHUYỆN:
{history_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 NGỮ CẢNH NỘI BỘ (Ưu tiên 1 - Data đã được train sẵn):
{context_str if context_str else "Không có dữ liệu nội bộ phù hợp."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ DỮ LIỆU THỰC TẾ TỪ API FOOTBALL (Ưu tiên 1 - cho lịch/kết quả):
{api_data if api_data else "Không có dữ liệu API Football cho câu hỏi này."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC SỬ DỤNG DỮ LIỆU:
- Câu hỏi về LỊCH / KẾT QUẢ / LIVE → LUÔN dùng dữ liệu API Football ở trên, KHÔNG bịa thêm.
- Câu hỏi về LỊCH SỬ / CẦU THỦ / KIẾN THỨC → Ưu tiên ngữ cảnh nội bộ (data đã train).
- Nếu cả hai nguồn đều không có → Dùng kiến thức chung, thành thật nếu không chắc chắn.

CÂU HỎI: {query}

Trả lời bằng tiếng Việt, rõ ràng, đẹp mắt với emoji phù hợp."""

        try:
            response = self.client.models.generate_content(
                model='gemini-flash-lite-latest',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4)
            )
            return {
                "answer": response.text,
                "retrieved_context": contexts,
                "detected_topic": topic or "General"
            }
        except Exception as e:
            return {
                "answer": f"Lỗi Gemini: {str(e)}",
                "retrieved_context": contexts,
                "detected_topic": topic or "Error"
            }
