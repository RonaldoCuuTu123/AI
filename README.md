# Football AI RAG Project

Đây là dự án cuối kỳ môn "Nhập môn Trí tuệ Nhân tạo". Hệ thống là một Chatbot tư vấn dữ liệu bóng đá châu Âu, được xây dựng theo kiến trúc Hybrid RAG (Retrieval-Augmented Generation).

## Kiến trúc Hệ thống

Dự án này kết hợp chặt chẽ giữa các thuật toán AI truyền thống (được xây dựng bằng Scikit-learn) và mô hình ngôn ngữ lớn (LLM) hiện đại.

### Bước 1: Retrieval (AI Truyền thống)
Hệ thống sử dụng các thuật toán truyền thống để phân tích câu hỏi và truy xuất ngữ cảnh phù hợp từ cơ sở dữ liệu nội bộ (`data/qa_train.csv`).
- **NLP Preprocessing:** Sử dụng thư viện `pyvi` để tách từ tiếng Việt một cách chính xác (ví dụ: "Ngoại_hạng_Anh", "Việt_vị").
- **Vectorization:** Sử dụng `TfidfVectorizer` để chuyển đổi văn bản thành vector toán học.
- **Phân loại Intent (Topic Classification):** Mô hình `MultinomialNB` (Naive Bayes) dự đoán chủ đề của câu hỏi (Kết quả, Vua phá lưới, Bảng xếp hạng, Thẻ phạt).
- **Truy xuất Ngữ cảnh (Semantic Search):** Mô hình `NearestNeighbors` (KNN với K=3, đo lường bằng Cosine Similarity) tìm kiếm 3 kết quả phù hợp nhất với câu hỏi.

### Bước 2: Generation (LLM Generation)
- **API:** Tích hợp `google-generativeai` (Gemini API) của Google.
- **Prompt Engineering Khắt khe:** LLM chỉ được phép trả lời dựa trên ngữ cảnh đã được mô hình KNN truy xuất. Nếu ngữ cảnh không chứa thông tin, hệ thống sẽ yêu cầu LLM lịch sự xin lỗi để ngăn chặn hiện tượng ảo giác (Hallucination).

## Cấu trúc Thư mục

```
/football_ai_rag_project
├── data/
│   └── qa_train.csv        # Dữ liệu mẫu tiếng Việt
├── models/                 # Thư mục chứa các mô hình đã huấn luyện (.pkl)
├── src/
│   ├── preprocess.py       # Code tiền xử lý văn bản (pyvi)
│   ├── train.py            # Code huấn luyện mô hình (TF-IDF, Naive Bayes, KNN)
│   ├── evaluate.py         # Code tính toán các chỉ số đánh giá học thuật
│   └── rag_pipeline.py     # Code kết hợp KNN và Gemini API
├── static/
│   ├── style.css           # Giao diện Premium UI (Glassmorphism)
│   └── script.js           # Logic xử lý giao diện chat
├── templates/
│   └── index.html          # Giao diện chính của ứng dụng
├── .env                    # File chứa API key
├── requirements.txt        # Các thư viện cần thiết
├── README.md               # File tài liệu hướng dẫn
└── app.py                  # Server FastAPI
```

## Hướng dẫn Cài đặt

1. **Clone dự án & Cài đặt môi trường:**
   Tạo virtual environment và cài đặt các thư viện cần thiết:
   ```bash
   git clone <link-repo-cua-ban>
   cd football_ai_rag_project
   python -m venv .venv
   .\.venv\Scripts\activate      # Dành cho Windows
   # source .venv/bin/activate    # Dành cho Mac/Linux
   pip install -r requirements.txt
   ```

2. **Đăng ký và lấy API Keys (Quan trọng):**
   Dự án này cần 2 API key để hoạt động đầy đủ tính năng:
   
   * **Gemini API Key (Miễn phí):**
     - Truy cập trang [Google AI Studio](https://aistudio.google.com/app/apikey).
     - Đăng nhập bằng tài khoản Google của bạn.
     - Nhấn nút **"Create API Key"**.
     - Copy đoạn mã API key vừa tạo.
   
   * **API-Football Key (Miễn phí 100 requests/ngày):**
     - Truy cập trang [API-Sports](https://dashboard.api-football.com/register).
     - Đăng ký một tài khoản và đăng nhập vào Dashboard.
     - Ở mục **Account**, bạn sẽ thấy **API Key** của mình. Copy đoạn mã này.

3. **Cấu hình file `.env`:**
   Tạo một file có tên `.env` ở thư mục gốc của dự án (nằm cùng chỗ với file `app.py`). Sau đó dán 2 API keys vừa copy vào theo định dạng sau:
   ```env
   GEMINI_API_KEY=dán_gemini_api_key_cua_ban_vao_day
   API_FOOTBALL_KEY=dán_api_football_key_cua_ban_vao_day
   ```

4. **Huấn luyện mô hình cơ sở (Làm 1 lần duy nhất):**
   Hệ thống cần huấn luyện các mô hình AI truyền thống (TF-IDF, Naive Bayes, KNN) từ file CSV nội bộ:
   ```bash
   python src/train.py
   ```

5. **Chạy Server:**
   Khởi động server backend bằng lệnh:
   ```bash
   python app.py
   ```
   *(Hoặc chạy thông qua uvicorn nếu bạn muốn: `uvicorn app:app --reload`)*

6. **Sử dụng:**
   Mở trình duyệt web và truy cập vào địa chỉ: [http://127.0.0.1:8000](http://127.0.0.1:8000). Giao diện chatbot sẽ hiện ra để bạn trải nghiệm.

## Các tính năng nổi bật để test

- **Hỏi lịch thi đấu & kết quả thực tế:** 
  Hệ thống được thiết kế ưu tiên **API Football** làm nguồn sự thật (Source of Truth). Bạn có thể hỏi *"Lịch World Cup hôm nay?"* hoặc *"Kết quả World Cup hôm qua?"* — hệ thống sẽ gọi API thật và trả về dữ liệu siêu chuẩn xác.
- **Conversational Memory (Nhớ ngữ cảnh trò chuyện):**
  Hỏi câu đầu: *"Ai vô địch World Cup 2022?"*
  Hỏi tiếp (không nhắc lại): *"Họ đã thắng ai ở chung kết?"*
  Hệ thống sẽ tự hiểu "Họ" là Argentina.
- **Phân tích & Dự đoán thông minh:**
  Sức mạnh của Gemini 2.5 Flash cho phép bạn đưa ra các câu hỏi mở như *"Dự đoán đội vô địch Ngoại hạng Anh năm nay"*. Trợ lý sẽ phân tích sâu sắc dựa trên kiến thức hiện có.
