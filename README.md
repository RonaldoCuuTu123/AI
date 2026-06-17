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

1. **Cài đặt môi trường:**
   Tạo virtual environment và cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình API Key:**
   Mở file `.env` và điền API key của bạn:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

3. **Tạo dữ liệu và Huấn luyện mô hình:**
   Sinh dữ liệu mẫu:
   ```bash
   python generate_data.py
   ```
   Huấn luyện các mô hình truyền thống (AI Core):
   ```bash
   python src/train.py
   ```

4. **Đánh giá Mô hình (Tuỳ chọn nhưng quan trọng cho chấm điểm):**
   ```bash
   python src/evaluate.py
   ```

5. **Chạy Server Backend:**
   Khởi động server bằng FastAPI/Uvicorn:
   ```bash
   uvicorn app:app --reload
   ```

6. **Sử dụng:**
   Truy cập `http://127.0.0.1:8000` trên trình duyệt để sử dụng Chatbot.
