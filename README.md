# Football AI RAG Project (Local Version)

Đây là dự án cuối kỳ môn "Nhập môn Trí tuệ Nhân tạo". Hệ thống là một Chatbot tư vấn dữ liệu bóng đá châu Âu, được xây dựng theo kiến trúc Hybrid RAG (Retrieval-Augmented Generation) phiên bản chạy offline/local (không sử dụng LLM).

## Kiến trúc Hệ thống

Dự án này sử dụng các thuật toán AI truyền thống (được xây dựng bằng Scikit-learn) để tìm kiếm và trả về thông tin trực tiếp.

### Bước 1: Retrieval (AI Truyền thống)
Hệ thống sử dụng các thuật toán truyền thống để phân tích câu hỏi và truy xuất ngữ cảnh phù hợp từ cơ sở dữ liệu nội bộ (`data/qa_train.csv`).
- **NLP Preprocessing:** Sử dụng thư viện `pyvi` để tách từ tiếng Việt một cách chính xác.
- **Vectorization:** Sử dụng `TfidfVectorizer`.
- **Phân loại Intent (Topic Classification):** Mô hình `MultinomialNB` dự đoán chủ đề.
- **Truy xuất Ngữ cảnh (Semantic Search):** Mô hình `NearestNeighbors` (KNN) tìm kiếm kết quả phù hợp nhất.

### Bước 2: Direct Response
- Hệ thống trả về trực tiếp kết quả tìm thấy từ cơ sở dữ liệu hoặc từ **API Football** nếu câu hỏi liên quan đến lịch thi đấu/kết quả thực tế.
- Không sử dụng LLM để sinh văn bản, đảm bảo tính chính xác 100% dựa trên nguồn dữ liệu.

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
│   └── rag_pipeline.py     # Code kết hợp KNN và API Football
├── static/
│   ├── style.css           # Giao diện Premium UI
│   └── script.js           # Logic xử lý giao diện chat
├── templates/
│   └── index.html          # Giao diện chính
├── .env                    # File chứa API key (API Football)
├── requirements.txt        # Các thư viện cần thiết
├── README.md               # File tài liệu hướng dẫn
└── app.py                  # Server FastAPI
```

## Hướng dẫn Cài đặt

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Cấu hình file `.env`:**
   Dự án cần **API-Football Key** để xem lịch thi đấu thực tế:
   ```env
   API_FOOTBALL_KEY=dán_api_football_key_cua_ban_vao_day
   ```

3. **Huấn luyện mô hình cơ sở:**
   ```bash
   python src/train.py
   ```

4. **Chạy Server:**
   ```bash
   python app.py
   ```

5. **Sử dụng:**
   Truy cập: [http://127.0.0.1:8888](http://127.0.0.1:8888).
