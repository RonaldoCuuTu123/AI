import pandas as pd
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Tải biến môi trường
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("Vui lòng thiết lập biến môi trường GEMINI_API_KEY trong file .env")
    exit(1)

genai.configure(api_key=API_KEY)

def generate_augmented_questions(original_question, context_answer):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
Bạn là một trợ lý AI phân tích dữ liệu văn bản. Dựa trên câu trả lời gốc dưới đây, hãy tạo ra 10 biến thể câu hỏi khác nhau bằng tiếng Việt.
Hãy đa dạng hóa các biến thể câu hỏi theo nhiều phong cách: 
- Câu hỏi nghiêm túc, chuẩn mực.
- Câu hỏi thân mật, giao tiếp hàng ngày.
- Câu hỏi sử dụng từ lóng, viết tắt phổ biến trong bóng đá (ví dụ: bú cúp, vđ, cúp c1, epl, nha, man đỏ, tbn...).

Câu trả lời gốc: "{context_answer}"
Câu hỏi gốc: "{original_question}"

Hãy trả về kết quả dưới dạng một danh sách JSON hợp lệ (không chứa markdown nào khác), ví dụ:
[
  "biến thể 1",
  "biến thể 2"
]
"""
    try:
        response = model.generate_content(prompt)
        # Loại bỏ markdown code block nếu có
        text = response.text.replace('```json', '').replace('```', '').strip()
        variants = json.loads(text)
        if isinstance(variants, list):
            return variants
    except Exception as e:
        print(f"Lỗi khi gọi API cho câu hỏi '{original_question}': {e}")
    
    return []

def augment_data():
    base_file = 'data/qa_base.csv'
    if not os.path.exists(base_file):
        # Nếu chưa có file base, thử dùng file qa_train.csv làm base
        base_file = 'data/qa_train.csv'
        if not os.path.exists(base_file):
            print(f"Không tìm thấy file base data tại {base_file}")
            return
            
    print(f"Đọc dữ liệu từ {base_file}...")
    df = pd.read_csv(base_file)
    
    expanded_data = []
    
    for index, row in df.iterrows():
        question = row['Question']
        answer = row['Context_Answer']
        topic = row['Topic']
        
        # Thêm câu hỏi gốc
        expanded_data.append({
            "Question": question,
            "Context_Answer": answer,
            "Topic": topic
        })
        
        print(f"Đang xử lý dòng {index + 1}/{len(df)}: {question}")
        variants = generate_augmented_questions(question, answer)
        
        for variant in variants:
            expanded_data.append({
                "Question": variant,
                "Context_Answer": answer,
                "Topic": topic
            })
            
        time.sleep(2)  # Tránh rate limit
        
    augmented_df = pd.DataFrame(expanded_data)
    
    output_file = 'data/qa_train_augmented.csv'
    augmented_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nHoàn tất! Đã lưu file augmented data tại {output_file} với {len(augmented_df)} bản ghi.")

if __name__ == "__main__":
    augment_data()
