import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from preprocess import preprocess_text
from sentence_transformers import SentenceTransformer

def train_models():
    # 1. Load Data
    data_path = 'data/qa_base_massive.csv'
    if not os.path.exists(data_path):
        data_path = '../data/qa_base_massive.csv'
    if not os.path.exists(data_path):
        print(f"File {data_path} không tồn tại. Vui lòng chạy generate_data.py trước.")
        return
        
    df = pd.read_csv(data_path)
    print("Đã tải dữ liệu thành công.")

    # 2. Preprocess text (cho TF-IDF)
    print("Đang tiền xử lý văn bản...")
    df['Processed_Question'] = df['Question'].apply(preprocess_text)
    
    # 3. Vectorization (TF-IDF dành riêng cho Naive Bayes)
    print("Đang huấn luyện TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_sparse = vectorizer.fit_transform(df['Processed_Question'])
    
    # 4. Intent Classification Model (Naive Bayes)
    print("Đang huấn luyện mô hình Naive Bayes (Phân loại Topic)...")
    y = df['Topic']
    nb_model = MultinomialNB()
    nb_model.fit(X_sparse, y)
    
    # 5. Semantic Search Model (Dense Embeddings với SentenceTransformers)
    print("Đang tải mô hình SentenceTransformer và tạo Dense Embeddings...")
    # paraphrase-multilingual-MiniLM-L12-v2 hỗ trợ tiếng Việt rất tốt
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # Sử dụng câu hỏi nguyên bản (raw text) vì mô hình đã được train trên raw text
    # Thêm batch_size=4 để tiết kiệm tối đa RAM (tránh lỗi Out of Memory)
    X_dense = embedder.encode(df['Question'].tolist(), batch_size=4, show_progress_bar=True)
    
    print("Đang huấn luyện mô hình KNN với Dense Embeddings...")
    knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn_model.fit(X_dense)
    
    # 6. Save models and data for inference
    models_dir = 'models'
    if not os.path.exists(data_path.replace('data/qa_train.csv', 'models')):
        models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(models_dir, 'nb_model.pkl'), 'wb') as f:
        pickle.dump(nb_model, f)
        
    with open(os.path.join(models_dir, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(knn_model, f)
        
    # Lưu dataframe đã process để lấy context
    df.to_pickle(os.path.join(models_dir, 'processed_df.pkl'))
    
    print("Huấn luyện hoàn tất và đã lưu các mô hình vào thư mục 'models/'.")

if __name__ == "__main__":
    train_models()
