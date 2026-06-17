import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from preprocess import preprocess_text

def train_models():
    # 1. Load Data
    data_path = 'data/qa_train_augmented.csv'
    # Adjust path if script is run from src directory
    if not os.path.exists(data_path):
        data_path = '../data/qa_train_augmented.csv'
    if not os.path.exists(data_path):
        print(f"File {data_path} không tồn tại. Vui lòng chạy generate_data.py trước.")
        return
        
    df = pd.read_csv(data_path)
    print("Đã tải dữ liệu thành công.")

    # 2. Preprocess text
    print("Đang tiền xử lý văn bản...")
    df['Processed_Question'] = df['Question'].apply(preprocess_text)
    
    # 3. Vectorization (TF-IDF)
    print("Đang huấn luyện TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Processed_Question'])
    
    # 4. Intent Classification Model (Naive Bayes)
    print("Đang huấn luyện mô hình Naive Bayes (Phân loại Topic)...")
    y = df['Topic']
    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    
    # 5. Semantic Search Model (KNN / NearestNeighbors)
    print("Đang huấn luyện mô hình KNN (Truy xuất ngữ cảnh)...")
    # k=3 như yêu cầu, cosine metric cho tf-idf
    knn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn_model.fit(X)
    
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
