import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess_text

def evaluate_models():
    models_dir = 'models'
    if not os.path.exists('models/processed_df.pkl'):
        models_dir = '../models'
        
    try:
        df = pd.read_pickle(os.path.join(models_dir, 'processed_df.pkl'))
        with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(models_dir, 'nb_model.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
        with open(os.path.join(models_dir, 'knn_model.pkl'), 'rb') as f:
            knn_model = pickle.load(f)
    except FileNotFoundError:
        print("Không tìm thấy mô hình. Vui lòng chạy src/train.py trước.")
        return

    print("--- ĐÁNH GIÁ MÔ HÌNH NAIVE BAYES (Intent Classification) ---")
    X = vectorizer.transform(df['Processed_Question'])
    y_true = df['Topic']
    y_pred = nb_model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\n--- MOCK ĐÁNH GIÁ KNN (Context Retrieval) ---")
    distances, indices = knn_model.kneighbors(X)
    
    correct_retrievals = 0
    total_queries = len(df)
    
    for i in range(total_queries):
        if i in indices[i]:
            correct_retrievals += 1
            
    knn_acc = correct_retrievals / total_queries
    print(f"KNN Retrieval Accuracy (Top 3): {knn_acc:.4f}")
    print("Nhận xét: Vì đánh giá trên tập huấn luyện, KNN dễ dàng đạt tỷ lệ 100% tìm thấy ngữ cảnh chính xác.")

if __name__ == "__main__":
    evaluate_models()
