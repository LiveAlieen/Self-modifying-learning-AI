import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import joblib

def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        return pd.DataFrame()

def build_model(tfidf_matrix, labels):
    vectorizer = TfidfVectorizer()
    # 假设输入维度为tfidf_matrix.shape[1]
    input_dim = tfidf_matrix.shape[1]
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 假设有10个类别
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 训练模型
    model.fit(tfidf_matrix, labels, epochs=10)
    return model, vectorizer

# 更新模型的函数，需要传入新的对话数据和vectorizer
def update_model(model, vectorizer, new_dialogue):
    # 假设 new_dialogue 是一个包含新对话的 DataFrame
    # 将新对话文本转换为TF-IDF特征
    tfidf_new = vectorizer.transform(new_dialogue['text'])
    new_labels = new_dialogue['response'].values
    # 使用新数据进行进一步的训练
    model.fit(tfidf_new, new_labels, epochs=5)
    return model

# 加载模型
def load_model(model_path):
    return joblib.load(model_path)

# 保存模型
def save_model(model, model_path):
    joblib.dump(model, model_path)

# 加载历史对话数据并转换为TF-IDF特征
def prepare_data():
    old_dialogues = load_dataset("old_dialogues.csv")
    texts = old_dialogues['text'].values
    responses = old_dialogues['response'].values
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, responses

# 主程序逻辑
if __name__ == "__main__":
    # 准备数据
    tfidf_matrix, labels = prepare_data()
    
    # 构建模型
    model, vectorizer = build_model(tfidf_matrix, labels)
    
    # 假设您接收到了新的对话数据 new_dialogues，每个对话都包含在一个 DataFrame 中
    # 然后您可以使用 update_model 函数来在线更新模型
    new_dialogues = load_dataset("new_dialogues.csv")
    if not new_dialogues.empty:
        model = update_model(model, vectorizer, new_dialogues)
    
    # 保存训练好的模型到文件中
    model_path = "model.pkl"
    save_model(model, model_path)
    
    # 加载模型
    loaded_model = load_model(model_path)