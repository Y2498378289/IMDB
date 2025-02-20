# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 超参数配置[1,4](@ref)
MAX_LEN = 500
VOCAB_SIZE = 20000
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10


def load_imdb_data(data_dir):
    """加载IMDB数据集[5](@ref)"""

    def load_texts_labels(folder):
        texts, labels = [], []
        for label in ['pos', 'neg']:
            path = os.path.join(folder, label)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1 if label == 'pos' else 0)
        return texts, labels

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    return (load_texts_labels(train_dir),
            load_texts_labels(test_dir))


def preprocess_text(text):
    """文本预处理[1,5](@ref)"""
    text = re.sub(r'<[^>]+>', ' ', text)  # 去除HTML标签
    text = re.sub(r"([.!?,'/()])", r' \1 ', text)  # 保留标点
    text = re.sub(r'[^a-zA-Z.!?,\']', ' ', text)  # 去除非字母字符
    text = re.sub(r'\s+', ' ', text).strip().lower()  # 合并空格并转小写
    return text


# 数据加载与预处理
(train_texts, train_labels), (test_texts, test_labels) = load_imdb_data("aclImdb")
train_texts = [preprocess_text(t) for t in train_texts]
test_texts = [preprocess_text(t) for t in test_texts]

# 数据集分割[5](@ref)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels,
    test_size=0.2,
    stratify=train_labels
)

# 文本向量化层[4](@ref)
vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_LEN,
    output_mode='int'
)
vectorizer.adapt(train_texts)

# 构建模型[1,4](@ref)
model = Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    Embedding(VOCAB_SIZE + 1, EMBED_DIM, mask_zero=True),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 模型编译[4](@ref)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# 训练配置[4](@ref)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 模型训练[4](@ref)
history = model.fit(
    tf.convert_to_tensor(train_texts, dtype=tf.string),
    tf.convert_to_tensor(train_labels, dtype=tf.float32),
    validation_data=(
        tf.convert_to_tensor(val_texts, dtype=tf.string),
        tf.convert_to_tensor(val_labels, dtype=tf.float32)
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
)

# 模型评估[4](@ref)
test_loss, test_acc, test_precision, test_recall = model.evaluate(
    tf.convert_to_tensor(test_texts, dtype=tf.string),
    tf.convert_to_tensor(test_labels, dtype=tf.float32),
    batch_size=BATCH_SIZE
)

print(f"\n测试准确率：{test_acc:.4f}")
print(f"F1 Score：{2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# 可视化训练过程
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Trend')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Trend')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('b.train.png')  # 保存训练过程图像