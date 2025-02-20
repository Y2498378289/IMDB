# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# 超参数配置（文献[1][4]）
MAX_LEN = 500
VOCAB_SIZE = 20000
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10

# 持久化路径
MODEL_PATH = 'work/imdb_sentiment_model.h5'
VECTORIZER_PATH = 'work/text_vectorizer'


def load_imdb_data(data_dir):
    """
    加载IMDB原始数据（文献[3]规范实现）
    参数：
        data_dir: 数据集根目录（必须包含train/test子目录）
    返回：
        ((train_texts, train_labels), (test_texts, test_labels))
    """

    def load_from_dir(folder):
        texts, labels = [], []
        for sentiment in ['pos', 'neg']:
            dir_path = os.path.join(folder, sentiment)
            for filename in os.listdir(dir_path):
                if not filename.endswith('.txt'):
                    continue
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1 if sentiment == 'pos' else 0)
        return texts, np.array(labels)

    return (
        load_from_dir(os.path.join(data_dir, 'train')),
        load_from_dir(os.path.join(data_dir, 'test'))
    )


def preprocess_text(text):
    """
    文本预处理流水线（文献[1][5]最佳实践）
    参数：
        text: 原始评论文本
    返回：
        标准化后的文本
    """
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    # 保留基础标点并添加空格隔离
    text = re.sub(r"([.!?,'/()])", r' \1 ', text)
    # 去除非字母符号
    text = re.sub(r'[^a-zA-Z.!?,\']', ' ', text)
    # 合并连续空格并转小写
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def build_vectorizer(train_texts):
    """
    构建文本向量化层（文献[4]实现）
    参数：
        train_texts: 训练文本列表
    返回：
        适配完成的TextVectorization层
    """
    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LEN,
        output_mode='int'
    )
    vectorizer.adapt(tf.convert_to_tensor(train_texts))
    return vectorizer


def build_model(vectorizer):
    """
    构建深度学习模型（文献[4]改进架构）
    参数：
        vectorizer: 已适配的文本向量化层
    返回：
        编译完成的Keras模型
    """
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


def plot_training_history(history):
    """
    可视化训练过程（文献[7]推荐格式）
    参数：
        history: 训练历史对象
    """
    plt.figure(figsize=(12, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练集')
    plt.plot(history.history['val_accuracy'], label='验证集')
    plt.title('模型准确率趋势')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练集')
    plt.plot(history.history['val_loss'], label='验证集')
    plt.title('模型损失趋势')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()

    plt.tight_layout()
    plt.savefig('c.train.png')
    plt.show()


def save_pipeline(model, vectorizer):
    """
    保存完整处理流水线（文献[6]最佳实践）
    参数：
        model: 训练好的模型
        vectorizer: 文本向量化层
    """
    # 保存完整模型（包含预处理层）
    model.save(MODEL_PATH)
    # 单独保存向量化层（用于后续服务化）
    tf.saved_model.save(vectorizer, VECTORIZER_PATH)


def load_pipeline():
    """
    加载完整处理流水线
    返回：
        model: 预训练模型
        vectorizer: 文本向量化器
    """
    model = load_model(MODEL_PATH, custom_objects={'TextVectorization': TextVectorization})
    vectorizer = tf.saved_model.load(VECTORIZER_PATH)
    return model, vectorizer


if __name__ == "__main__":
    # === 数据准备阶段 ===
    # 加载原始数据
    (train_texts, train_labels), (test_texts, test_labels) = load_imdb_data("aclImdb")

    # 数据预处理
    train_texts = [preprocess_text(t) for t in train_texts]
    test_texts = [preprocess_text(t) for t in test_texts]

    # 数据集分割
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=0.2,
        stratify=train_labels
    )

    # === 模型构建阶段 ===
    # 构建文本向量化层
    vectorizer = build_vectorizer(train_texts)

    # 构建模型
    model = build_model(vectorizer)

    # === 模型训练阶段 ===
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    history = model.fit(
        tf.convert_to_tensor(train_texts, dtype=tf.string),
        tf.convert_to_tensor(train_labels, dtype=tf.float32),
        validation_data=(
            tf.convert_to_tensor(val_texts, dtype=tf.string),
            tf.convert_to_tensor(val_labels, dtype=tf.float32)
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(patience=3), checkpoint]
    )

    # === 后处理阶段 ===
    # 可视化训练过程
    plot_training_history(history)

    # 保存完整流水线
    save_pipeline(model, vectorizer)

    # === 模型评估 ===
    # 加载最佳模型
    best_model, vectorizer = load_pipeline()

    # 测试集评估
    test_loss, test_acc, test_precision, test_recall = best_model.evaluate(
        tf.convert_to_tensor(test_texts, dtype=tf.string),
        tf.convert_to_tensor(test_labels, dtype=tf.float32)
    )

    print(f"\n测试准确率：{test_acc:.4f}")
    print(f"F1 Score：{2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

    # === 使用示例 ===
    sample_texts = [
        "This movie is a masterpiece! The acting is superb and the plot is captivating.",
        "Terrible waste of time. Poor acting and nonsensical storyline."
    ]
    preprocessed_texts = [preprocess_text(t) for t in sample_texts]
    predictions = best_model.predict(tf.convert_to_tensor(preprocessed_texts, dtype=tf.string))

    print("\n预测示例：")
    for text, prob in zip(sample_texts, predictions.flatten()):
        print(f"'{text[:50]}...' => {'正面' if prob > 0.5 else '负面'} (置信度: {prob:.4f})")