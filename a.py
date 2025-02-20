# 环境配置：需预先安装 pip install tensorflow keras numpy matplotlib
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing import sequence

# 超参数设置
max_features = 10000  # 保留最高频的10000个词汇
max_len = 500         # 截断/填充评论长度为500词
batch_size = 128
embedding_dims = 128  # 词向量维度
epochs = 5

# 加载预处理数据集（自动处理文本向量化）
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(f"训练样本数：{len(x_train)}，测试样本数：{len(x_test)}")

# 序列填充（统一输入长度）
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# 构建双向LSTM模型（文献[6][7]推荐架构）
model = Sequential([
    Embedding(max_features, embedding_dims, input_length=max_len),
    tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(LSTM(32)),
    Dense(1, activation='sigmoid')
])

# 模型编译与训练
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2)

# 模型评估
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n测试准确率：{test_acc:.4f}")

# 可视化训练过程（需安装matplotlib）
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型训练过程')
plt.ylabel('准确率')
plt.xlabel('训练轮次')
plt.legend()
plt.show()
plt.savefig('a.train.png')  # 保存训练过程图像