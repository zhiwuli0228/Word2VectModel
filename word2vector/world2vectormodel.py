import jieba
import numpy as np
import random
from collections import Counter


class SimpleWord2Vec:
    def __init__(self, corpus, embedding_dim=10, window_size=2, learning_rate=0.01, epochs=1000, neg_samples=3):
        """
        简易版 Word2Vec (Skip-Gram + Negative Sampling)
        :param corpus: 输入语料（list[str]，每个元素是一句话）
        :param embedding_dim: 词向量维度
        :param window_size: 上下文窗口大小
        :param learning_rate: 学习率
        :param epochs: 训练轮数
        :param neg_samples: 负采样个数
        """
        self.corpus = [list(jieba.cut(sentence)) for sentence in corpus]  # 默认空格分词
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.neg_samples = neg_samples

        # 构建词表
        words = [w for sentence in self.corpus for w in sentence]
        word_count = Counter(words)
        self.vocab = list(word_count.keys())
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # 初始化参数
        self.W1 = np.random.uniform(-0.8, 0.8, (self.vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-0.8, 0.8, (embedding_dim, self.vocab_size))

    def _generate_training_data(self):
        """生成 Skip-Gram 训练数据"""
        training_data = []
        for sentence in self.corpus:
            for i, target_word in enumerate(sentence):
                target_idx = self.word2idx[target_word]
                context = []
                for j in range(max(0, i - self.window_size), min(len(sentence), i + self.window_size + 1)):
                    if j != i:
                        context.append(self.word2idx[sentence[j]])
                for context_word in context:
                    training_data.append((target_idx, context_word))
        return training_data

    def _get_negative_samples(self, target, num_samples):
        """生成负采样样本"""
        samples = []
        while len(samples) < num_samples:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target:
                samples.append(neg)
        return samples

    def train(self):
        """训练模型"""
        training_data = self._generate_training_data()
        for epoch in range(self.epochs):
            loss = 0
            for target, context in training_data:
                target_vector = self.W1[target]  # (embedding_dim,)

                # 正样本 + 负样本
                samples = [context] + self._get_negative_samples(context, self.neg_samples)
                labels = [1] + [0] * self.neg_samples

                for sample, label in zip(samples, labels):
                    z = np.dot(self.W2[:, sample], target_vector)
                    pred = 1 / (1 + np.exp(-z))  # sigmoid
                    error = label - pred

                    # 梯度更新
                    self.W1[target] += self.learning_rate * error * self.W2[:, sample]
                    self.W2[:, sample] += self.learning_rate * error * target_vector

                    # loss
                    loss += - (label * np.log(pred + 1e-9) + (1 - label) * np.log(1 - pred + 1e-9))

            if epoch % (self.epochs // 5) == 0:
                print(f"Epoch {epoch}, Loss={loss:.4f}")

    def get_word_vector(self, word):
        """获取词向量"""
        if word in self.word2idx:
            return self.W1[self.word2idx[word]]
        else:
            raise ValueError(f"词 '{word}' 不在词表中")

    def most_similar(self, word, topn=5):
        """获取最相似的词"""
        if word not in self.word2idx:
            raise ValueError(f"词 '{word}' 不在词表中")

        word_vec = self.get_word_vector(word)
        sims = {}
        for other in self.vocab:
            if other == word:
                continue
            other_vec = self.get_word_vector(other)
            sim = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
            )
            sims[other] = sim
        return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:topn]


if __name__ == '__main__':
    corpus = [
        "我喜欢深度学习",
        "我喜欢自然语言处理",
        "这个是我自己训练的模型，请查看"
    ]
    model = SimpleWord2Vec(corpus)
    model.train()
    print(model.get_word_vector('小猫'))