import re
import math
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            sentence, label = parts[0], int(parts[1])
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 平滑参数
        self.class_word_counts = defaultdict(Counter)  # 每个类别的词频
        self.class_total_words = defaultdict(int)  # 每个类别的总词数
        self.class_prior = dict()  # 类别先验概率
        self.vocab = set()  # 所有词汇
        self.classes = []

    def fit(self, X_train, y_train):
        self.classes = list(set(y_train))

        # 遍历每条句子，统计词频
        for sentence, label in zip(X_train, y_train):
            words = tokenize(sentence)
            for word in words:
                self.class_word_counts[label][word] += 1
                self.class_total_words[label] += 1
                self.vocab.add(word)

        vocab_size = len(self.vocab)

        # 计算每个类别的先验概率
        class_counts = Counter(y_train)
        total_samples = len(y_train)
        for cls in self.classes:
            self.class_prior[cls] = class_counts[cls] / total_samples

        # 保存用于平滑的词汇数量
        self.vocab_size = vocab_size

    def _log_likelihood(self, words, label):
        log_prob = math.log(self.class_prior[label])
        for word in words:
            count_in_class = self.class_word_counts[label].get(word, 0)
            total_words_in_class = self.class_total_words[label]
            prob = (count_in_class + self.alpha) / (total_words_in_class + self.vocab_size * self.alpha)
            log_prob += math.log(prob)
        return log_prob

    def predict(self, X_test):
        predictions = []
        for sentence in X_test:
            words = tokenize(sentence)
            scores = [(cls, self._log_likelihood(words, cls)) for cls in self.classes]
            predicted_class = max(scores, key=lambda x: x[1])[0]
            predictions.append(predicted_class)
        return predictions


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    train_sentences, train_labels = load_data(data_dir / "train.txt")
    test_sentences, test_labels = load_data(data_dir / "test.txt")
    target_names = ["Negative", "Positive"]

    nb = NaiveBayesClassifier(alpha=1.0)
    nb.fit(train_sentences, train_labels)

    y_pred = nb.predict(test_sentences)

    # 输出准确率
    print("Evaluation: ")
    print(classification_report(test_labels, y_pred, target_names=target_names))

    # 打印混淆矩阵以了解分类效果
    print("Confusion Matrix: ")
    mat = confusion_matrix(test_labels, y_pred)
    mat_df = pd.DataFrame(mat, index=target_names, columns=target_names)
    print(mat_df.to_markdown())
