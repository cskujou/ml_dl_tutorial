import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path


# 从本地文件加载数据
def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t")
    return data["Sentence"].values, data["Label"].values


if __name__ == "__main__":
    # 加载训练和测试数据
    dataset_path = Path(__file__).parent / "data"
    train_sentences, train_labels = load_data(dataset_path / "train.txt")
    test_sentences, test_labels = load_data(dataset_path / "test.txt")

    # 创建一个包含数据向量化和分类模型的流水线
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_sentences)
    X_test = vectorizer.transform(test_sentences)
    labels = ["Negative", "Positive"]

    # 训练朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    # 在测试集上进行预测
    y_pred = clf.predict(X_test)

    # 输出准确率
    print("Evaluation: ")
    print(classification_report(test_labels, y_pred, target_names=labels))

    # 打印混淆矩阵以了解分类效果
    print("Confusion Matrix: ")
    mat = confusion_matrix(test_labels, y_pred)
    mat_df = pd.DataFrame(mat, index=labels, columns=labels)
    print(mat_df.to_markdown())
