# 机器学习和深度学习实战

本仓库面向初学者，提供机器学习和深度学习实战代码，帮助读者快速上手。

## 目录

### 机器学习

以 scikit-learn 为主，提供机器学习实战代码。其中多层感知机和贝叶斯分类器提供了从零开始的实现。

- [X] 简单回归
    - [X] 多元线性回归
    - [X] 逻辑回归
- [X] 树分类器
    - [X] 决策树
    - [X] 随机森林
    - [X] XGBoost
- [X] 支持向量机
- [X] 聚类
    - [X] K-Mean
    - [X] DBSCAN
- [X] 贝叶斯分类器
    - [X] 从零开始实现词袋模型 + 朴素贝叶斯
    - [X] 词袋模型 + MultinomialNB
    - [X] TF-IDF + MultinomialNB
- [X] 多层感知机
    - [X] 从零开始实现
    - [X] scikit-learn version

### 深度学习

以 PyTorch 为主，提供深度学习实战代码。预训练模型来自 Hugging Face。部分训练框架使用 PyTorch Lightning。

- [X] 深度神经网络
    - [X] 多层感知机
- [ ] 文本嵌入模型
    - [ ] 词嵌入
        - [ ] Word2Vec + SVM + 情感分析
        - [ ] GloVe + SVM + 情感分析
    - [ ] 句嵌入
        - [ ] Laser + SVM + 情感分析
- [X] 卷积神经网络
    - [X] LeNet + MNIST
    - [ ] ResNet + PlantDoc
    - [ ] SwinTransformer + PlantDoc
    - [ ] ConvNeXt + PlantDoc
    - [ ] LeNet + MNIST (Lightning version)
    - [ ] ResNet + PlantDoc (Lightning version)
    - [ ] SwinTransformer + PlantDoc (Lightning version)
    - [ ] ConvNeXt + PlantDoc (Lightning version)
- [ ] 循环神经网络
    - [ ] RNN + 情感分析
    - [ ] LSTM + 情感分析
    - [ ] GRU + 情感分析

### 大语言模型专题

- [ ] 仅编码器模型 (Encoder-only Model)
    - [ ] BERT + 情感分析
- [ ] 编码器-解码器模型 (Encoder-Decoder Model)
    - [ ] T5 + ASQP
- [ ] 仅解码器模型 (Decoder-only Model)
    - [ ] LLaMA + ASQP
- [ ] 视觉语言模型 (Vision-Language Model)
    - [ ] Qwen3 + 图像标签预测

## 数据集来源

- [PlantDoc](https://github.com/PlantDoc/PlantDoc)
- [ASQP](https://github.com/IsakZhang/ABSA-QUAD)
- [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [Stanford Sentiment Analysis Dataset](https://nlp.stanford.edu/sentiment/)
- [Iris](https://archive.ics.uci.edu/ml/datasets/Iris)