import numpy as np
import matplotlib.pyplot as plt


def activate_func(x):
    return 1 / (1 + np.exp(-x))
    # return x * (x > 0)


def activate_func_derivative(x):
    return x * (1 - x)
    # return 1 * (x > 0)


def forward_propagation(X):
    z1 = np.dot(W1, X) + b1
    h1 = activate_func(z1)

    z2 = np.dot(W2, h1) + b2
    h2 = activate_func(z2)

    y_hat = np.dot(W3, h2) + b3

    return h1, h2, y_hat


def f(x, y):
    return forward_propagation(np.array([[x, y]]).T)[-1][0]


def backward_propagation(X, y, h1, h2, y_hat):
    global W1, b1, W2, b2, W3, b3

    # 计算输出层误差
    delta3 = y_hat - y
    dW3 = np.dot(delta3, h2.T)
    db3 = delta3

    # 计算第二个隐藏层的误差
    delta2 = np.dot(W3.T, delta3) * activate_func_derivative(h2)
    dW2 = np.dot(delta2, h1.T)
    db2 = delta2

    # 计算第一个隐藏层的误差
    delta1 = np.dot(W2.T, delta2) * activate_func_derivative(h1)
    dW1 = np.dot(delta1, X.T)
    db1 = delta1

    # 更新权重和偏置
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1.sum(axis=1, keepdims=True)
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2.sum(axis=1, keepdims=True)
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3.sum(axis=1, keepdims=True)


def train(X, y, epochs):
    for epoch in range(epochs):
        h1, h2, y_hat = forward_propagation(X)
        backward_propagation(X, y, h1, h2, y_hat)

        if epoch % 10 == 0:
            loss = np.mean((y_hat - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")


def get_data():
    # 输入数据和标签
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0], [1], [1], [0]]).T
    return X, y


def draw():
    def calculate_z(X, Y):
        # 初始化 Z 为与 X 和 Y 相同形状的数组
        Z = np.empty_like(X)

        # 遍历每个 (x, y) 对，并计算对应的 z 值
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(X[i, j], Y[i, j])

        return Z

    # 创建 x 和 y 的网格
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)

    # 计算 z 值
    Z = calculate_z(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    input_size = 2
    hidden_size = 64
    output_size = 1
    learning_rate = 0.001

    X, y = get_data()

    # 初始化权重和偏置
    W1 = np.random.randn(hidden_size, input_size)
    b1 = np.zeros((hidden_size, 1))

    W2 = np.random.randn(hidden_size, hidden_size)
    b2 = np.zeros((hidden_size, 1))

    W3 = np.random.randn(output_size, hidden_size)
    b3 = np.zeros((output_size, 1))

    draw()
    print(f(0, 0))
    print(f(0, 1))
    print(f(1, 0))
    print(f(1, 1))
    # 训练模型
    train(X, y, epochs=1000)
    draw()
    print(f(0, 0))
    print(f(0, 1))
    print(f(1, 0))
    print(f(1, 1))
