import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ActivateFunction:
    def forward(self, x):
        pass

    def derivative(self, x):
        pass


class Sigmoid(ActivateFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class Tanh(ActivateFunction):
    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - x**2


class ReLU(ActivateFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


np.random.seed(42)


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, af=ReLU(), learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = [
            np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) for i in range(len(self.layer_sizes) - 1)
        ]
        self.biases = [np.zeros((self.layer_sizes[i + 1], 1)) for i in range(len(self.layer_sizes) - 1)]
        self.af = af
        self.X = None
        self.y = None

    def forward_propagation(self, X):
        activations = [X]

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(W, activations[-1]) + b
            if i != len(self.weights) - 1:
                z = self.af.forward(z)
            activations.append(z)

        return activations

    def backward_propagation(self, y, activations):
        delta = activations[-1] - y
        # delta = (activations[-1] - y) * self.af.derivative(activations[-1])
        dWs = []
        dbs = []

        for i in range(len(self.layer_sizes) - 2, -1, -1):
            dW = np.dot(delta, activations[i].T)
            db = delta
            dWs.insert(0, dW)
            dbs.insert(0, db)

            if i != 0:
                delta = np.dot(self.weights[i].T, delta) * self.af.derivative(activations[i])

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dWs[i]
            self.biases[i] -= self.learning_rate * dbs[i].sum(axis=1, keepdims=True)

    def predict(self, X):
        activations = self.forward_propagation(X)
        return activations[-1]

    def train(self, X, y, epochs, update_interval=10):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.update_interval = update_interval

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        x = np.linspace(-0.5, 1.5, 30)
        y = np.linspace(-0.5, 1.5, 30)
        self.X_grid, self.Y_grid = np.meshgrid(x, y)
        self.Z_grid = np.zeros_like(self.X_grid)

        self.surf = self.ax.plot_surface(self.X_grid, self.Y_grid, self.Z_grid, cmap=plt.cm.coolwarm)
        # self.fig.colorbar(self.surf, shrink=0.5, aspect=5)
        self.ax.set_title("Epoch: 0")

        FuncAnimation(
            self.fig,
            self.update_plot,
            frames=range(0, self.epochs, self.update_interval),
            interval=self.update_interval * 10,
            repeat=False,
        )

        plt.show()

    def update_plot(self, epoch):
        activations = self.forward_propagation(self.X)
        self.backward_propagation(self.y, activations)
        loss = np.mean((activations[-1] - self.y) ** 2)

        for i in range(self.X_grid.shape[0]):
            for j in range(self.X_grid.shape[1]):
                self.Z_grid[i, j] = f(self.X_grid[i, j], self.Y_grid[i, j], self)

        self.ax.clear()
        self.surf = self.ax.plot_surface(self.X_grid, self.Y_grid, self.Z_grid, cmap=plt.cm.coolwarm)
        self.ax.set_title(f"Epoch: {epoch}   Loss: {loss:.6f}")
        return (self.surf,)


def f(x, y, mlp):
    return mlp.predict(np.array([[x, y]]).T)[0, 0]


# 网络参数
input_size = 2
output_size = 1

hidden_sizes = [16, 16]
af = ReLU()
learning_rate = 0.01

# hidden_sizes = [64, 64, 32, 32]
# af = Sigmoid()
# learning_rate = 0.01

# hidden_sizes = [16, 16]
# af = Tanh()
# learning_rate = 0.01

num_epochs = 1000


mlp = MLP(input_size, hidden_sizes, output_size, af, learning_rate)

# 测试用数据集
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
y = np.array([[0], [1], [1], [0]]).T

mlp.train(X, y, num_epochs, update_interval=10)
