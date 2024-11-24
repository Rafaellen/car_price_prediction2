import numpy as np
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation="tanh"):
        self.learning_rate = learning_rate
        self.activation = activation

        self.weights = []
        prev_size = input_size + 1
        for layer_size in hidden_layers:
            self.weights.append(np.random.randn(layer_size, prev_size))
            prev_size = layer_size + 1
        self.weights.append(np.random.randn(output_size, prev_size))

    def activate(self, z):
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError("Função de ativação não reconhecida!")

    def activate_derivative(self, z):
        if self.activation == "tanh":
            return 1 - np.tanh(z) ** 2
        elif self.activation == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activation == "sigmoid":
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid * (1 - sigmoid)
        else:
            raise ValueError("Função de ativação não reconhecida!")

    def train(self, X, Y, epocs=100):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # add bias
        Etm = []

        for epoc in range(epocs):
            E = []
            for i in range(X.shape[0]):
                # Feedforward
                z1 = np.dot(self.weights[0], X_bias[i])
                o1 = self.activate(z1)
                o1b = np.insert(o1, 0, 1)

                z2 = np.dot(self.weights[1], o1b)
                Y_pred = self.activate(z2)[0]

                e = Y[i] - Y_pred
                E.append((e**2) / 2)

                # backpropagation
                delta2 = e * self.activate_derivative(z2)[0]
                delta1 = self.activate_derivative(
                    z1) * np.dot(self.weights[1][0, 1:], delta2)

                # att pesos
                dw2 = self.learning_rate * delta2 * o1b
                self.weights[1] += dw2.reshape(1, -1)

                dw1 = self.learning_rate * np.outer(delta1, X_bias[i])
                self.weights[0] += dw1

            # calculo do erro medio
            error_mean = np.mean(E)
            Etm.append(error_mean)

            if epoc % 10 == 0:
                print(f'Época {epoc}, Erro médio: {error_mean:.6f}')

        plt.figure(figsize=(10, 6))
        plt.plot(Etm, label="Erro Médio")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Médio")
        plt.title(f"Evolução do Erro Médio ({self.activation})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def predict(self, X):
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        predictions = np.zeros(X.shape[0])

        for i in range(X_bias.shape[0]):
            z1 = np.dot(self.weights[0], X_bias[i])
            o1 = self.activate(z1)
            o1b = np.insert(o1, 0, 1)

            z2 = np.dot(self.weights[1], o1b)
            predictions[i] = self.activate(z2)[0]

        return predictions
