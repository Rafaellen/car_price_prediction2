import numpy as np
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Inicializando pesos aleatórios
        self.weights = []
        prev_size = input_size + 1  # Inclui o bias
        for layer_size in hidden_layers:
            self.weights.append(np.random.randn(layer_size, prev_size))
            prev_size = layer_size + 1
        self.weights.append(np.random.randn(output_size, prev_size))

    def train(self, X, Y, epocs=100):
        # Adiciona o bias às entradas
        bias = 1
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Lista para armazenar os erros médios
        Etm = []

        for epoc in range(epocs):
            E = []  # Erros de cada padrão
            for i in range(X.shape[0]):
                # Apresentação do padrão
                Xb = X_bias[i]

                # Feedforward
                # Saída da camada escondida
                o1 = np.tanh(self.weights[0].dot(Xb))
                # Inclui o bias para a camada de saída
                o1b = np.insert(o1, 0, bias)
                Y_pred = np.tanh(self.weights[1].dot(o1b))  # Saída da rede

                # Cálculo do erro
                e = Y[i] - Y_pred
                E.append((e**2) / 2)

                # Backpropagation
                delta2 = e * (1 - Y_pred**2)  # Gradiente da camada de saída
                # Gradiente da camada escondida
                delta1 = (1 - o1b**2) * (self.weights[1].T.dot(delta2))

                # Atualização dos pesos
                self.weights[1] += self.learning_rate * np.outer(delta2, o1b)
                self.weights[0] += self.learning_rate * \
                    np.outer(delta1[1:], Xb)  # Ignorar bias

            # Calcula o erro médio por época
            Etm.append(np.mean(E))

        # Plota o erro médio ao longo das épocas
        plt.plot(Etm, label="Erro Médio")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Médio")
        plt.legend()
        plt.show()

    def predict(self, X):
        bias = 1
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # Adiciona o bias
        predictions = []

        for i in range(X_bias.shape[0]):
            Xb = X_bias[i]  # Uma amostra por vez
            o1 = np.tanh(self.weights[0].dot(Xb))  # Saída da camada escondida
            # Inclui o bias para a camada de saída
            o1b = np.insert(o1, 0, bias)
            # Saída da rede (uma previsão por amostra)
            Y_pred = np.tanh(self.weights[1].dot(o1b))
            predictions.append(Y_pred)

    # Retorna o vetor de previsões 1D
        # Retorna uma matriz 1D para facilitar o cálculo de erro
        return np.array(predictions).flatten()
