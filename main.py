from model import Regressor
from processing import processing_data, normal_data
import pandas as pd
from sklearn.model_selection import train_test_split


def exibir_resultados(arquitetura, mse_ativacoes, mse_teste):
    print("\n### Resultados ###")
    print(f"- Erro Quadrático Médio por Função de Ativação:")
    for func, mse in mse_ativacoes.items():
        print(f"  - {func}: {mse:.4f}")
    print(
        f"- Erro Quadrático Médio no Teste: {mse_teste:.4f}")

    print("\n### Arquitetura da Rede ###")
    print(f"- Entrada (Input size): {arquitetura['input_size']}")
    print(f"- Camadas Ocultas (Hidden layers): {arquitetura['hidden_layers']}")
    print(f"- Saída (Output size): {arquitetura['output_size']}")

    print("\n### Dimensões dos Dados ###")
    for k, v in arquitetura['data_shapes'].items():
        print(f"- {k}: {v}")


file = 'C:/Users/Usuario/Desktop/trabalho_malga2/data/train.csv'
ds = pd.read_csv(file)
X, Y = processing_data(ds)
X_normal, _ = normal_data(X)
Y_normal = (Y - Y.min()) / (Y.max() - Y.min())


X_train, X_test, Y_train, Y_test = train_test_split(
    X_normal, Y_normal, test_size=0.6, random_state=42)


input_size = X_train.shape[1]
hidden_layers = [16, 16, 8]
output_size = 1
learning_rate = 0.01
epocs = 100


funcoes_ativacao = ["tanh", "relu", "sigmoid"]
mse_ativacoes = {}
for ativacao in funcoes_ativacao:
    regressor = Regressor(
        input_size=input_size,
        hidden_layers=hidden_layers,
        output_size=output_size,
        learning_rate=learning_rate,
        activation=ativacao
    )
    regressor.train(X_train, Y_train.values, epocs=epocs)
    Y_pred = regressor.predict(X_test)
    mse_ativacoes[ativacao] = ((Y_test.values - Y_pred.flatten())**2).mean()


regressor_default = Regressor(
    input_size=input_size,
    hidden_layers=hidden_layers,
    output_size=output_size,
    learning_rate=learning_rate
)
regressor_default.train(X_train, Y_train.values, epocs=epocs)
Y_pred_default = regressor_default.predict(X_test)
mse_teste = ((Y_test.values - Y_pred_default.flatten())**2).mean()

# organizacao
arquitetura = {
    "input_size": input_size,
    "hidden_layers": hidden_layers,
    "output_size": output_size,
    "data_shapes": {
        "X_train": X_train.shape,
        "Y_train": Y_train.shape,
        "X_test": X_test.shape,
        "Y_test": Y_test.shape,
    }
}
exibir_resultados(arquitetura, mse_ativacoes, mse_teste)
