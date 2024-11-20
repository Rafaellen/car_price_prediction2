from model import Regressor
from processing import processing_data, normal_data
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Carrega o dataset
file = 'C:/Users/Usuario/Desktop/trabalho_malga2/data/train.csv'
ds = pd.read_csv(file)

# Processa os dados
X, Y = processing_data(ds)
X_normal, _ = normal_data(X)
Y_normal = (Y - Y.min()) / (Y.max() - Y.min())  # Normaliza Y

# Divide em treinamento e teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X_normal, Y_normal, test_size=0.6, random_state=42)

# Configura a rede neural
input_size = X_train.shape[1]
hidden_layers = [16, 16, 8]
output_size = 1
learning_rate = 0.01
epocs = 100

# Inicializa e treina a rede
ia = Regressor(input_size=input_size, hidden_layers=hidden_layers,
               output_size=output_size, learning_rate=learning_rate)
ia.train(X_train, Y_train.values, epocs=epocs)

# Avalia a rede
Y_pred = ia.predict(X_test)
print(f"Forma de Y_test: {Y_test.shape}")
print(f"Forma de Y_pred: {Y_pred.shape}")
# Verifique o tamanho de X_test
print(f"Forma de X_test: {X_test.shape}")

mse = ((Y_test.values - Y_pred.flatten())**2).mean()
print(f"Erro Quadrático Médio no Teste: {mse}")


"""
O que falta? 
- Testar mais funções de ativação (por exemplo, relu e sigmoid).
- Ajustar o número de camadas ocultas (feito)
- func predict está gerando mais previsões do q esperamos (ajustar urgente)
"""
