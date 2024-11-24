# Projeto de Previsão de Preço de Carros

Este projeto utiliza uma rede neural para prever o preço de carros com base em um conjunto de características de entrada. O modelo é treinado usando diferentes funções de ativação, e a performance do modelo é avaliada com o erro quadrático médio (MSE) para diferentes funções de ativação (tanh, relu e sigmoid). O código foi desenvolvido em Python com o uso de bibliotecas como Pandas, Scikit-learn e Matplotlib.

## Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**: Manipulação e processamento de dados
- **Scikit-learn**: Divisão dos dados em treinamento e teste, além de cálculos de métricas
- **Matplotlib**: Visualização de dados e resultados
- **Numpy**: Operações com matrizes e álgebra linear (caso tenha sido utilizado em funções auxiliares)

## Como Rodar o Projeto

### 1. Clonar o Repositório

Primeiro, clone o repositório para o seu ambiente local:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Instale as bibliotecas utilizadas no código

Para instalar as bibliotecas necessárias, você pode usar o pip:

```bash
pip install -r requirements.txt
```

## Resultados Obtidos

Após a execução do treinamento, o código imprime os seguintes resultados:

### Erro Quadrático Médio (MSE) para diferentes funções de ativação:

- tanh
- relu
- sigmoid

### Arquitetura da Rede Neural:

- Tamanho da entrada
- Camadas ocultas
- Tamanho da saída

### Dimensões dos Dados de Entrada:

- Tamanho de `X_train` e `Y_train`

### Erro Quadrático Médio no Teste:

A precisão do modelo nos dados de teste é calculada com base no MSE.