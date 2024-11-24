Projeto de Previsão de Preço de Carros
Este projeto utiliza uma rede neural para prever o preço de carros com base em um conjunto de características de entrada. O modelo é treinado usando diferentes funções de ativação, e a performance do modelo é avaliada com o erro quadrático médio (MSE) para diferentes funções de ativação (tanh, relu e sigmoid). O código foi desenvolvido em Python com o uso de bibliotecas como Pandas, Scikit-learn e Matplotlib.

Tecnologias Utilizadas
Python 3.x
Pandas: Manipulação e processamento de dados
Scikit-learn: Divisão dos dados em treinamento e teste, além de cálculos de métricas
Matplotlib: Visualização de dados e resultados
Numpy: Operações com matrizes e álgebra linear (caso tenha sido utilizado em funções auxiliares)
Como Rodar o Projeto
1. Clonar o Repositório
Primeiro, clone o repositório para o seu ambiente local:

bash
Copiar código
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
2. Criar um Ambiente Virtual
É recomendável criar um ambiente virtual para isolar as dependências do projeto. Execute o seguinte comando para criar um ambiente virtual:

bash
Copiar código
python -m venv venv
Ative o ambiente virtual:

No Windows:

bash
Copiar código
venv\Scripts\activate
No macOS/Linux:

bash
Copiar código
source venv/bin/activate
3. Instalar Dependências
Instale as dependências listadas no arquivo requirements.txt:

bash
Copiar código
pip install -r requirements.txt
4. Rodar o Código
Após configurar o ambiente, você pode executar o script main.py para treinar o modelo e avaliar o desempenho. Certifique-se de que o arquivo train.csv esteja presente no diretório correto ou altere o caminho do arquivo conforme necessário.

bash
Copiar código
python main.py
5. Visualização e Resultados
O código gera a comparação do erro quadrático médio para diferentes funções de ativação (tanh, relu e sigmoid) e imprime informações sobre a arquitetura da rede neural e dimensões dos dados.

Estrutura do Projeto
bash
Copiar código
/trabalho_malga2
    ├── data/
    │   └── train.csv            # Dataset de entrada
    ├── model.py                 # Definição da classe da rede neural
    ├── processing.py            # Funções de pré-processamento e normalização dos dados
    ├── main.py                  # Código principal para treinamento e avaliação
    ├── requirements.txt         # Dependências do projeto
    └── README.md                # Este arquivo
Resultados Obtidos
Após a execução do treinamento, o código imprime os seguintes resultados:

Erro Quadrático Médio (MSE) para diferentes funções de ativação:

tanh
relu
sigmoid
Arquitetura da Rede Neural:

Tamanho da entrada
Camadas ocultas
Tamanho da saída
Dimensões dos Dados de Entrada:

Tamanho de X_train e Y_train
Erro Quadrático Médio no Teste:

A precisão do modelo nos dados de teste é calculada com base no MSE.
Contribuições
