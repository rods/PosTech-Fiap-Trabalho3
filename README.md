# Flight Delay ML Analysis

Análise e previsão de atrasos de voos nos EUA usando Machine Learning.
Projeto acadêmico de pós-graduação desenvolvido como Tech Challenge.

## Descrição

Este projeto realiza uma análise completa de dados históricos de voos nos EUA, cobrindo:

- **EDA** — análise exploratória de padrões de atrasos por aeroporto, companhia aérea e período
- **Classificação** — modelos para prever se um voo vai atrasar (variável binária)
- **Regressão** — modelos para estimar o tempo de atraso em minutos
- **Clusterização** — agrupamento de aeroportos por perfil de atraso (KMeans)
- **PCA** — redução de dimensionalidade para visualização da estrutura dos dados


## Estrutura do Projeto

```
flight-delay-ml-analysis/
├── data/               # Arquivos CSV do dataset (não versionados)
├── notebooks/          # Jupyter Notebooks com a análise
├── docs/               # Documentação e visualizações salvas
├── requirements.txt    # Dependências Python
└── analysis_completo.html  # Output da execução do notebook
└── analysis_resumo.html  # Versão resumida usada no vídeo
└── README.md
```

## Dataset

Os dados são compostos por 3 arquivos CSV de voos nos EUA:

| Arquivo        | Tamanho  | Descrição                          |
|----------------|----------|------------------------------------|
| airlines.csv   | ~359 B   | Códigos e nomes das companhias     |
| airports.csv   | ~24 KB   | Informações dos aeroportos         |
| flights.csv    | ~592 MB  | Dados históricos de voos (dataset principal) |

Baixe os arquivos em: https://www.kaggle.com/datasets/usdot/flight-delays  
Coloque-os no diretório `data/` antes de executar os notebooks.

## Setup

### Pré-requisitos

- Python 3.8+
- pip

### Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd flight-delay-ml-analysis

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Executar os Notebooks

```bash
jupyter notebook
```

Abra o arquivo `notebooks/flight_delay_analysis.ipynb` e execute as células sequencialmente.

> **Nota:** O arquivo `flights.csv` tem ~592 MB. Certifique-se de ter pelo menos 4 GB de RAM disponível.

### Alternativa: Google Colab

1. Faça upload do notebook `notebooks/flight_delay_analysis.ipynb` no [Google Colab](https://colab.research.google.com/)
2. Faça upload dos arquivos CSV para o Colab ou monte o Google Drive
3. Ajuste o caminho dos dados (`DATA_PATH`) na primeira célula de configuração
4. Execute as células sequencialmente

## Reprodutibilidade

Todas as operações aleatórias usam `RANDOM_STATE = 42` para garantir resultados reproduzíveis.

## Tecnologias

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## Licença

Projeto acadêmico desenvolvido como Tech Challenge de pós-graduação em Machine Learning.
Dataset disponibilizado pelo [US Department of Transportation](https://www.kaggle.com/datasets/usdot/flight-delays).
