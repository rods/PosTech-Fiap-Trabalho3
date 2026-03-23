# Requirements Document

## Introduction

Este documento especifica os requisitos para um sistema de análise e previsão de atrasos de voos nos EUA, desenvolvido como Tech Challenge acadêmico de pós-graduação em Machine Learning. O sistema deve realizar exploração de dados, modelagem supervisionada, modelagem não supervisionada e apresentar resultados críticos sobre padrões de atrasos em voos.

## Glossary

- **Flight_Delay_System**: Sistema completo de análise e previsão de atrasos de voos
- **EDA_Module**: Módulo de Análise Exploratória de Dados (Exploratory Data Analysis)
- **Supervised_Model**: Modelo de aprendizado supervisionado para classificação ou regressão
- **Unsupervised_Model**: Modelo de aprendizado não supervisionado para clusterização ou redução de dimensionalidade
- **Flight_Dataset**: Conjunto de dados históricos de voos nos EUA (3 arquivos CSV)
- **Delay_Threshold**: Limite de tempo (em minutos) que define se um voo é considerado atrasado
- **Feature**: Característica ou variável do dataset usada para modelagem
- **Metric**: Métrica de avaliação de desempenho do modelo
- **Cluster**: Grupo de dados similares identificado por algoritmo de clusterização
- **Notebook**: Jupyter Notebook ou Google Colab notebook contendo código e análises

## Data Dictionary

O Flight_Dataset é composto por 3 arquivos CSV:

1. **airlines.csv** (359 bytes): Informações sobre companhias aéreas
2. **airports.csv** (24 KB): Informações sobre aeroportos dos EUA
3. **flights.csv** (592,4 MB): Dataset principal com dados históricos de voos

### Colunas do flights.csv

O arquivo principal flights.csv contém as seguintes colunas:

### Informações Temporais
- **YEAR**: Ano do voo (ex.: 2015) - Inteiro
- **MONTH**: Mês do voo (1 a 12) - Inteiro
- **DAY**: Dia do mês do voo (1 a 31) - Inteiro
- **DAY_OF_WEEK**: Dia da semana (1 = Segunda, 7 = Domingo) - Inteiro

### Informações do Voo
- **AIRLINE**: Código da companhia aérea (ex.: AA = American Airlines) - Categórica
- **FLIGHT_NUMBER**: Número do voo - Inteiro
- **TAIL_NUMBER**: Número de registro da aeronave - Texto
- **ORIGIN_AIRPORT**: Código IATA do aeroporto de origem (ex.: ATL) - Categórica
- **DESTINATION_AIRPORT**: Código IATA do aeroporto de destino - Categórica

### Horários e Tempos de Partida
- **SCHEDULED_DEPARTURE**: Horário de partida programado (HHMM) - Inteiro
- **DEPARTURE_TIME**: Horário real de partida (HHMM) - Inteiro
- **DEPARTURE_DELAY**: Atraso na partida (em minutos) - Numérico
- **TAXI_OUT**: Tempo gasto taxiando até a decolagem (em minutos) - Numérico
- **WHEELS_OFF**: Horário em que o avião decolou (HHMM) - Inteiro

### Tempos de Voo
- **SCHEDULED_TIME**: Tempo total programado de voo (em minutos) - Numérico
- **ELAPSED_TIME**: Tempo total real de voo (em minutos) - Numérico
- **AIR_TIME**: Tempo no ar (em minutos) - Numérico
- **DISTANCE**: Distância entre origem e destino (em milhas) - Numérico

### Horários e Tempos de Chegada
- **WHEELS_ON**: Horário em que as rodas tocaram o solo (HHMM) - Inteiro
- **TAXI_IN**: Tempo taxiando até o portão de desembarque (em minutos) - Numérico
- **SCHEDULED_ARRIVAL**: Horário de chegada programado (HHMM) - Inteiro
- **ARRIVAL_TIME**: Horário de chegada real (HHMM) - Inteiro
- **ARRIVAL_DELAY**: Atraso na chegada (em minutos) - Numérico

### Status do Voo
- **DIVERTED**: Indica se o voo foi desviado (1 = sim, 0 = não) - Binária
- **CANCELLED**: Indica se o voo foi cancelado (1 = sim, 0 = não) - Binária
- **CANCELLATION_REASON**: Motivo do cancelamento (A = Airline, B = Weather, C = NAS, D = Security) - Categórica

### Tipos de Atraso
- **AIR_SYSTEM_DELAY**: Atraso causado por controle de tráfego aéreo - Numérico
- **SECURITY_DELAY**: Atraso causado por problemas de segurança - Numérico
- **AIRLINE_DELAY**: Atraso causado pela companhia aérea - Numérico
- **LATE_AIRCRAFT_DELAY**: Atraso causado por chegada tardia da aeronave - Numérico
- **WEATHER_DELAY**: Atraso causado por condições meteorológicas - Numérico

## Requirements

### Requirement 1: Carregar e Preparar Dados de Voos

**User Story:** Como cientista de dados, eu quero carregar e preparar o dataset de voos, para que eu possa realizar análises e modelagem.

#### Acceptance Criteria

1. THE EDA_Module SHALL load the 3 CSV files (airlines.csv, airports.csv, flights.csv)
2. THE EDA_Module SHALL merge or join datasets where appropriate (ex.: enriquecer flights com nomes de airlines e airports)
3. WHEN missing values are detected, THE EDA_Module SHALL document the percentage of missing values per column
4. THE EDA_Module SHALL apply a strategy to handle missing values (removal, imputation, or flagging)
5. THE EDA_Module SHALL create derived features for temporal analysis (day of week, month, hour)
6. THE EDA_Module SHALL display the final dataset shape and column types

### Requirement 2: Realizar Análise Exploratória de Dados

**User Story:** Como cientista de dados, eu quero explorar os dados de voos, para que eu possa entender padrões e características dos atrasos.

#### Acceptance Criteria

1. THE EDA_Module SHALL calculate descriptive statistics for numerical features (mean, median, standard deviation, min, max)
2. THE EDA_Module SHALL generate a distribution visualization for delay times
3. THE EDA_Module SHALL create visualizations showing delay patterns by airport
4. THE EDA_Module SHALL create visualizations showing delay patterns by time period (hour, day of week, month)
5. THE EDA_Module SHALL create visualizations showing delay patterns by airline
6. THE EDA_Module SHALL identify and visualize the top 10 airports with highest delay rates
7. THE EDA_Module SHALL generate a correlation matrix visualization for numerical features

### Requirement 3: Implementar Modelo de Classificação de Atrasos

**User Story:** Como cientista de dados, eu quero classificar se um voo vai atrasar, para que eu possa prever atrasos binariamente.

#### Acceptance Criteria

1. THE Supervised_Model SHALL define a binary target variable based on Delay_Threshold
2. THE Supervised_Model SHALL split the Flight_Dataset into training and test sets with at least 20% for testing
3. THE Supervised_Model SHALL implement at least 2 different classification algorithms
4. WHEN training is complete, THE Supervised_Model SHALL calculate accuracy, precision, recall and F1-score for each algorithm
5. THE Supervised_Model SHALL generate a confusion matrix visualization for each algorithm
6. THE Supervised_Model SHALL compare the performance of all algorithms in a summary table or chart
7. THE Supervised_Model SHALL identify the top 10 most important features for the best performing model

### Requirement 4: Implementar Modelo de Regressão de Tempo de Atraso

**User Story:** Como cientista de dados, eu quero prever o tempo de atraso em minutos, para que eu possa estimar a duração dos atrasos.

#### Acceptance Criteria

1. THE Supervised_Model SHALL define a continuous target variable representing delay duration in minutes
2. THE Supervised_Model SHALL filter the dataset to include only delayed flights for regression analysis
3. THE Supervised_Model SHALL implement at least 2 different regression algorithms
4. WHEN training is complete, THE Supervised_Model SHALL calculate MAE, RMSE and R² for each algorithm
5. THE Supervised_Model SHALL generate a scatter plot of predicted vs actual values for each algorithm
6. THE Supervised_Model SHALL compare the performance of all algorithms in a summary table or chart

### Requirement 5: Realizar Clusterização de Aeroportos ou Rotas

**User Story:** Como cientista de dados, eu quero agrupar aeroportos ou rotas com perfis similares, para que eu possa identificar padrões de comportamento.

#### Acceptance Criteria

1. THE Unsupervised_Model SHALL aggregate Flight_Dataset by airport or route to create clustering features
2. THE Unsupervised_Model SHALL normalize or standardize features before clustering
3. THE Unsupervised_Model SHALL implement at least 1 clustering algorithm
4. THE Unsupervised_Model SHALL determine the optimal number of clusters using elbow method or silhouette score
5. THE Unsupervised_Model SHALL assign each airport or route to a Cluster
6. THE Unsupervised_Model SHALL generate a visualization showing the clusters in 2D space
7. THE Unsupervised_Model SHALL provide interpretation of each Cluster with descriptive statistics

### Requirement 6: Aplicar Redução de Dimensionalidade

**User Story:** Como cientista de dados, eu quero reduzir a dimensionalidade dos dados, para que eu possa visualizar e entender a estrutura dos dados em espaço reduzido.

#### Acceptance Criteria

1. THE Unsupervised_Model SHALL select numerical features for dimensionality reduction
2. THE Unsupervised_Model SHALL apply PCA or another dimensionality reduction technique
3. THE Unsupervised_Model SHALL reduce features to 2 or 3 principal components for visualization
4. THE Unsupervised_Model SHALL calculate and display the explained variance ratio for each component
5. THE Unsupervised_Model SHALL generate a scatter plot visualization of the reduced dimensions
6. WHERE clusters exist, THE Unsupervised_Model SHALL color the reduced dimension plot by cluster assignment

### Requirement 7: Apresentar Resultados e Conclusões

**User Story:** Como cientista de dados, eu quero documentar resultados e conclusões, para que eu possa comunicar os achados do projeto.

#### Acceptance Criteria

1. THE Flight_Delay_System SHALL document the main insights from EDA in the Notebook
2. THE Flight_Delay_System SHALL document which supervised model performed best and why
3. THE Flight_Delay_System SHALL document the interpretation of unsupervised learning results
4. THE Flight_Delay_System SHALL identify and document at least 3 limitations of the models or analysis
5. THE Flight_Delay_System SHALL propose at least 3 improvements or next steps for future work
6. THE Flight_Delay_System SHALL answer the guiding questions about critical airports, delay characteristics, temporal patterns, and prediction capability

### Requirement 8: Organizar Código em Notebooks

**User Story:** Como desenvolvedor, eu quero organizar o código em notebooks estruturados, para que o projeto seja fácil de entender e reproduzir.

#### Acceptance Criteria

1. THE Flight_Delay_System SHALL organize code in one or more Jupyter Notebooks or Google Colab notebooks
2. THE Notebook SHALL include markdown cells explaining each analysis step
3. THE Notebook SHALL display all visualizations inline
4. THE Notebook SHALL include a table of contents or clear section headers
5. WHEN executed sequentially, THE Notebook SHALL run without errors and reproduce all results
6. THE Flight_Delay_System SHALL include a README file with instructions to run the notebooks

### Requirement 9: Preparar Repositório para Entrega

**User Story:** Como estudante, eu quero preparar o repositório para entrega, para que o projeto atenda aos requisitos do Tech Challenge.

#### Acceptance Criteria

1. THE Flight_Delay_System SHALL include all notebooks in the repository
2. THE Flight_Delay_System SHALL include a requirements.txt or environment.yml file listing dependencies
3. THE Flight_Delay_System SHALL include a README with project description, setup instructions, and how to run
4. WHERE data files are small, THE Flight_Delay_System SHALL include them in the repository or provide download links
5. THE Flight_Delay_System SHALL include visualizations as outputs in the notebooks or as saved image files
6. THE Flight_Delay_System SHALL organize files in a clear directory structure
