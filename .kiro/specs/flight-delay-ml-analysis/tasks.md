# Plano de Implementação: Flight Delay ML Analysis

## Visão Geral

Este plano detalha a implementação de um sistema de análise e previsão de atrasos de voos nos EUA usando Machine Learning. O projeto será desenvolvido em notebooks Jupyter/Colab, focando em análise exploratória, modelagem supervisionada (classificação e regressão) e não supervisionada (clustering e PCA).

## Tarefas

- [x] 1. Configurar estrutura do projeto e ambiente
  - Criar diretório do projeto com estrutura organizada
  - Criar arquivo requirements.txt com dependências (pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter)
  - Criar README.md com descrição do projeto e instruções de setup
  - _Requisitos: 9.2, 9.3_

- [x] 2. Implementar carregamento e preparação de dados
  - [x] 2.1 Criar notebook principal e implementar carregamento dos 3 CSVs
    - Implementar função `load_datasets()` para carregar airlines.csv, airports.csv e flights.csv
    - Usar dtypes otimizados para reduzir uso de memória (category para strings, int8/int16 para inteiros pequenos)
    - Exibir shape e informações básicas de cada DataFrame
    - _Requisitos: 1.1, 8.1_
  
  - [x] 2.2 Implementar merge dos datasets
    - Implementar função `merge_datasets()` para enriquecer flights com informações de airlines e airports
    - Fazer merge de flights com airlines usando coluna AIRLINE
    - Fazer merge de flights com airports para origem e destino (ORIGIN_AIRPORT e DESTINATION_AIRPORT)
    - Validar que número de linhas permanece consistente após merge
    - _Requisitos: 1.2_
  
  - [x] 2.3 Implementar análise e tratamento de valores ausentes
    - Calcular e exibir porcentagem de missing values por coluna
    - Documentar colunas com alta porcentagem de valores ausentes (>50%)
    - Implementar função `handle_missing_values()` com estratégia de remoção ou imputação
    - Aplicar tratamento e validar resultado
    - _Requisitos: 1.3, 1.4_
  
  - [x] 2.4 Implementar feature engineering
    - Implementar função `create_temporal_features()` para criar HOUR_OF_DAY, IS_WEEKEND
    - Implementar função `create_delay_flag()` para criar variável binária IS_DELAYED (threshold=15 minutos)
    - Criar feature ROUTE concatenando origem e destino
    - Validar que features derivadas têm valores no intervalo esperado
    - _Requisitos: 1.5, 1.6_

- [x] 3. Checkpoint - Validar preparação de dados
  - Verificar que dataset preparado tem shape esperado e sem erros críticos
  - Confirmar que todas as features necessárias foram criadas
  - Perguntar ao usuário se há dúvidas ou ajustes necessários

- [x] 4. Implementar análise exploratória de dados (EDA)
  - [x] 4.1 Implementar estatísticas descritivas
    - Implementar função `calculate_descriptive_stats()` para calcular mean, median, std, min, max
    - Exibir estatísticas para features numéricas principais (delays, tempos, distância)
    - _Requisitos: 2.1_
  
  - [x] 4.2 Criar visualizações de distribuição de atrasos
    - Implementar função `plot_delay_distribution()` com histograma de ARRIVAL_DELAY
    - Adicionar visualização de distribuição por categoria de atraso
    - _Requisitos: 2.2_
  
  - [x] 4.3 Criar visualizações de atrasos por aeroporto
    - Implementar função `plot_delays_by_airport()` para top 10 aeroportos
    - Calcular taxa de atraso por aeroporto (% de voos atrasados)
    - Criar gráfico de barras ordenado por taxa de atraso
    - _Requisitos: 2.3, 2.6_
  
  - [x] 4.4 Criar visualizações de padrões temporais
    - Implementar função `plot_delays_by_time()` para analisar padrões por hora, dia da semana e mês
    - Criar subplots mostrando taxa de atraso por cada dimensão temporal
    - _Requisitos: 2.4_
  
  - [x] 4.5 Criar visualizações de atrasos por companhia aérea
    - Implementar função `plot_delays_by_airline()` para comparar companhias
    - Criar gráfico de barras com taxa de atraso por airline
    - _Requisitos: 2.5_
  
  - [x] 4.6 Criar matriz de correlação
    - Implementar função `plot_correlation_matrix()` com heatmap
    - Selecionar features numéricas relevantes para correlação
    - Identificar features altamente correlacionadas
    - _Requisitos: 2.7_

- [x] 5. Checkpoint - Revisar insights da EDA
  - Documentar principais insights identificados na análise exploratória
  - Perguntar ao usuário se há análises adicionais necessárias

- [x] 6. Implementar modelos de classificação de atrasos
  - [x] 6.1 Preparar dados para classificação
    - Implementar função `prepare_features_and_target()` para separar X e y
    - Selecionar features relevantes (temporais, aeroportos, airline, distância)
    - Aplicar encoding para features categóricas (One-Hot ou Label Encoding)
    - Fazer split treino/teste com test_size=0.2 e stratify por target
    - _Requisitos: 3.1, 3.2_
  
  - [x] 6.2 Treinar múltiplos algoritmos de classificação
    - Implementar função `train_classification_models()` com pelo menos 2 algoritmos
    - Treinar Logistic Regression como baseline
    - Treinar Random Forest Classifier
    - Armazenar modelos treinados e predições
    - _Requisitos: 3.3_
  
  - [x] 6.3 Avaliar modelos de classificação
    - Implementar função `evaluate_classification()` para calcular accuracy, precision, recall, F1
    - Calcular métricas para cada modelo no conjunto de teste
    - Criar tabela comparativa de performance dos modelos
    - _Requisitos: 3.4, 3.6_
  
  - [x] 6.4 Visualizar resultados de classificação
    - Implementar função `plot_confusion_matrix()` para cada modelo
    - Implementar função `plot_feature_importance()` para o melhor modelo
    - Identificar e exibir top 10 features mais importantes
    - _Requisitos: 3.5, 3.7_

- [x] 7. Implementar modelos de regressão de tempo de atraso
  - [x] 7.1 Preparar dados para regressão
    - Filtrar dataset para incluir apenas voos atrasados (ARRIVAL_DELAY > 0)
    - Definir ARRIVAL_DELAY como target contínuo
    - Selecionar e preparar features (mesmas da classificação)
    - Fazer split treino/teste com test_size=0.2
    - _Requisitos: 4.1, 4.2_
  
  - [x] 7.2 Treinar múltiplos algoritmos de regressão
    - Implementar função `train_regression_models()` com pelo menos 2 algoritmos
    - Treinar Linear Regression como baseline
    - Treinar Random Forest Regressor
    - Armazenar modelos treinados e predições
    - _Requisitos: 4.3_
  
  - [x] 7.3 Avaliar modelos de regressão
    - Implementar função `evaluate_regression()` para calcular MAE, RMSE, R²
    - Calcular métricas para cada modelo no conjunto de teste
    - Criar tabela comparativa de performance dos modelos
    - Validar que RMSE >= MAE
    - _Requisitos: 4.4, 4.6_
  
  - [x] 7.4 Visualizar resultados de regressão
    - Implementar função `plot_predictions_vs_actual()` com scatter plot
    - Criar visualização para cada modelo mostrando predições vs valores reais
    - Adicionar linha de referência (predição perfeita)
    - _Requisitos: 4.5_

- [x] 8. Checkpoint - Revisar modelos supervisionados
  - Documentar qual modelo de classificação teve melhor performance e por quê
  - Documentar qual modelo de regressão teve melhor performance e por quê
  - Perguntar ao usuário se há ajustes ou experimentos adicionais necessários

- [x] 9. Implementar clusterização de aeroportos
  - [x] 9.1 Agregar dados por aeroporto
    - Implementar função `aggregate_by_airport()` para criar features de clustering
    - Calcular métricas por aeroporto: total de voos, taxa de atraso, taxa de cancelamento, atraso médio
    - Criar DataFrame agregado com uma linha por aeroporto
    - _Requisitos: 5.1_
  
  - [x] 9.2 Preparar dados para clustering
    - Implementar função `normalize_features()` para standardizar features
    - Aplicar StandardScaler nas features numéricas
    - Validar que features normalizadas têm mean≈0 e std≈1
    - _Requisitos: 5.2_
  
  - [x] 9.3 Determinar número ótimo de clusters
    - Implementar função `find_optimal_clusters()` usando elbow method ou silhouette score
    - Testar diferentes valores de k (2 a 10)
    - Plotar curva de elbow ou silhouette scores
    - Selecionar número ótimo de clusters
    - _Requisitos: 5.4_
  
  - [x] 9.4 Aplicar clustering e interpretar resultados
    - Implementar função `perform_clustering()` com KMeans
    - Atribuir cluster a cada aeroporto
    - Implementar função `interpret_clusters()` para calcular estatísticas descritivas por cluster
    - Documentar características de cada cluster identificado
    - _Requisitos: 5.3, 5.5, 5.7_
  
  - [x] 9.5 Visualizar clusters
    - Implementar função `plot_clusters_2d()` para visualizar clusters em 2D
    - Usar PCA para reduzir dimensionalidade se necessário
    - Colorir pontos por cluster atribuído
    - _Requisitos: 5.6_

- [x] 10. Implementar redução de dimensionalidade com PCA
  - [x] 10.1 Preparar dados para PCA
    - Selecionar apenas features numéricas do dataset preparado
    - Validar que não há valores ausentes nas features selecionadas
    - _Requisitos: 6.1_
  
  - [x] 10.2 Aplicar PCA
    - Implementar função `apply_pca()` com n_components=2
    - Aplicar PCA nas features selecionadas
    - Calcular e exibir explained variance ratio para cada componente
    - Validar que soma de variance ratios <= 1.0
    - _Requisitos: 6.2, 6.3, 6.4_
  
  - [x] 10.3 Visualizar resultados do PCA
    - Criar scatter plot 2D dos componentes principais
    - Se clusters existem, colorir pontos por cluster atribuído
    - Adicionar labels dos eixos com variance explained
    - _Requisitos: 6.5, 6.6_

- [x] 11. Documentar resultados e conclusões
  - [x] 11.1 Documentar insights da EDA
    - Criar seção markdown com principais descobertas da análise exploratória
    - Identificar aeroportos críticos com maiores taxas de atraso
    - Documentar padrões temporais identificados (horários, dias, meses)
    - _Requisitos: 7.1, 7.6_
  
  - [x] 11.2 Documentar resultados de modelos supervisionados
    - Documentar qual modelo de classificação teve melhor performance
    - Documentar qual modelo de regressão teve melhor performance
    - Explicar por que esses modelos foram superiores
    - Documentar capacidade preditiva alcançada
    - _Requisitos: 7.2, 7.6_
  
  - [x] 11.3 Documentar resultados de aprendizado não supervisionado
    - Documentar interpretação dos clusters identificados
    - Explicar o que cada cluster representa em termos de perfil de aeroporto
    - Documentar insights do PCA sobre estrutura dos dados
    - _Requisitos: 7.3_
  
  - [x] 11.4 Documentar limitações e melhorias futuras
    - Identificar e documentar pelo menos 3 limitações dos modelos ou análise
    - Propor pelo menos 3 melhorias ou próximos passos
    - Considerar limitações de dados, features, algoritmos e validação
    - _Requisitos: 7.4, 7.5_

- [x] 12. Finalizar organização do projeto
  - [x] 12.1 Organizar notebooks
    - Adicionar células markdown explicando cada etapa
    - Criar índice ou cabeçalhos de seção claros
    - Garantir que todas as visualizações são exibidas inline
    - Testar execução sequencial completa do notebook
    - _Requisitos: 8.2, 8.3, 8.4, 8.5_
  
  - [x] 12.2 Preparar arquivos de entrega
    - Atualizar README com descrição completa, setup e instruções de execução
    - Verificar que requirements.txt está completo e atualizado
    - Organizar estrutura de diretórios (data/, notebooks/, docs/)
    - Adicionar instruções para download dos datasets se necessário
    - _Requisitos: 8.6, 9.1, 9.4_
  
  - [x] 12.3 Validação final
    - Executar notebook completo do início ao fim sem erros
    - Verificar que todas as visualizações são geradas corretamente
    - Confirmar que resultados são reproduzíveis (seeds fixados)
    - Revisar documentação e conclusões
    - _Requisitos: 9.5, 9.6_

- [x] 13. Checkpoint final
  - Garantir que todos os testes passam e não há erros
  - Perguntar ao usuário se há dúvidas ou ajustes finais necessários

## Notas

- Este é um projeto acadêmico focado em exploração e experimentação
- Não há testes automatizados - validação é feita por inspeção manual dos outputs
- Todas as tarefas devem ser implementadas em Python usando Jupyter Notebooks
- Usar RANDOM_STATE=42 em todas as operações aleatórias para reprodutibilidade
- Priorizar clareza e documentação sobre otimização de performance
- Dataset principal (flights.csv) é grande (592 MB) - usar dtypes otimizados
- Cada tarefa referencia requisitos específicos para rastreabilidade
