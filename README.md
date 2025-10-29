# Análise e Previsão de Qualidade do Ar (Demo)

Projeto demonstrativo para coletar, pré-processar, treinar e exibir previsões de qualidade do ar (PM2.5) em tempo quase real.

Estrutura proposta:

- src/
  - collector.py # OpenAQ client; busca leituras recentes
  - preprocess.py # limpeza, criação de features e escalonamento
  - train.py # script de treino (DecisionTree/LinearRegression)
  - predict.py # carregamento de modelo + função de inferência
- app/
  - streamlit_app.py # dashboard interativo usando Streamlit
- requirements.txt

Como usar (resumo):

1. Instale dependências: pip install -r requirements.txt
2. Treine um modelo (opcional): python -m src.train --pollutant pm25 --model decision_tree
3. Rode o dashboard: streamlit run app/streamlit_app.py

Notas:

- O projeto usa OpenAQ (https://docs.openaq.org/) como fonte primária.
- Se não houver CSV histórico disponível, o treino tenta usar leituras públicas ou um conjunto sintético mínimo como fallback.

Ver o código em `src/` e `app/` para detalhes de implementação.
