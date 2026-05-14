\# 📈 Binary Options Analyzer — EUR/USD



Sistema de análise multi-timeframe com Machine Learning para operações binárias.



\## Funcionalidades

\- Upload de dados históricos M1 (HistData.com)

\- Análise de confluência: 5m + 15m + 30m

\- Comparação de modelos: XGBoost, LightGBM, RandomForest

\- Tuning de hiperparâmetros com Optuna

\- Backtest específico para binárias (payout, win rate, drawdown)

\- Geração de sinais de alta probabilidade



\## Rodar Localmente

```bash

pip install -r requirements.txt

streamlit run app.py

```



\## Deploy no Railway

1\. Crie um projeto no Railway (railway.app)

2\. Adicione o token em GitHub > Settings > Secrets > RAILWAY\_TOKEN

3\. Push para o branch main — deploy automático

