import pandas as pd
import numpy as np


def generate_signals_tomorrow(df_full: pd.DataFrame, model,
                               feature_cols: list,
                               prob_threshold=0.60) -> pd.DataFrame:
    """
    Usa os últimos candles do dataset para gerar sinais do próximo período.
    Na prática, pega as N últimas linhas com alta probabilidade.
    """
    X = df_full[feature_cols]
    proba = model.predict_proba(X)[:, 1]

    df_signals = df_full[['open', 'high', 'low', 'close']].copy()
    df_signals['proba_call'] = proba
    df_signals['proba_put']  = 1 - proba
    df_signals['sinal'] = 'NEUTRO'
    df_signals.loc[proba >= prob_threshold, 'sinal'] = 'CALL'
    df_signals.loc[proba <= (1 - prob_threshold), 'sinal'] = 'PUT'

    # Filtrar últimas 24h de dados (próximo "dia operacional")
    last_24h = df_signals[
        df_signals.index >= df_signals.index.max() - pd.Timedelta(hours=24)
    ]

    return last_24h[last_24h['sinal'] != 'NEUTRO'].sort_index()