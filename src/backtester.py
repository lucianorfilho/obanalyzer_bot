import numpy as np
import pandas as pd


def run_backtest(df_test: pd.DataFrame, model, feature_cols: list,
                 payout=0.80, prob_threshold=0.60, stake=1.0) -> dict:
    """
    Simula as operações binárias nos últimos 5 dias.
    Retorna métricas detalhadas e histórico de trades.
    """
    X = df_test[feature_cols]
    y = df_test['target'].values
    proba = model.predict_proba(X)[:, 1]

    signals = np.full(len(y), -1)
    signals[proba >= prob_threshold]       = 1   # CALL
    signals[proba <= (1 - prob_threshold)] = 0   # PUT

    mask = signals != -1

    trades = []
    capital_curve = [0.0]
    capital = 0.0

    for i in range(len(y)):
        if not mask[i]:
            continue

        truth = y[i]
        sig   = signals[i]
        prob  = proba[i] if sig == 1 else (1 - proba[i])

        win   = (sig == truth)
        profit = payout * stake if win else -stake

        capital += profit
        capital_curve.append(capital)

        trades.append({
            'datetime':   df_test.index[i],
            'sinal':      'CALL' if sig == 1 else 'PUT',
            'confianca':  round(prob * 100, 2),
            'resultado':  'WIN' if win else 'LOSS',
            'lucro':      round(profit, 4),
            'capital':    round(capital, 4)
        })

    df_trades = pd.DataFrame(trades)

    if df_trades.empty:
        return {'error': 'Nenhuma operação gerada com esse threshold.'}

    wins  = df_trades[df_trades['resultado'] == 'WIN']
    losses = df_trades[df_trades['resultado'] == 'LOSS']

    win_rate   = len(wins) / len(df_trades)
    total_gain = wins['lucro'].sum()
    total_loss = abs(losses['lucro'].sum())
    profit_factor = total_gain / total_loss if total_loss > 0 else np.inf

    # Drawdown máximo
    cap_arr = np.array(capital_curve)
    peak = np.maximum.accumulate(cap_arr)
    dd   = cap_arr - peak
    max_dd = dd.min()

    return {
        'total_trades':   len(df_trades),
        'wins':           len(wins),
        'losses':         len(losses),
        'win_rate':       round(win_rate * 100, 2),
        'lucro_total':    round(capital, 4),
        'profit_factor':  round(profit_factor, 3),
        'max_drawdown':   round(max_dd, 4),
        'capital_curve':  capital_curve,
        'df_trades':      df_trades
    }