import pandas as pd
import numpy as np
import ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Tendência ---
    df['ema_9']  = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['macd']   = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_diff']   = ta.trend.macd_diff(df['close'])

    # --- Momento ---
    df['rsi']   = ta.momentum.rsi(df['close'], window=14)
    df['stoch'] = ta.momentum.stoch(
        df['high'], df['low'], df['close'], window=14, smooth_window=3
    )
    df['stoch_signal'] = ta.momentum.stoch_signal(
        df['high'], df['low'], df['close'], window=14, smooth_window=3
    )

    # --- Volatilidade ---
    df['bb_high']  = ta.volatility.bollinger_hband(df['close'], window=20)
    df['bb_low']   = ta.volatility.bollinger_lband(df['close'], window=20)
    df['bb_mid']   = ta.volatility.bollinger_mavg(df['close'], window=20)
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'], window=20)
    df['atr']      = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=14
    )

    # --- Retornos ---
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_6'] = df['close'].pct_change(6)

    # --- Range relativo ---
    df['range_rel'] = (df['high'] - df['low']) / df['close']

    # --- Posição do close dentro da vela ---
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

    return df


def build_multi_timeframe_dataset(tfs: dict, threshold_bp=3) -> pd.DataFrame:
    """
    tfs: dict retornado por resample_timeframes(), já com dados de treino ou teste.
    Gera o dataset final com features multi-timeframe + target.
    """
    # Features no 5m (base)
    df_5m  = add_indicators(tfs['5m'])

    # Features no 15m e 30m (contexto de tendência)
    df_15m = add_indicators(tfs['15m'])
    df_30m = add_indicators(tfs['30m'])

    # Alinhar 15m e 30m no índice 5m
    cols_context = [
        'ema_9', 'ema_21', 'ema_50',
        'macd_diff', 'rsi', 'atr',
        'bb_width', 'stoch', 'ret_3'
    ]

    tf15 = df_15m[cols_context].reindex(df_5m.index, method='ffill')
    tf30 = df_30m[cols_context].reindex(df_5m.index, method='ffill')

    for c in cols_context:
        df_5m[f'15m_{c}'] = tf15[c]
        df_5m[f'30m_{c}'] = tf30[c]

    # Target binário (UP=1, DOWN=0)
    future = df_5m['close'].shift(-1)
    ret_bp = (future / df_5m['close'] - 1) * 10000

    df_5m['target'] = np.nan
    df_5m.loc[ret_bp >  threshold_bp, 'target'] = 1
    df_5m.loc[ret_bp < -threshold_bp, 'target'] = 0

    df_5m.dropna(inplace=True)
    df_5m['target'] = df_5m['target'].astype(int)

    return df_5m


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target']
    return [c for c in df.columns if c not in exclude]