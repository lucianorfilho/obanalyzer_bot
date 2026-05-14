import pandas as pd
import numpy as np

def load_csv(file) -> pd.DataFrame:
    """
    Aceita CSV do HistData.com (formato M1).
    Colunas esperadas: datetime, open, high, low, close, volume
    Separador: ; ou ,
    """
    try:
        df = pd.read_csv(
            file,
            sep=None,
            engine='python',
            names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
            header=None,
            parse_dates=['datetime'],
            infer_datetime_format=True
        )
    except Exception:
        df = pd.read_csv(
            file,
            sep=',',
            parse_dates=[0],
            index_col=0
        )
        df.index.name = 'datetime'
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.reset_index(inplace=True)

    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df


def resample_timeframes(df_m1: pd.DataFrame) -> dict:
    """
    Recebe DataFrame M1 e retorna dict com os 4 timeframes.
    """
    def _resample(rule):
        return df_m1.resample(rule).agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum'
        }).dropna()

    return {
        '1m':  df_m1.copy(),
        '5m':  _resample('5T'),
        '15m': _resample('15T'),
        '30m': _resample('30T')
    }


def split_train_test(df: pd.DataFrame, train_days=25, test_days=5):
    """
    Separa os dados:
    - treino: 25 dias mais antigos do recorte
    - teste:  últimos 5 dias
    """
    end   = df.index.max()
    start = end - pd.Timedelta(days=train_days + test_days)

    df_window = df[df.index >= start]

    cutoff = end - pd.Timedelta(days=test_days)

    df_train = df_window[df_window.index <= cutoff]
    df_test  = df_window[df_window.index >  cutoff]

    return df_train, df_test


def get_summary(df: pd.DataFrame) -> dict:
    return {
        'total_candles': len(df),
        'inicio':        str(df.index.min()),
        'fim':           str(df.index.max()),
        'dias_cobertos': (df.index.max() - df.index.min()).days,
        'preco_minimo':  df['low'].min(),
        'preco_maximo':  df['high'].max(),
        'variacao_pct':  round(
            (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100, 4
        )
    }