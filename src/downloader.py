import pandas as pd
import requests
from datetime import datetime, timedelta
import time

def download_eurusd_twelvedata(
    api_key: str,
    interval: str = '5min',
    days: int = 30
) -> pd.DataFrame:
    """
    Baixa dados EUR/USD via Twelve Data (gratuito).
    interval: '1min', '5min', '15min', '30min', '1h'
    days: quantos dias de histórico buscar
    Plano gratuito: 800 req/dia, 8 req/min
    """

    all_chunks = []
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    # Twelve Data retorna até 5000 candles por chamada
    # Para 30 dias de 5min = ~8640 candles
    # Fazemos em 2 blocos de 15 dias

    chunk_days = 15
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)

        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol':     'EUR/USD',
            'interval':   interval,
            'start_date': current_start.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date':   current_end.strftime('%Y-%m-%d %H:%M:%S'),
            'outputsize': 5000,
            'apikey':     api_key,
            'format':     'JSON',
            'timezone':   'UTC'
        }

        print(f"Baixando {current_start.strftime('%d/%m')} → {current_end.strftime('%d/%m')}...")
        r = requests.get(url, params=params, timeout=30)

        if r.status_code != 200:
            raise ConnectionError(f"Erro HTTP {r.status_code}: {r.text[:200]}")

        data = r.json()

        if data.get('status') == 'error':
            raise ValueError(f"Erro Twelve Data: {data.get('message', str(data))}")

        values = data.get('values', [])
        if values:
            df_chunk = pd.DataFrame(values)
            df_chunk['datetime'] = pd.to_datetime(df_chunk['datetime'])
            df_chunk.set_index('datetime', inplace=True)
            df_chunk = df_chunk.rename(columns={
                'open':   'open',
                'high':   'high',
                'low':    'low',
                'close':  'close',
                'volume': 'volume'
            })
            if 'volume' not in df_chunk.columns:
                df_chunk['volume'] = 0
            df_chunk = df_chunk[['open', 'high', 'low', 'close', 'volume']]
            df_chunk = df_chunk.apply(pd.to_numeric, errors='coerce')
            df_chunk.dropna(inplace=True)
            all_chunks.append(df_chunk)

        current_start = current_end
        time.sleep(1)  # respeitar limite de 8 req/min

    if not all_chunks:
        raise ValueError("Nenhum dado retornado pelo Twelve Data.")

    df = pd.concat(all_chunks)
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # Remove timezone se houver
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    print(f"✅ {len(df)} candles | {df.index.min()} → {df.index.max()}")
    return df