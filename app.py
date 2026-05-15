import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

from src.data_loader import resample_timeframes, split_train_test, get_summary
from src.feature_engineering import build_multi_timeframe_dataset, get_feature_cols
from src.model_trainer import train_and_evaluate, tune_best_model
from src.backtester import run_backtest
from src.predictor import generate_signals_tomorrow
from src.downloader import download_eurusd_twelvedata

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OBAnalyzer Bot",
    page_icon="📈",
    layout="wide"
)

DATA_PATH  = "data/raw/eurusd_m1.csv"
MODEL_PATH = "models/best_model.pkl"
META_PATH  = "models/model_meta.pkl"

os.makedirs("data/raw", exist_ok=True)
os.makedirs("models",   exist_ok=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("📈 OBAnalyzer Bot — EUR/USD")
st.caption("Operações Binárias • Multi-timeframe • ML • Sinais Diários")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("📅 Dados")
    download_days = st.slider("Dias para baixar", 20, 60, 30)

    st.subheader("🧠 Treinamento")
    train_days     = st.slider("Dias de Treino",  10, 25, 20)
    test_days      = st.slider("Dias de Teste",    3, 10,  5)
    threshold_bp   = st.slider("Threshold (basis points)", 1, 10, 3)

    st.subheader("🎯 Operação")
    prob_threshold = st.slider("Confiança mínima (%)", 50, 90, 65) / 100
    payout         = st.slider("Payout da corretora (%)", 60, 95, 80) / 100
    stake          = st.number_input("Stake por operação ($)", 1.0, 100.0, 1.0)

    st.subheader("🔧 Modelo")
    model_choice = st.selectbox("Modelo para Tuning", ['XGBoost', 'LightGBM'])
    n_trials     = st.slider("Trials Optuna", 10, 100, 30)

    st.divider()
    st.subheader("🔑 API")
    td_key = st.text_input(
        "Twelve Data API Key",
        type="password",
        help="Gratuito em twelvedata.com — 800 req/dia"
    )

    st.divider()
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        meta = joblib.load(META_PATH)
        st.success(f"✅ Modelo treinado em:\n{meta.get('trained_at', '—')}")
        st.info(
            f"Modelo: **{meta.get('model_name', '—')}**\n\n"
            f"Win Rate Backtest: **{meta.get('win_rate', '—')}%**"
        )
    else:
        st.warning("⚠️ Nenhum modelo treinado ainda.\nRode o modo Fim de Semana.")

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🗓️ Fim de Semana — Treinar Modelo",
    "📅 Semana — Gerar Sinais do Dia"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — FIM DE SEMANA
# ══════════════════════════════════════════════════════════
with tab1:

    # ── ETAPA 1: Baixar Dados
    st.header("📥 Etapa 1 — Baixar Dados EUR/USD")

    if os.path.exists(DATA_PATH):
        df_check = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        st.info(
            f"Dados atuais: **{len(df_check):,} candles** | "
            f"De {df_check.index.min().strftime('%d/%m/%Y')} "
            f"até {df_check.index.max().strftime('%d/%m/%Y')}"
        )
    else:
        st.warning("Nenhum dado local encontrado. Clique em Baixar.")

    if st.button("⬇️ Baixar dados agora (EUR/USD 5m via Twelve Data)"):
        if not td_key:
            st.error("❌ Insira sua Twelve Data API Key no sidebar.")
            st.stop()

        with st.spinner(f"Baixando últimos {download_days} dias de EUR/USD 5m..."):
            try:
                df_raw = download_eurusd_twelvedata(
                    api_key=td_key,
                    interval='5min',
                    days=download_days
                )

                if df_raw.index.tz is not None:
                    df_raw.index = df_raw.index.tz_localize(None)

                df_raw.to_csv(DATA_PATH)
                summary = get_summary(df_raw)

                st.success(
                    f"✅ Download concluído! "
                    f"**{summary['total_candles']:,} candles** | "
                    f"{summary['dias_cobertos']} dias | "
                    f"De {df_raw.index.min().strftime('%d/%m/%Y')} "
                    f"até {df_raw.index.max().strftime('%d/%m/%Y')}"
                )

                df_prev = df_raw.last('5D')
                fig_prev = go.Figure(go.Candlestick(
                    x=df_prev.index,
                    open=df_prev['open'],
                    high=df_prev['high'],
                    low=df_prev['low'],
                    close=df_prev['close'],
                    name='EUR/USD 5m'
                ))
                fig_prev.update_layout(
                    title='Preview — EUR/USD últimos 5 dias (5m)',
                    xaxis_rangeslider_visible=False,
                    height=350
                )
                st.plotly_chart(fig_prev, use_container_width=True)

            except Exception as e:
                st.error(f"Erro ao baixar: {e}")

    st.divider()

    # ── ETAPA 2: Treino
    st.header("🧠 Etapa 2 — Treinar Modelo")

    if not os.path.exists(DATA_PATH):
        st.warning("⚠️ Baixe os dados primeiro (Etapa 1).")
    else:
        if st.button("▶️ Iniciar Treinamento"):
            with st.spinner("Carregando e preparando dados..."):
                df_base = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
                df_base.index.name = 'datetime'
                tfs = resample_timeframes(df_base)

                tfs_train = {
                    '5m':  split_train_test(tfs['5m'],  train_days, test_days)[0],
                    '15m': split_train_test(tfs['15m'], train_days, test_days)[0],
                    '30m': split_train_test(tfs['30m'], train_days, test_days)[0],
                }
                tfs_test = {
                    '5m':  split_train_test(tfs['5m'],  train_days, test_days)[1],
                    '15m': split_train_test(tfs['15m'], train_days, test_days)[1],
                    '30m': split_train_test(tfs['30m'], train_days, test_days)[1],
                }

                df_train = build_multi_timeframe_dataset(tfs_train, threshold_bp)
                df_test  = build_multi_timeframe_dataset(tfs_test,  threshold_bp)
                feature_cols = get_feature_cols(df_train)

            with st.spinner("Treinando e comparando modelos..."):
                results = train_and_evaluate(
                    df_train, feature_cols,
                    prob_threshold=prob_threshold
                )

            # Tabela comparativa
            st.subheader("📊 Comparativo de Modelos")
            rows = []
            for name, r in results.items():
                rows.append({
                    'Modelo':         name,
                    'Acurácia Média': f"{r['avg_acc']*100:.2f}%",
                    'MCC Médio':      f"{r['avg_mcc']:.4f}",
                    'AUC Médio':      f"{r['avg_auc']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            best_name = max(results, key=lambda k: results[k]['avg_mcc'])
            st.success(f"✅ Melhor modelo: **{best_name}**")

            # Gráfico por fold
            df_folds = pd.DataFrame(results[best_name]['fold_metrics'])
            fig_folds = go.Figure()
            fig_folds.add_trace(go.Bar(
                x=df_folds['fold'].astype(str),
                y=df_folds['acc'],
                name='Acurácia',
                marker_color='steelblue'
            ))
            fig_folds.add_trace(go.Scatter(
                x=df_folds['fold'].astype(str),
                y=df_folds['mcc'],
                name='MCC',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='orange', width=2)
            ))
            fig_folds.update_layout(
                title=f'{best_name} — Métricas por Fold',
                yaxis=dict(title='Acurácia'),
                yaxis2=dict(title='MCC', overlaying='y', side='right'),
                height=350
            )
            st.plotly_chart(fig_folds, use_container_width=True)

            # Tuning Optuna
            st.subheader(f"🔧 Tuning — {model_choice}")
            with st.spinner(f"Rodando {n_trials} trials Optuna..."):
                best_model, best_params, study = tune_best_model(
                    df_train, feature_cols,
                    model_name=model_choice,
                    n_trials=n_trials
                )
            st.success("✅ Tuning concluído!")
            st.json(best_params)

            trials_df = study.trials_dataframe()
            fig_opt = px.line(
                trials_df, x='number', y='value',
                title='Optuna — Evolução AUC por Trial',
                labels={'number': 'Trial', 'value': 'AUC'},
                color_discrete_sequence=['darkorange']
            )
            st.plotly_chart(fig_opt, use_container_width=True)

            # Salvar modelo e metadados
            joblib.dump(best_model, MODEL_PATH)
            joblib.dump({
                'feature_cols':   feature_cols,
                'threshold_bp':   threshold_bp,
                'prob_threshold': prob_threshold,
                'payout':         payout,
                'model_name':     model_choice,
                'trained_at':     datetime.now().strftime('%d/%m/%Y %H:%M')
            }, META_PATH)

            st.divider()

            # ── ETAPA 3: Backtest
            st.header("🧪 Etapa 3 — Backtest (últimos 5 dias)")
            with st.spinner("Simulando operações binárias..."):
                bt = run_backtest(
                    df_test, best_model, feature_cols,
                    payout=payout,
                    prob_threshold=prob_threshold,
                    stake=stake
                )

            if 'error' in bt:
                st.error(bt['error'])
            else:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Trades",  bt['total_trades'])
                c2.metric("Win Rate",      f"{bt['win_rate']}%")
                c3.metric("Lucro Total",   f"${bt['lucro_total']:.2f}")
                c4.metric("Profit Factor", bt['profit_factor'])
                c5.metric("Max Drawdown",  f"${bt['max_drawdown']:.2f}")

                # Curva de capital
                fig_cap = go.Figure(go.Scatter(
                    y=bt['capital_curve'],
                    mode='lines',
                    line=dict(color='green', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,200,0,0.1)'
                ))
                fig_cap.update_layout(
                    title='Curva de Capital — Backtest',
                    xaxis_title='Operações',
                    yaxis_title='Lucro ($)',
                    height=300
                )
                st.plotly_chart(fig_cap, use_container_width=True)

                # Candles + sinais
                df_trades = bt['df_trades']
                df_plot   = df_test.last(f'{test_days}D')

                fig_bt = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.75, 0.25],
                    vertical_spacing=0.02
                )
                fig_bt.add_trace(go.Candlestick(
                    x=df_plot.index,
                    open=df_plot['open'], high=df_plot['high'],
                    low=df_plot['low'],   close=df_plot['close'],
                    name='EUR/USD'
                ), row=1, col=1)

                calls = df_trades[df_trades['sinal'] == 'CALL']
                puts  = df_trades[df_trades['sinal'] == 'PUT']

                if not calls.empty:
                    fig_bt.add_trace(go.Scatter(
                        x=calls['datetime'],
                        y=calls['datetime'].map(
                            lambda d: df_plot['low'].asof(d) * 0.9999
                        ),
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='lime'),
                        name='CALL'
                    ), row=1, col=1)

                if not puts.empty:
                    fig_bt.add_trace(go.Scatter(
                        x=puts['datetime'],
                        y=puts['datetime'].map(
                            lambda d: df_plot['high'].asof(d) * 1.0001
                        ),
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=12, color='red'),
                        name='PUT'
                    ), row=1, col=1)

                fig_bt.add_trace(go.Bar(
                    x=df_trades['datetime'],
                    y=df_trades['lucro'],
                    marker_color=df_trades['resultado'].map(
                        {'WIN': 'green', 'LOSS': 'red'}
                    ),
                    name='P&L'
                ), row=2, col=1)

                fig_bt.update_layout(
                    title='Backtest — Sinais e Resultado',
                    xaxis_rangeslider_visible=False,
                    height=550
                )
                st.plotly_chart(fig_bt, use_container_width=True)

                st.subheader("📋 Histórico de Operações")
                st.dataframe(bt['df_trades'], use_container_width=True)

                # Salvar win_rate no meta
                meta = joblib.load(META_PATH)
                meta['win_rate'] = bt['win_rate']
                joblib.dump(meta, META_PATH)

                st.balloons()
                st.success(
                    "✅ Modelo treinado e salvo! "
                    "Vá para a aba 'Semana' para gerar sinais diários."
                )

# ══════════════════════════════════════════════════════════
# TAB 2 — MODO SEMANA
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("🔮 Gerar Sinais para Amanhã")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        st.warning(
            "⚠️ Nenhum modelo treinado. "
            "Vá para a aba 'Fim de Semana' primeiro."
        )
    else:
        meta         = joblib.load(META_PATH)
        best_model   = joblib.load(MODEL_PATH)
        feature_cols = meta['feature_cols']

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Modelo",      meta.get('model_name', '—'))
        col_m2.metric("Treinado em", meta.get('trained_at', '—'))
        col_m3.metric("Win Rate BT", f"{meta.get('win_rate', '—')}%")

        st.divider()

        if st.button("🚀 Gerar Sinais para Amanhã"):
            with st.spinner("Analisando dados e gerando sinais..."):
                df_live = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
                df_live.index.name = 'datetime'
                tfs_live = resample_timeframes(df_live)

                df_full = build_multi_timeframe_dataset(
                    {
                        '5m':  tfs_live['5m'],
                        '15m': tfs_live['15m'],
                        '30m': tfs_live['30m'],
                    },
                    meta['threshold_bp']
                )

                signals_df = generate_signals_tomorrow(
                    df_full, best_model, feature_cols,
                    prob_threshold=meta['prob_threshold']
                )

            if signals_df.empty:
                st.warning(
                    "Nenhum sinal encontrado. "
                    "Tente reduzir a confiança mínima no sidebar."
                )
            else:
                st.success(f"✅ {len(signals_df)} sinais encontrados!")

                calls_c = (signals_df['sinal'] == 'CALL').sum()
                puts_c  = (signals_df['sinal'] == 'PUT').sum()
                max_c   = signals_df[['proba_call', 'proba_put']].max(axis=1).max()

                c1, c2, c3 = st.columns(3)
                c1.metric("Sinais CALL",     calls_c)
                c2.metric("Sinais PUT",      puts_c)
                c3.metric("Maior Confiança", f"{max_c*100:.1f}%")

                st.subheader("📋 Sinais de Alta Probabilidade")
                df_show = signals_df[['close', 'sinal', 'proba_call', 'proba_put']].copy()
                df_show['proba_call'] = (df_show['proba_call'] * 100).round(2)
                df_show['proba_put']  = (df_show['proba_put']  * 100).round(2)
                df_show.columns       = ['Preço', 'Sinal', 'Prob CALL (%)', 'Prob PUT (%)']
                df_show.index.name    = 'Horário'
                st.dataframe(df_show, use_container_width=True)

                # Gráfico de sinais
                df_plot = df_full[['open', 'high', 'low', 'close']].last('3D')
                calls_s = signals_df[signals_df['sinal'] == 'CALL']
                puts_s  = signals_df[signals_df['sinal'] == 'PUT']

                fig_sig = go.Figure()
                fig_sig.add_trace(go.Candlestick(
                    x=df_plot.index,
                    open=df_plot['open'], high=df_plot['high'],
                    low=df_plot['low'],   close=df_plot['close'],
                    name='EUR/USD 5m'
                ))

                if not calls_s.empty:
                    fig_sig.add_trace(go.Scatter(
                        x=calls_s.index,
                        y=calls_s['close'] * 0.9999,
                        mode='markers+text',
                        marker=dict(symbol='triangle-up', size=15, color='lime'),
                        text=[f"CALL {p*100:.0f}%" for p in calls_s['proba_call']],
                        textposition='bottom center',
                        name='CALL'
                    ))

                if not puts_s.empty:
                    fig_sig.add_trace(go.Scatter(
                        x=puts_s.index,
                        y=puts_s['close'] * 1.0001,
                        mode='markers+text',
                        marker=dict(symbol='triangle-down', size=15, color='red'),
                        text=[f"PUT {p*100:.0f}%" for p in puts_s['proba_put']],
                        textposition='top center',
                        name='PUT'
                    ))

                fig_sig.update_layout(
                    title='Sinais de Alta Probabilidade — Próximas Horas',
                    xaxis_rangeslider_visible=False,
                    height=500
                )
                st.plotly_chart(fig_sig, use_container_width=True)

                # Histograma de confiança
                confs = signals_df[['proba_call', 'proba_put']].max(axis=1) * 100
                fig_conf = px.histogram(
                    confs, nbins=20,
                    title='Distribuição de Confiança dos Sinais',
                    labels={'value': 'Confiança (%)', 'count': 'Qtd'},
                    color_discrete_sequence=['steelblue']
                )
                st.plotly_chart(fig_conf, use_container_width=True)