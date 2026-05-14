import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_loader import (
    load_csv, resample_timeframes, split_train_test, get_summary
)
from src.feature_engineering import build_multi_timeframe_dataset, get_feature_cols
from src.model_trainer import train_and_evaluate, tune_best_model, load_model
from src.backtester import run_backtest
from src.predictor import generate_signals_tomorrow

# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Binary Options Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Binary Options Analyzer — EUR/USD")
st.caption("Multi-timeframe • ML • Backtest • Sinais")

# ─────────────────────────────────────────────────────────
# SIDEBAR — Configurações
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")
    train_days     = st.slider("Dias de Treino",  10, 60, 25)
    test_days      = st.slider("Dias de Teste",    3, 15,  5)
    threshold_bp   = st.slider("Threshold (basis points)", 1, 10, 3)
    prob_threshold = st.slider("Confiança mínima (%)", 50, 90, 60) / 100
    payout         = st.slider("Payout da corretora (%)", 60, 95, 80) / 100
    stake          = st.number_input("Stake por operação ($)", 1.0, 100.0, 1.0)
    model_tune     = st.selectbox("Modelo para Tuning", ['XGBoost', 'LightGBM'])
    n_trials       = st.slider("Trials Optuna", 10, 100, 30)

# ─────────────────────────────────────────────────────────
# ETAPA 1 — Upload e Visualização dos Dados
# ─────────────────────────────────────────────────────────
st.header("📂 Etapa 1 — Carregar Dados Históricos")

uploaded = st.file_uploader(
    "Envie o CSV do EURUSD M1 (HistData.com ou similar)",
    type=['csv', 'txt']
)

if uploaded:
    with st.spinner("Carregando e processando dados..."):
        df_m1 = load_csv(uploaded)
        tfs   = resample_timeframes(df_m1)
        summary = get_summary(df_m1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candles M1", f"{summary['total_candles']:,}")
    col2.metric("Período",          f"{summary['dias_cobertos']} dias")
    col3.metric("Mín / Máx",       f"{summary['preco_minimo']:.5f} / {summary['preco_maximo']:.5f}")
    col4.metric("Variação Total",   f"{summary['variacao_pct']}%")

    # Gráfico de candlestick geral
    df_vis = tfs['5m'].last('30D')
    fig_main = go.Figure(go.Candlestick(
        x=df_vis.index,
        open=df_vis['open'], high=df_vis['high'],
        low=df_vis['low'],   close=df_vis['close'],
        name='EUR/USD 5m'
    ))
    fig_main.update_layout(
        title='EUR/USD — Últimos 30 dias (5m)',
        xaxis_rangeslider_visible=False,
        height=400
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # ETAPA 2 — Treino + Validação Cruzada
    # ─────────────────────────────────────────────────────────
    st.header("🧠 Etapa 2 — Treinamento (25 dias) + Validação Cruzada")

    if st.button("▶️ Iniciar Treinamento"):
        with st.spinner("Preparando features multi-timeframe..."):
            df_train_raw, df_test_raw = split_train_test(
                tfs['5m'], train_days=train_days, test_days=test_days
            )
            _, df_test_raw_15m = split_train_test(tfs['15m'], train_days, test_days)
            _, df_test_raw_30m = split_train_test(tfs['30m'], train_days, test_days)

            tfs_train = {
                '5m':  df_train_raw,
                '15m': split_train_test(tfs['15m'], train_days, test_days)[0],
                '30m': split_train_test(tfs['30m'], train_days, test_days)[0]
            }
            tfs_test = {
                '5m':  df_test_raw,
                '15m': df_test_raw_15m,
                '30m': df_test_raw_30m
            }

            df_train = build_multi_timeframe_dataset(tfs_train, threshold_bp)
            df_test  = build_multi_timeframe_dataset(tfs_test,  threshold_bp)
            feature_cols = get_feature_cols(df_train)

        with st.spinner("Treinando modelos com TimeSeriesSplit..."):
            results = train_and_evaluate(
                df_train, feature_cols,
                prob_threshold=prob_threshold
            )

        st.session_state['results']      = results
        st.session_state['df_train']     = df_train
        st.session_state['df_test']      = df_test
        st.session_state['feature_cols'] = feature_cols

        # ── Tabela comparativa de modelos
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

        # ── Gráfico de métricas por fold (XGBoost como exemplo)
        best_model_name = max(results, key=lambda k: results[k]['avg_mcc'])
        st.success(f"✅ Melhor modelo na validação cruzada: **{best_model_name}**")

        folds_data = results[best_model_name]['fold_metrics']
        df_folds = pd.DataFrame(folds_data)

        fig_folds = go.Figure()
        fig_folds.add_trace(go.Bar(
            x=df_folds['fold'].astype(str), y=df_folds['acc'],
            name='Acurácia', marker_color='steelblue'
        ))
        fig_folds.add_trace(go.Scatter(
            x=df_folds['fold'].astype(str), y=df_folds['mcc'],
            name='MCC', mode='lines+markers', yaxis='y2',
            line=dict(color='orange', width=2)
        ))
        fig_folds.update_layout(
            title=f'{best_model_name} — Métricas por Fold',
            yaxis=dict(title='Acurácia'),
            yaxis2=dict(title='MCC', overlaying='y', side='right'),
            height=350
        )
        st.plotly_chart(fig_folds, use_container_width=True)

        # ── Tuning com Optuna
        st.subheader(f"🔧 Tuning de Hiperparâmetros — {model_tune}")
        with st.spinner(f"Rodando {n_trials} trials com Optuna..."):
            best_model, best_params, study = tune_best_model(
                df_train, feature_cols,
                model_name=model_tune,
                n_trials=n_trials
            )
        st.success("✅ Tuning concluído!")
        st.json(best_params)

        # Curva de otimização Optuna
        trials_df = study.trials_dataframe()
        fig_optuna = px.line(
            trials_df, x='number', y='value',
            title='Optuna — Evolução do AUC por Trial',
            labels={'number': 'Trial', 'value': 'AUC'},
            color_discrete_sequence=['darkorange']
        )
        st.plotly_chart(fig_optuna, use_container_width=True)

        st.session_state['best_model']      = best_model
        st.session_state['best_model_name'] = model_tune

    # ─────────────────────────────────────────────────────────
    # ETAPA 3 — Backtest (últimos 5 dias)
    # ─────────────────────────────────────────────────────────
    if 'best_model' in st.session_state:
        st.header("🧪 Etapa 3 — Backtest (últimos 5 dias)")

        if st.button("▶️ Rodar Backtest"):
            with st.spinner("Simulando operações binárias..."):
                bt = run_backtest(
                    st.session_state['df_test'],
                    st.session_state['best_model'],
                    st.session_state['feature_cols'],
                    payout=payout,
                    prob_threshold=prob_threshold,
                    stake=stake
                )
            st.session_state['backtest'] = bt

            if 'error' in bt:
                st.error(bt['error'])
            else:
                # ── Métricas principais
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Trades",    bt['total_trades'])
                c2.metric("Win Rate",        f"{bt['win_rate']}%")
                c3.metric("Lucro Total",     f"${bt['lucro_total']:.2f}")
                c4.metric("Profit Factor",   bt['profit_factor'])
                c5.metric("Max Drawdown",    f"${bt['max_drawdown']:.2f}")

                # ── Curva de capital
                fig_cap = go.Figure(go.Scatter(
                    y=bt['capital_curve'],
                    mode='lines',
                    line=dict(color='green', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0,200,0,0.1)',
                    name='Capital'
                ))
                fig_cap.update_layout(
                    title='Curva de Capital — Backtest',
                    xaxis_title='Operações',
                    yaxis_title='Lucro ($)',
                    height=350
                )
                st.plotly_chart(fig_cap, use_container_width=True)

                # ── Gráfico de candles com sinais sobrepostos
                df_trades = bt['df_trades']
                df_test   = st.session_state['df_test']
                df_test_display = df_test[
                    df_test.index >= df_test.index.max() - pd.Timedelta(days=test_days)
                ]

                fig_signals = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            row_heights=[0.75, 0.25],
                                            vertical_spacing=0.02)

                fig_signals.add_trace(go.Candlestick(
                    x=df_test_display.index,
                    open=df_test_display['open'],
                    high=df_test_display['high'],
                    low=df_test_display['low'],
                    close=df_test_display['close'],
                    name='EUR/USD'
                ), row=1, col=1)

                calls = df_trades[df_trades['sinal'] == 'CALL']
                puts  = df_trades[df_trades['sinal'] == 'PUT']
                wins_df  = df_trades[df_trades['resultado'] == 'WIN']
                losses_df = df_trades[df_trades['resultado'] == 'LOSS']

                fig_signals.add_trace(go.Scatter(
                    x=calls['datetime'], y=calls['datetime'].map(
                        lambda d: df_test_display['low'].get(d, None)
                    ),
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='lime'),
                    name='CALL'
                ), row=1, col=1)

                fig_signals.add_trace(go.Scatter(
                    x=puts['datetime'], y=puts['datetime'].map(
                        lambda d: df_test_display['high'].get(d, None)
                    ),
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='PUT'
                ), row=1, col=1)

                # Distribuição WIN/LOSS
                fig_signals.add_trace(go.Bar(
                    x=df_trades['datetime'],
                    y=df_trades['lucro'],
                    marker_color=df_trades['resultado'].map(
                        {'WIN': 'green', 'LOSS': 'red'}
                    ),
                    name='P&L'
                ), row=2, col=1)

                fig_signals.update_layout(
                    title='Sinais de Operação — Backtest 5 dias',
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                st.plotly_chart(fig_signals, use_container_width=True)

                # ── Tabela de trades
                st.subheader("📋 Histórico de Operações")
                st.dataframe(bt['df_trades'], use_container_width=True)

    # ─────────────────────────────────────────────────────────
    # ETAPA 4 — Sinais para Amanhã
    # ─────────────────────────────────────────────────────────
    if 'best_model' in st.session_state and 'backtest' in st.session_state:
        st.header("🔮 Etapa 4 — Sinais para o Próximo Dia")

        if st.button("▶️ Gerar Sinais"):
            with st.spinner("Gerando sinais de alta probabilidade..."):
                df_full = build_multi_timeframe_dataset(
                    {
                        '5m':  tfs['5m'],
                        '15m': tfs['15m'],
                        '30m': tfs['30m']
                    },
                    threshold_bp
                )
                signals_df = generate_signals_tomorrow(
                    df_full,
                    st.session_state['best_model'],
                    st.session_state['feature_cols'],
                    prob_threshold=prob_threshold
                )

            st.success(f"✅ {len(signals_df)} sinais encontrados!")

            # Tabela de sinais
            st.dataframe(
                signals_df[['close', 'sinal', 'proba_call', 'proba_put']].rename(columns={
                    'close': 'Preço', 'sinal': 'Sinal',
                    'proba_call': 'Prob CALL (%)', 'proba_put': 'Prob PUT (%)'
                }).assign(**{
                    'Prob CALL (%)': lambda d: (d['Prob CALL (%)'] * 100).round(2),
                    'Prob PUT (%)':  lambda d: (d['Prob PUT (%)'] * 100).round(2),
                }),
                use_container_width=True
            )

            # Gráfico de sinais futuros
            fig_fut = go.Figure()
            fig_fut.add_trace(go.Candlestick(
                x=df_full.index[-200:],
                open=df_full['open'].iloc[-200:],
                high=df_full['high'].iloc[-200:],
                low=df_full['low'].iloc[-200:],
                close=df_full['close'].iloc[-200:],
                name='EUR/USD'
            ))
            calls_sig = signals_df[signals_df['sinal'] == 'CALL']
            puts_sig  = signals_df[signals_df['sinal'] == 'PUT']
            fig_fut.add_trace(go.Scatter(
                x=calls_sig.index, y=calls_sig['close'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=14, color='lime'),
                name='CALL'
            ))
            fig_fut.add_trace(go.Scatter(
                x=puts_sig.index, y=puts_sig['close'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=14, color='red'),
                name='PUT'
            ))
            fig_fut.update_layout(
                title='Sinais de Alta Probabilidade — Próximas Horas',
                xaxis_rangeslider_visible=False,
                height=450
            )
            st.plotly_chart(fig_fut, use_container_width=True)

else:
    st.info("👆 Faça o upload do CSV histórico do EURUSD M1 para começar.")