import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS = {
    'XGBoost': XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='logloss',
        n_jobs=-1, verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='binary', verbose=-1, n_jobs=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=7,
        min_samples_leaf=50, n_jobs=-1
    )
}


def train_and_evaluate(df_train: pd.DataFrame, feature_cols: list,
                       prob_threshold=0.60, n_splits=5) -> dict:
    """
    Treina todos os modelos com TimeSeriesSplit e retorna métricas comparativas.
    """
    X = df_train[feature_cols]
    y = df_train['target']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for name, model in MODELS.items():
        fold_metrics = []

        for fold, (ti, vi) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[ti], X.iloc[vi]
            y_tr, y_val = y.iloc[ti], y.iloc[vi]

            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]

            # Aplicar threshold de confiança
            pred = np.full(len(y_val), -1)
            pred[proba >= prob_threshold]       = 1
            pred[proba <= (1 - prob_threshold)] = 0
            mask = pred != -1

            if mask.sum() < 10:
                continue

            y_v = y_val.values[mask]
            p_v = pred[mask]
            pr_v = proba[mask]

            acc  = (y_v == p_v).mean()
            mcc  = matthews_corrcoef(y_v, p_v)
            try:
                auc = roc_auc_score(y_v, pr_v)
            except Exception:
                auc = None

            report = classification_report(y_v, p_v, output_dict=True)

            fold_metrics.append({
                'fold':       fold,
                'trades':     mask.sum(),
                'acc':        acc,
                'mcc':        mcc,
                'auc':        auc,
                'precision_1': report.get('1', {}).get('precision', None),
                'recall_1':    report.get('1', {}).get('recall', None),
                'f1_1':        report.get('1', {}).get('f1-score', None),
            })

        results[name] = {
            'model':        model,
            'fold_metrics': fold_metrics,
            'avg_acc':      np.mean([m['acc'] for m in fold_metrics]),
            'avg_mcc':      np.mean([m['mcc'] for m in fold_metrics]),
            'avg_auc':      np.mean([m['auc'] for m in fold_metrics if m['auc']]),
        }

    return results


def tune_best_model(df_train: pd.DataFrame, feature_cols: list,
                    model_name='XGBoost', n_trials=30) -> object:
    """
    Usa Optuna para tunar hiperparâmetros do modelo escolhido.
    """
    X = df_train[feature_cols]
    y = df_train['target']
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        if model_name == 'XGBoost':
            params = {
                'n_estimators':    trial.suggest_int('n_estimators', 200, 600),
                'max_depth':       trial.suggest_int('max_depth', 3, 7),
                'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0, 'n_jobs': -1
            }
            model = XGBClassifier(**params)

        elif model_name == 'LightGBM':
            params = {
                'n_estimators':  trial.suggest_int('n_estimators', 200, 600),
                'max_depth':     trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
                'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary', 'verbose': -1, 'n_jobs': -1
            }
            model = LGBMClassifier(**params)

        scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    if model_name == 'XGBoost':
        best_model = XGBClassifier(**best_params,
                                   objective='binary:logistic',
                                   eval_metric='logloss',
                                   verbosity=0, n_jobs=-1)
    else:
        best_model = LGBMClassifier(**best_params,
                                    objective='binary',
                                    verbose=-1, n_jobs=-1)

    best_model.fit(X, y)
    joblib.dump(best_model, f'models/best_{model_name}.pkl')

    return best_model, best_params, study


def load_model(path: str):
    return joblib.load(path)