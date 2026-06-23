import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score, f1_score)
import pickle
import os

FEATURE_COLS = [
    'vo2max', 'pushup', 'situp', 'shuttle_run', 'sprint_30m',
    'sit_reach', 'body_fat', 'bmi', 'grip_strength', 'vertical_jump'
]
TARGET_COL   = 'prontidaun_fiziku'
CLASS_ORDER  = ['Fraku', 'Moderadu', 'Diak', 'Ekselente']

FEATURE_LABELS = {
    'vo2max':        'VO2 Max (ml/kg/min)',
    'pushup':        'Push-up (reps/min)',
    'situp':         'Sit-up (reps/min)',
    'shuttle_run':   'Shuttle Run (segundus)',
    'sprint_30m':    'Sprint 30m (segundus)',
    'sit_reach':     'Sit & Reach (cm)',
    'body_fat':      'Body Fat (%)',
    'bmi':           'BMI',
    'grip_strength': 'Grip Strength (kg)',
    'vertical_jump': 'Vertical Jump (cm)',
}

CLASS_COLORS = {
    'Ekselente': '#2ecc71',
    'Diak':      '#3498db',
    'Moderadu':  '#f39c12',
    'Fraku':     '#e74c3c',
}

CLASS_DESC = {
    'Ekselente': 'Atleta iha kondisaun fiziku diak tebes. Prontu ba kompetisaun iha nivel aas.',
    'Diak':      'Atleta iha kondisaun fiziku diak. Presiza mellora area kiik balun.',
    'Moderadu':  'Atleta iha kondisaun fiziku mediu. Presiza treinamentu intensivu iha area barak.',
    'Fraku':     'Atleta iha kondisaun fiziku fraku. Labele kompete too depois mellora signifikantivu.',
}


def load_data_mysql():
    """Karga dadus husi MySQL database — buirenggoboxfit.dataset_boxing"""
    try:
        import mysql.connector
        from db_config import MYSQL_CONFIG, MYSQL_TABLE
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        query = f"SELECT id_atleta, {', '.join(FEATURE_COLS)}, {TARGET_COL} FROM {MYSQL_TABLE}"
        df = pd.read_sql(query, conn)
        conn.close()
        return df, "mysql"
    except Exception as e:
        return None, str(e)


def load_data(filepath: str):
    """
    Karga dadus — prioridade MySQL, fallback ba CSV.
    1. Tenta koneksaun MySQL (buirenggoboxfit.dataset_boxing)
    2. Se MySQL la disponivel, uza CSV
    """
    df_mysql, status = load_data_mysql()
    if df_mysql is not None and len(df_mysql) > 0:
        df = df_mysql
        df.attrs['source'] = 'MySQL'
    else:
        df = pd.read_csv(filepath)
        df.attrs['source'] = 'CSV'

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y, df


def train_model(X, y):
    le = LabelEncoder()
    le.fit(CLASS_ORDER)
    y_enc = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        oob_score=True,
    )
    rf.fit(X_train, y_train)

    y_pred    = rf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc = cross_val_score(rf, X, y_enc, cv=cv, scoring='accuracy')

    metrics = {
        'accuracy':              accuracy,
        'precision':             precision,
        'recall':                recall,
        'f1':                    f1,
        'oob_score':             rf.oob_score_,
        'cv_mean':               cv_sc.mean(),
        'cv_std':                cv_sc.std(),
        'classification_report': classification_report(
                                     y_test, y_pred,
                                     target_names=[le.inverse_transform([i])[0]
                                                   for i in range(len(le.classes_))],
                                     zero_division=0),
        'confusion_matrix':      confusion_matrix(y_test, y_pred),
        'feature_importance':    dict(zip(FEATURE_COLS, rf.feature_importances_)),
        'X_test':  X_test,
        'y_test':  y_test,
        'y_pred':  y_pred,
        'classes': le.classes_,
    }

    return rf, le, metrics, X_train, X_test, y_train, y_test


def predict(model, le, input_data: dict):
    input_df   = pd.DataFrame([input_data])[FEATURE_COLS]
    pred       = model.predict(input_df)
    proba      = model.predict_proba(input_df)[0]
    label      = le.inverse_transform(pred)[0]
    confidence = proba.max() * 100
    prob_dict  = {le.inverse_transform([i])[0]: round(p * 100, 2) for i, p in enumerate(proba)}
    return label, confidence, prob_dict


def save_model(model, le, path='model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'le': le}, f)


def load_model(path='model.pkl'):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['model'], obj['le']
