"""
Implementasaun Manual Random Forest — BuirenggoBoxFit
Kalkulasaun Gini Impurity no Decision Tree LAHO scikit-learn
DIT Machine Learning 2026
"""
import numpy as np
import pandas as pd


# ── Gini Impurity ─────────────────────────────────────────────────────────────
def gini_impurity(labels):
    """Kalkulasaun Gini Impurity: G = 1 - Σ(pi²)"""
    labels = list(labels)
    n = len(labels)
    if n == 0:
        return 0.0
    classes = set(labels)
    return 1.0 - sum((labels.count(c) / n) ** 2 for c in classes)


def gini_weighted(y_left, y_right):
    """Gini Impurity pezadu depois split"""
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) / n) * gini_impurity(y_left) + \
           (len(y_right) / n) * gini_impurity(y_right)


def information_gain(y_parent, y_left, y_right):
    return gini_impurity(y_parent) - gini_weighted(y_left, y_right)


# ── Find best split ───────────────────────────────────────────────────────────
def find_best_split(X_df, y, feature, thresholds=None):
    """Buka split di'ak liu ba feature ida"""
    vals = sorted(X_df[feature].unique())
    if thresholds is None:
        thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals)-1)]

    best = {'threshold': None, 'gini': 1.0, 'ig': 0.0,
            'left_idx': [], 'right_idx': []}
    for thr in thresholds:
        mask  = X_df[feature] >= thr
        left  = list(y[~mask])
        right = list(y[mask])
        if not left or not right:
            continue
        g = gini_weighted(left, right)
        ig = gini_impurity(list(y)) - g
        if g < best['gini']:
            best = {'threshold': thr, 'gini': round(g, 4),
                    'ig': round(ig, 4),
                    'left_idx':  list(X_df[~mask].index),
                    'right_idx': list(X_df[mask].index)}
    return best


# ── Bootstrap sampling ────────────────────────────────────────────────────────
def bootstrap_sample(X_df, y, n_samples=None, random_state=None):
    rng = np.random.RandomState(random_state)
    n = n_samples or len(X_df)
    idx = rng.choice(len(X_df), size=n, replace=True)
    oob = [i for i in range(len(X_df)) if i not in idx]
    return X_df.iloc[idx].reset_index(drop=True), \
           y.iloc[idx].reset_index(drop=True), oob


# ── Simple manual decision tree (depth 1 = stump) ────────────────────────────
class ManualDecisionStump:
    """Decision tree sedira (depth 1) — manual tanpa scikit-learn"""
    def __init__(self):
        self.feature    = None
        self.threshold  = None
        self.left_class = None
        self.right_class = None
        self.gini       = None

    def fit(self, X_df, y):
        best_ig = -1
        for feat in X_df.columns:
            sp = find_best_split(X_df, y, feat)
            if sp['threshold'] is not None and sp['ig'] > best_ig:
                best_ig = sp['ig']
                self.feature   = feat
                self.threshold = sp['threshold']
                left_y  = y.iloc[[i for i in range(len(X_df))
                                   if X_df[feat].iloc[i] < self.threshold]]
                right_y = y.iloc[[i for i in range(len(X_df))
                                   if X_df[feat].iloc[i] >= self.threshold]]
                self.left_class  = left_y.mode()[0]  if len(left_y)  else y.mode()[0]
                self.right_class = right_y.mode()[0] if len(right_y) else y.mode()[0]
                self.gini = sp['gini']

    def predict_one(self, row):
        if self.feature is None:
            return None
        if row[self.feature] >= self.threshold:
            return self.right_class
        return self.left_class


# ── Manual Random Forest (ensemble of stumps) ─────────────────────────────────
class ManualRandomForest:
    def __init__(self, n_trees=5, random_state=42):
        self.n_trees = n_trees
        self.trees   = []
        self.random_state = random_state

    def fit(self, X_df, y):
        self.trees = []
        for i in range(self.n_trees):
            X_b, y_b, _ = bootstrap_sample(X_df, y, random_state=self.random_state + i)
            # Pilih random features (sqrt)
            n_feat = max(1, int(np.sqrt(len(X_df.columns))))
            rng    = np.random.RandomState(self.random_state + i)
            feats  = list(rng.choice(X_df.columns, size=n_feat, replace=False))
            stump  = ManualDecisionStump()
            stump.fit(X_b[feats], y_b)
            self.trees.append(stump)

    def predict_one(self, row):
        votes = [t.predict_one(row) for t in self.trees if t.feature is not None]
        if not votes:
            return None
        return max(set(votes), key=votes.count)

    def predict(self, X_df):
        return [self.predict_one(X_df.iloc[i]) for i in range(len(X_df))]


# ── Step-by-step trace for display ───────────────────────────────────────────
SAMPLE_DATA = pd.DataFrame({
    'ID':      ['ATL-S1', 'ATL-S2', 'ATL-S3', 'ATL-S4', 'ATL-S5', 'ATL-S6'],
    'VO2 Max': [62.0, 58.5, 25.0, 55.0, 28.5, 30.0],
    'Push-up': [65,   60,   15,   58,   18,   20],
    'Klase':   ['Ekselente', 'Ekselente', 'Fraku',
                'Ekselente', 'Fraku',     'Fraku'],
})


def step1_root_gini(labels):
    n       = len(labels)
    classes = {}
    for c in labels:
        classes[c] = classes.get(c, 0) + 1
    rows = []
    gini = 1.0
    for cls, cnt in classes.items():
        p = cnt / n
        rows.append({'Klase': cls, 'Kontas': cnt, 'pi': round(p, 4), 'pi²': round(p**2, 4)})
        gini -= p**2
    return round(gini, 4), rows


def step2_split_analysis(feature_vals, labels, threshold):
    left_idx  = [i for i, v in enumerate(feature_vals) if v < threshold]
    right_idx = [i for i, v in enumerate(feature_vals) if v >= threshold]
    y_left    = [labels[i] for i in left_idx]
    y_right   = [labels[i] for i in right_idx]

    g_left  = round(gini_impurity(y_left),  4)
    g_right = round(gini_impurity(y_right), 4)
    n       = len(labels)
    g_w     = round((len(y_left)/n)*g_left + (len(y_right)/n)*g_right, 4)
    ig      = round(gini_impurity(labels) - g_w, 4)

    return {
        'left_idx':  left_idx,  'y_left':  y_left,  'g_left':  g_left,
        'right_idx': right_idx, 'y_right': y_right, 'g_right': g_right,
        'gini_weighted': g_w,   'info_gain': ig,
    }
