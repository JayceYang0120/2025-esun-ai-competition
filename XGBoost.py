# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_curve
from xgboost import XGBClassifier

def select_threshold_by_objective(y_true, proba,
                                  objective="max_f1",
                                  beta=1.0,
                                  min_precision=None,
                                  alert_rate=None,
                                  top_k=None):
    """
    根據不同目標回傳最佳 threshold 與對應指標。
    支援：
      - objective="max_f1": 使 F1 最大
      - objective="max_fbeta": 使 Fβ 最大（β>1 偏 recall，β<1 偏 precision）
      - min_precision: 在 precision >= 此下限下，取 F1 最大（實務常用）
      - alert_rate: 固定預警比例（例如 0.005 -> 0.5% 帳戶會被標 1）
      - top_k: 固定預警數量（例如每天只能人工審 500 件）

    備註：
      - 若同時提供 alert_rate / top_k，優先使用 top_k。
      - y_true/proba 來自 validation set（且以帳戶為 group 的切分）。
    """
    proba = np.asarray(proba)
    y_true = np.asarray(y_true).astype(int)

    # 固定量/比例類目標：直接從分位數或排序取門檻，不用 PR 曲線
    n = len(proba)
    if top_k is not None:
        top_k = max(1, min(top_k, n))
        thr = np.partition(proba, -top_k)[-top_k]  # 第 top_k 名的分數
        y_pred = (proba >= thr).astype(int)
        prec = (y_true[y_pred==1].mean() if y_pred.sum()>0 else 0.0)
        rec  = (y_pred[y_true==1].mean() if y_true.sum()>0 else 0.0)
        f1   = f1_score(y_true, y_pred) if (prec+rec)>0 else 0.0
        return dict(threshold=float(thr), precision=float(prec), recall=float(rec), f1=float(f1),
                    objective="top_k", selected_k=int(y_pred.sum()))

    if alert_rate is not None:
        k = int(round(alert_rate * n))
        k = max(1, min(k, n))
        thr = np.partition(proba, -k)[-k]
        y_pred = (proba >= thr).astype(int)
        prec = (y_true[y_pred==1].mean() if y_pred.sum()>0 else 0.0)
        rec  = (y_pred[y_true==1].mean() if y_true.sum()>0 else 0.0)
        f1   = f1_score(y_true, y_pred) if (prec+rec)>0 else 0.0
        return dict(threshold=float(thr), precision=float(prec), recall=float(rec), f1=float(f1),
                    objective=f"alert_rate={alert_rate}", selected_k=int(y_pred.sum()))

    # 其餘：走 PR 曲線掃描
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve 會回傳 len(thresholds)=len(precision)-1
    precision, recall = precision[:-1], recall[:-1]

    if min_precision is not None:
        mask = precision >= min_precision
        if not np.any(mask):
            # 若門檻太嚴格，退而求其次改用最高 F1
            mask = slice(None)
        precision, recall, thresholds = precision[mask], recall[mask], thresholds[mask]

    if objective == "max_f1":
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = int(np.argmax(f1))
    elif objective == "max_fbeta":
        beta2 = beta * beta
        fbeta = (1+beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
        idx = int(np.argmax(fbeta))
    else:
        raise ValueError("objective must be 'max_f1' or 'max_fbeta'")

    thr = float(thresholds[idx])
    y_pred = (proba >= thr).astype(int)
    return dict(
        threshold=thr,
        precision=float(precision[idx]),
        recall=float(recall[idx]),
        f1=float(f1_score(y_true, y_pred)),
        objective=objective if min_precision is None else f"{objective}_with_min_precision={min_precision}"
    )

# ========= 路徑設定 =========
DATA_DIR = "./assets/preprocess/"
PATH_TRAIN_WIN = os.path.join(DATA_DIR, "train_window_samples.csv")
PATH_TEST_LAST = os.path.join(DATA_DIR, "test_last_window_features.csv")
OUT_DIR = "./assets/result/"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 讀取資料 =========
print("(1/5) Loading data ...")
df_train = pd.read_csv(PATH_TRAIN_WIN)
df_test  = pd.read_csv(PATH_TEST_LAST)

print(df_train["label"].value_counts(normalize=True))
print("Max txn_date:", df_train["window_end"].max())

# ========= 特徵欄位 =========
drop_cols = ["acct", "window_start", "window_end", "label"]
feature_cols = [c for c in df_train.columns if c not in drop_cols]

X = df_train[feature_cols].fillna(0.0).astype(np.float32)
y = df_train["label"].astype(int).values
groups = df_train["acct"].astype(str).values

# ========= Group-based Train/Valid Split =========
print("(2/5) Splitting train/validation by account ...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=309)
train_idx, valid_idx = next(gss.split(X, y, groups=groups))
X_tr, y_tr = X.iloc[train_idx], y[train_idx]
X_va, y_va = X.iloc[valid_idx], y[valid_idx]

# ========= 處理不平衡 =========
pos = y_tr.sum()
neg = len(y_tr) - pos
scale_pos_weight = float(neg / max(1, pos)) if pos > 0 else 1.0
print(f"[Info] Train positives={pos}, negatives={neg}, scale_pos_weight={scale_pos_weight:.2f}")

# ========= 訓練 XGBoost =========
print("(3/5) Training XGBoost model ...")
xgb_params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="binary:logistic",  # ✅ 機率輸出
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",          # ✅ 不平衡資料表現穩定
    tree_method="hist",           # 若有 GPU 可改 "gpu_hist"
    random_state=309
)

model = XGBClassifier(**xgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=100,
)

# ========= 驗證：以 F1-score 選擇最佳閾值 =========
print("(4/5) Searching best threshold by F1 ...")
proba_va = model.predict_proba(X_va)[:, 1]

## method 1: probability threshold sweep
thresholds = np.linspace(0.05, 0.7, 19)
f1_list = []
for th in tqdm(thresholds, desc="Evaluating thresholds"):
    f1 = f1_score(y_va, (proba_va >= th).astype(int))
    f1_list.append((th, f1))
best_th, best_f1 = max(f1_list, key=lambda x: x[1])

## method 2: use function
# res_f1 = select_threshold_by_objective(y_va, proba_va, objective="max_f1")
# best_th = res_f1["threshold"]
# best_f1 = res_f1["f1"]

print(f"\n========== Validation Result ==========")
print(f"Best F1 = {best_f1:.4f} @ threshold = {best_th:.2f}")
y_pred_best = (proba_va >= best_th).astype(int)
print(confusion_matrix(y_va, y_pred_best))
print(classification_report(y_va, y_pred_best, digits=4))

# ========= 對測試集產生預測結果 =========
print("(5/5) Predicting test set ...")
X_test = df_test[feature_cols].fillna(0.0).astype(np.float32)

# tqdm 包住 predict_proba 迴圈（分批處理防止記憶體爆炸）
batch_size = 50000
proba_list = []
for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting batches"):
    batch = X_test.iloc[i:i+batch_size]
    proba_batch = model.predict_proba(batch)[:, 1]
    proba_list.append(proba_batch)
proba_te = np.concatenate(proba_list)

y_te = (proba_te >= best_th).astype(int)

df_submit = pd.DataFrame({
    "acct": df_test["acct"].astype(str).values,
    "label": y_te
})

submit_path = os.path.join(OUT_DIR, "result_xgb.csv")
df_submit.to_csv(submit_path, index=False)
print(f"\n✅ (Saved) result_xgb.csv -> {submit_path}")