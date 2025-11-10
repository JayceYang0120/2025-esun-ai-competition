import os
import pandas as pd
import numpy as np
from tqdm import tqdm

FX_TWD = {
    'TWD': 1.0,
    'USD': 32.5, 'JPY': 0.22, 'AUD': 20.8, 'CNY': 4.5, 'EUR': 35.0,
    'SEK': 3.1,  'GBP': 41.0, 'HKD': 4.15, 'THB': 0.9, 'CAD': 24.0,
    'NZD': 19.0, 'CHF': 36.5, 'SGD': 24.2, 'ZAR': 1.8, 'MXN': 1.9
}

def load_csv(dir_path: str):
    """讀取三個檔案：交易、警示帳戶註記、待預測帳戶清單"""
    df_txn   = pd.read_csv(os.path.join(dir_path, "acct_transaction.csv"))
    df_alert = pd.read_csv(os.path.join(dir_path, "acct_alert.csv"))
    df_test  = pd.read_csv(os.path.join(dir_path, "acct_predict.csv"))
    print("(Finish) Load Dataset.")
    return df_txn, df_alert, df_test

def convert_to_twd(df_txn: pd.DataFrame,
                   amt_col: str = "txn_amt",
                   cur_col: str = "currency_type",
                   new_col: str = "txn_amt_twd"):
    """
    根據幣別轉換成新台幣金額。
    若幣別不在 FX_TWD 內，則匯率視為 1.0。
    """
    df = df_txn.copy()
    df[cur_col] = df[cur_col].astype(str)
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    # 匯率對應
    df["fx_rate"] = df[cur_col].map(FX_TWD).fillna(1.0)
    df[new_col] = df[amt_col] * df["fx_rate"]

    print(f"(Finish) Convert currency to TWD. Added column '{new_col}'.")
    return df


def split_txn_by_predict_list(df_txn: pd.DataFrame, df_test: pd.DataFrame):
    """
    依 df_test['acct'] 名單，將原始交易切成：
      - df_txn_test : 任一交易列的 from_acct 或 to_acct 在名單中
      - df_txn_train: 其餘交易列
    results: 
        Info: Original txn rows=4,435,890
        Info: There are 4,780 accounts in predict list.
        Info: There are 369,232 transactions related to predict accounts.
        (Finish) Txn Split. train_rows=4,066,658, test_rows=369,232
    """
    print(f"Info: Original txn rows={len(df_txn):,}")
    test_accts = set(df_test["acct"].dropna().astype(str))
    print(f"Info: There are {len(test_accts):,} accounts in predict list.")
    from_in = df_txn["from_acct"].astype(str).isin(test_accts)
    to_in   = df_txn["to_acct"].astype(str).isin(test_accts)
    mask_test = from_in | to_in
    print(f"Info: There are {mask_test.sum():,} transactions related to predict accounts.")

    df_txn_test  = df_txn[mask_test].copy()
    df_txn_train = df_txn[~mask_test].copy()

    print(f"(Finish) Txn Split. train_rows={len(df_txn_train):,}, test_rows={len(df_txn_test):,}")
    return df_txn_train, df_txn_test


def build_train_account_labels(df_txn_train: pd.DataFrame, df_alert: pd.DataFrame):
    """
    產出訓練帳戶清單與 label：
      - 蒐集 training 交易中出現過的帳戶（from_acct/to_acct）
      - label=1: 該帳戶在 df_alert['acct'] 出現過（有被標示為警示帳戶）
      - label=0: 否則
    results:
    (Finish) Build Train Labels. train_accts=1,677,829, positives=1,004
    """
    accts_train = pd.unique(
        pd.concat([
            df_txn_train["from_acct"].astype(str),
            df_txn_train["to_acct"].astype(str)
        ], ignore_index=True)
    )
    df_train_accts = pd.DataFrame({"acct": accts_train})

    alert_set = set(df_alert["acct"].dropna().astype(str))
    df_train_accts["label"] = df_train_accts["acct"].astype(str).isin(alert_set).astype(int)

    print(f"(Finish) Build Train Labels. train_accts={len(df_train_accts):,}, positives={df_train_accts['label'].sum():,}")
    return df_train_accts

def _to_long(df_txn: pd.DataFrame,
             acct_col_from="from_acct", acct_col_to="to_acct",
             date_col="txn_date", amt_col="txn_amt_twd",
             self_col="is_self_txn"):
    """把交易展成 (acct, counterparty, direction, date, txn_amt, is_self) 長格式。"""
    df = df_txn.copy()
    df[acct_col_from] = df[acct_col_from].astype(str)
    df[acct_col_to]   = df[acct_col_to].astype(str)
    df[date_col]      = pd.to_numeric(df[date_col], errors="coerce")
    df[amt_col]       = pd.to_numeric(df[amt_col], errors="coerce").fillna(0)

    # 正規化 is_self_txn → is_self ∈ {0,1}
    if self_col in df.columns:
        self_map_true = {"Y","y","1",1,"T","t","True","true"}
        self_series = df[self_col].astype(str).fillna("0")
        is_self = self_series.apply(lambda x: 1 if x in self_map_true else 0).astype(int)
    else:
        is_self = pd.Series(0, index=df.index, dtype=int)

    base_cols = [acct_col_from, acct_col_to, date_col, amt_col]
    df_from = df[base_cols].rename(
        columns={acct_col_from: "acct", acct_col_to: "counterparty", amt_col: "txn_amt"}
    )
    df_from["direction"] = "send"
    df_from["is_self"]   = is_self.values

    df_to = df[base_cols].rename(
        columns={acct_col_to: "acct", acct_col_from: "counterparty", amt_col: "txn_amt"}
    )
    df_to["direction"] = "recv"
    df_to["is_self"]   = is_self.values

    return pd.concat([df_from, df_to], ignore_index=True)

def _agg_one_window(acct, d, a, dirc, cp, self_flag, S, E, ed, horizon):
    """單一帳戶單一視窗的快速聚合（純 numpy），回傳 dict。"""
    # 分方向
    send_mask = (dirc == 1)
    recv_mask = ~send_mask

    send_amt = a[send_mask]
    recv_amt = a[recv_mask]

    send_cnt = int(send_mask.sum())
    recv_cnt = int(recv_mask.sum())
    total_cnt_long = int(len(a))

    # 自轉（long 會雙筆）→ 近似原始筆數修正
    self_cnt_long = int(self_flag.sum())
    self_txn_cnt  = int(self_cnt_long // 2)
    txn_cnt = max(1, total_cnt_long - self_txn_cnt)

    send_sum = float(send_amt.sum())
    recv_sum = float(recv_amt.sum())
    total_amt = send_sum + recv_sum
    denom_cnt = max(1, send_cnt + recv_cnt)
    denom_amt = total_amt if total_amt > 0 else 1.0

    # 不重複對手數
    uniq_cp = int(np.unique(cp).size)

    # 打標
    label = int((ed == ed) and (ed >= E) and (ed <= E + horizon))

    return {
        "acct": acct,
        "window_start": int(S),
        "window_end": int(E),

        "send_sum": send_sum,
        "send_cnt": send_cnt,
        "send_max": float(send_amt.max()) if send_cnt > 0 else 0.0,
        "send_min": float(send_amt.min()) if send_cnt > 0 else 0.0,
        "send_avg": float(send_amt.mean()) if send_cnt > 0 else 0.0,

        "recv_sum": recv_sum,
        "recv_cnt": recv_cnt,
        "recv_max": float(recv_amt.max()) if recv_cnt > 0 else 0.0,
        "recv_min": float(recv_amt.min()) if recv_cnt > 0 else 0.0,
        "recv_avg": float(recv_amt.mean()) if recv_cnt > 0 else 0.0,

        "total_cnt": total_cnt_long,
        "uniq_counterparties": uniq_cp,

        "send_cnt_ratio": float(send_cnt / denom_cnt),
        "recv_cnt_ratio": float(recv_cnt / denom_cnt),
        "send_share_amt": float(send_sum / denom_amt),
        "recv_share_amt": float(recv_sum / denom_amt),
        "uniq_counterparties_per_txn": float(uniq_cp / txn_cnt),
        "self_txn_cnt": int(self_txn_cnt),
        "self_txn_ratio": float(self_txn_cnt / txn_cnt),

        "label": label,
    }

def make_training_windows_fast(df_txn_train: pd.DataFrame,
                               df_alert: pd.DataFrame,
                               window_size: int = 30,
                               step_size: int = 30,
                               date_col: str = "txn_date",
                               amt_col: str = "txn_amt_twd"):
    """
    單機優化版：以陣列運算為主，每帳戶一個小迴圈，避免在迴圈中重複做昂貴的 DataFrame 篩選。
    """

    # 1) long + 預處理（一次做完）
    df_long = _to_long(df_txn_train, date_col=date_col, amt_col=amt_col, self_col="is_self_txn").copy()

    # map direction -> 0/1
    df_long["dir_code"] = (df_long["direction"].values == "send").astype(np.int8)  # 1=send,0=recv
    # counterparty 整數化，uniq 時用
    df_long["cp_code"] = pd.factorize(df_long["counterparty"])[0].astype(np.int32)
    # dtype 壓到數值型
    df_long[date_col] = pd.to_numeric(df_long[date_col], errors="coerce").astype(np.int32)
    df_long["txn_amt"] = pd.to_numeric(df_long["txn_amt"], errors="coerce").fillna(0.0).astype(np.float64)
    df_long["is_self"] = df_long["is_self"].astype(np.int8)

    # 2) alert 最早事件日 dict
    df_alert = df_alert.copy()
    df_alert["acct"] = df_alert["acct"].astype(str)
    df_alert["event_date"] = pd.to_numeric(df_alert["event_date"], errors="coerce")
    first_alert = (
        df_alert.dropna(subset=["event_date"])
                .sort_values(["acct", "event_date"])
                .drop_duplicates(subset=["acct"], keep="first")
                .set_index("acct")["event_date"]
                .to_dict()
    )
    horizon = 30 # 30 | 60

    # 3) 依帳戶分組，準備結果
    rows = []
    g = df_long.sort_values([ "acct", date_col ]).groupby("acct", sort=False)

    for acct, sub in tqdm(g, total=g.ngroups, desc="Building training windows (fast)"):
        # 取出陣列（避免反覆 .loc）
        d = sub[date_col].to_numpy()
        a = sub["txn_amt"].to_numpy()
        dirc = sub["dir_code"].to_numpy()   # 1=send, 0=recv
        cp = sub["cp_code"].to_numpy()
        self_flag = sub["is_self"].to_numpy()

        lo, hi = int(d.min()), int(d.max())
        E = hi

        ed = first_alert.get(acct, np.nan)

        # 用索引切片：找 [S,E] 的資料範圍
        # 我們簡化：每次用布林掩碼，但在「該帳戶的子表」上，成本已很小；或可用二分提升。
        while E >= lo:
            S = max(lo, E - window_size + 1)
            mask = (d >= S) & (d <= E)

            if not mask.any():
                E -= step_size
                if E < lo:
                    # 最末不足一窗的補窗
                    S2 = lo
                    E2 = min(hi, lo + window_size - 1)
                    mask2 = (d >= S2) & (d <= E2)
                    if mask2.any():
                        rows.append(_agg_one_window(acct, d[mask2], a[mask2], dirc[mask2], cp[mask2], self_flag[mask2],
                                                    S2, E2, ed, horizon))
                break

            rows.append(_agg_one_window(acct, d[mask], a[mask], dirc[mask], cp[mask], self_flag[mask],
                                        S, E, ed, horizon))

            E -= step_size
            if E < lo:
                # 最末不足一窗的補窗
                S2 = lo
                E2 = min(hi, lo + window_size - 1)
                if not (S2 == S and E2 == E + step_size):
                    mask2 = (d >= S2) & (d <= E2)
                    if mask2.any():
                        rows.append(_agg_one_window(acct, d[mask2], a[mask2], dirc[mask2], cp[mask2], self_flag[mask2],
                                                    S2, E2, ed, horizon))
                break

    df_samples = pd.DataFrame(rows).fillna(0)
    print(f"(Finish) Make Training Windows (fast). samples={len(df_samples):,}, positives={df_samples['label'].sum():,}")
    return df_samples

def make_testing_last_window_fast(df_txn_test: pd.DataFrame,
                                  df_predict_accts: pd.DataFrame,   # 👈 新增參數
                                  date_col: str = "txn_date",
                                  amt_col: str = "txn_amt_twd"):
    """
    針對每個「預測帳戶」建立最後 30 天視窗特徵（無 label）——快速版。
    只針對 acct_predict.csv 名單內帳戶計算。
    與訓練欄位同構：含金額統計、行為結構、關係密度與自轉比例。
    """
    # 1) 轉 long 並一次性預處理
    df_long = _to_long(df_txn_test, date_col=date_col, amt_col=amt_col, self_col="is_self_txn").copy()
    df_long["dir_code"] = (df_long["direction"].values == "send").astype(np.int8)   # 1=send,0=recv
    df_long["cp_code"]  = pd.factorize(df_long["counterparty"])[0].astype(np.int32)
    df_long[date_col]   = pd.to_numeric(df_long[date_col], errors="coerce").astype(np.int32)
    df_long["txn_amt"]  = pd.to_numeric(df_long["txn_amt"], errors="coerce").fillna(0.0).astype(np.float64)
    df_long["is_self"]  = df_long["is_self"].astype(np.int8)

    # 2) 僅保留 acct_predict.csv 中的帳戶
    target_accts = set(df_predict_accts["acct"].astype(str))
    before_rows = len(df_long)
    df_long = df_long[df_long["acct"].astype(str).isin(target_accts)]
    print(f"[Filter] Keep only predict list accounts: {len(df_long):,}/{before_rows:,} rows remain.")

    # 3) 依帳戶分組，逐帳戶用陣列做運算
    rows = []
    g = df_long.sort_values(["acct", date_col]).groupby("acct", sort=False)

    for acct, sub in tqdm(g, total=g.ngroups, desc="Building testing last windows (fast)"):
        d   = sub[date_col].to_numpy()
        a   = sub["txn_amt"].to_numpy()
        dirc= sub["dir_code"].to_numpy()
        cp  = sub["cp_code"].to_numpy()
        self_flag = sub["is_self"].to_numpy()

        hi = int(d.max())
        lo = max(1, hi - 29)   # 最後 30 天，不足則由 1 開始

        mask = (d >= lo) & (d <= hi)
        if not mask.any():
            continue

        d_w, a_w = d[mask], a[mask]
        dir_w, cp_w, self_w = dirc[mask], cp[mask], self_flag[mask]

        send_mask = (dir_w == 1)
        recv_mask = ~send_mask

        send_amt = a_w[send_mask]
        recv_amt = a_w[recv_mask]

        send_cnt = int(send_mask.sum())
        recv_cnt = int(recv_mask.sum())
        total_cnt_long = int(len(a_w))

        self_cnt_long = int(self_w.sum())
        self_txn_cnt  = int(self_cnt_long // 2)
        txn_cnt       = max(1, total_cnt_long - self_txn_cnt)

        send_sum = float(send_amt.sum())
        recv_sum = float(recv_amt.sum())
        total_amt = send_sum + recv_sum
        denom_cnt = max(1, send_cnt + recv_cnt)
        denom_amt = total_amt if total_amt > 0 else 1.0
        uniq_cp   = int(np.unique(cp_w).size)

        rows.append({
            "acct": acct,
            "window_start": int(lo),
            "window_end": int(hi),

            "send_sum": send_sum,
            "send_cnt": send_cnt,
            "send_max": float(send_amt.max()) if send_cnt > 0 else 0.0,
            "send_min": float(send_amt.min()) if send_cnt > 0 else 0.0,
            "send_avg": float(send_amt.mean()) if send_cnt > 0 else 0.0,

            "recv_sum": recv_sum,
            "recv_cnt": recv_cnt,
            "recv_max": float(recv_amt.max()) if recv_cnt > 0 else 0.0,
            "recv_min": float(recv_amt.min()) if recv_cnt > 0 else 0.0,
            "recv_avg": float(recv_amt.mean()) if recv_cnt > 0 else 0.0,

            "total_cnt": total_cnt_long,
            "uniq_counterparties": uniq_cp,

            "send_cnt_ratio": float(send_cnt / denom_cnt),
            "recv_cnt_ratio": float(recv_cnt / denom_cnt),
            "send_share_amt": float(send_sum / denom_amt),
            "recv_share_amt": float(recv_sum / denom_amt),
            "uniq_counterparties_per_txn": float(uniq_cp / txn_cnt),
            "self_txn_cnt": int(self_txn_cnt),
            "self_txn_ratio": float(self_txn_cnt / txn_cnt),
        })

    df_test_feats = pd.DataFrame(rows).fillna(0)
    print(f"(Finish) Make Testing Last Window (fast). accounts={df_test_feats['acct'].nunique():,}")
    return df_test_feats

def save_outputs(out_dir: str,
                 df_txn_train: pd.DataFrame,
                 df_txn_test: pd.DataFrame,
                 df_train_accts: pd.DataFrame,
                 df_train_windows: pd.DataFrame,
                 df_test_lastwin: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    path_train_txn = os.path.join(out_dir, "txn_train.csv")
    path_test_txn  = os.path.join(out_dir, "txn_test.csv")
    path_train_lbl = os.path.join(out_dir, "train_accounts_with_label.csv")
    path_train_win = os.path.join(out_dir, "train_window_samples.csv")
    path_test_feat = os.path.join(out_dir, "test_last_window_features.csv")

    df_txn_train.to_csv(path_train_txn, index=False)
    df_txn_test.to_csv(path_test_txn, index=False)
    df_train_accts.to_csv(path_train_lbl, index=False)
    df_train_windows.to_csv(path_train_win, index=False)
    df_test_lastwin.to_csv(path_test_feat, index=False)

    print("(Finish) Output saved:\n"
          f"  - {path_train_txn}\n"
          f"  - {path_test_txn}\n"
          f"  - {path_train_lbl}\n"
          f"  - {path_train_win}\n"
          f"  - {path_test_feat}")


if __name__ == "__main__":
    # === parameters ===
    dir_dataset = "./preliminary_data/"
    dir_output  = "./assets/preprocess/"
    WINDOW_SIZE = 30
    STEP_SIZE   = 30

    # === pipeline ===
    df_txn, df_alert, df_test = load_csv(dir_dataset)

    df_txn_train, df_txn_test = split_txn_by_predict_list(df_txn, df_test)

    df_txn_train = convert_to_twd(
        df_txn_train, amt_col="txn_amt", cur_col="currency_type", new_col="txn_amt_twd"
    )
    df_txn_test = convert_to_twd(
        df_txn_test, amt_col="txn_amt", cur_col="currency_type", new_col="txn_amt_twd"
    )

    df_train_accts = build_train_account_labels(df_txn_train, df_alert)

    # 造訓練視窗（多樣本/帳戶），並依 (E, E+30] 打標 — 使用換算後金額欄位 txn_amt_twd
    df_train_windows = make_training_windows_fast(
        df_txn_train = df_txn_train,
        df_alert = df_alert,
        window_size = WINDOW_SIZE,
        step_size = STEP_SIZE,
        date_col = "txn_date",
        amt_col = "txn_amt_twd",
    )

    # 造測試帳戶最後一個 30 天視窗特徵（無 label）— 也用 txn_amt_twd
    df_test_lastwin = make_testing_last_window_fast(
        df_txn_test = df_txn_test,
        df_predict_accts = df_test,
        date_col = "txn_date",
        amt_col = "txn_amt_twd",
    )

    # === 在訓練視窗加上「變化」特徵 ===
    # 先依帳戶與視窗結束時間排序，避免 diff 被打亂
    df_train_windows = df_train_windows.sort_values(["acct", "window_end"]).reset_index(drop=True)

    # 三個變化特徵（上一個視窗到當前視窗的差值）
    df_train_windows["send_sum_diff"] = df_train_windows.groupby("acct")["send_sum"].diff()
    df_train_windows["recv_sum_diff"] = df_train_windows.groupby("acct")["recv_sum"].diff()
    df_train_windows["uniq_cp_diff"]  = df_train_windows.groupby("acct")["uniq_counterparties"].diff()

    # 第一個視窗的 diff 會是 NaN，統一補 0（也較容易給模型解讀為「無變化/起點」）
    df_train_windows[["send_sum_diff", "recv_sum_diff", "uniq_cp_diff"]] = \
        df_train_windows[["send_sum_diff", "recv_sum_diff", "uniq_cp_diff"]].fillna(0.0)

    # （可選）壓 dtype，避免佔記憶體
    for c in ["send_sum_diff", "recv_sum_diff", "uniq_cp_diff"]:
        df_train_windows[c] = df_train_windows[c].astype(np.float32)

    # === 測試集補上同名欄位（最後一窗無前一窗可比，設 0） ===
    for c in ["send_sum_diff", "recv_sum_diff", "uniq_cp_diff"]:
        if c not in df_test_lastwin.columns:
            df_test_lastwin[c] = 0.0
        df_test_lastwin[c] = df_test_lastwin[c].astype(np.float32)

    save_outputs(
        dir_output,
        df_txn_train,
        df_txn_test,
        df_train_accts,
        df_train_windows,
        df_test_lastwin
    )

