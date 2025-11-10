"""
2025玉山人工智慧挑戰賽範例程式碼
"""
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def LoadCSV(dir_path):
    """
    讀取挑戰賽提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    Args:
        dir_path (str): 資料夾，請把上述3個檔案放在同一個資料夾
    
    Returns:
        df_txn: 交易資料 DataFrame
        df_alert: 警示帳戶註記 DataFrame
        df_test: 待預測帳戶清單 DataFrame
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    
    print("(Finish) Load Dataset.")
    return df_txn, df_alert, df_test


def PreProcessing(df):
    """
    資料處理的範例程式，計算每個帳戶的一些統計量，當作模型因子
    參賽者可自行發想、設計自己的因子
    """    
    # 1. 'total_send/recv_amt': total amount sent/received by each acct
    send = df.groupby('from_acct')['txn_amt'].sum().rename('total_send_amt')
    recv = df.groupby('to_acct')['txn_amt'].sum().rename('total_recv_amt')

    # 2. max, min, avg txn_amt for each account
    max_send = df.groupby('from_acct')['txn_amt'].max().rename('max_send_amt')
    min_send = df.groupby('from_acct')['txn_amt'].min().rename('min_send_amt')
    avg_send = df.groupby('from_acct')['txn_amt'].mean().rename('avg_send_amt')
    
    max_recv = df.groupby('to_acct')['txn_amt'].max().rename('max_recv_amt')
    min_recv = df.groupby('to_acct')['txn_amt'].min().rename('min_recv_amt')
    avg_recv = df.groupby('to_acct')['txn_amt'].mean().rename('avg_recv_amt')

    df_result = pd.concat([max_send, min_send, avg_send, max_recv, min_recv, avg_recv, send, recv], axis=1).fillna(0).reset_index()
    df_result.rename(columns={'index': 'acct'}, inplace=True)
    
    # 2. 'is_esun': is esun account or not
    df_from = df[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'})
    df_to = df[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'})
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # 4. merge (1), (2), and (3)
    df_result = pd.merge(df_result, df_acc, on='acct', how='left')    
    print("(Finish) PreProcessing.")
    return df_result

def TrainTestSplit(df, df_alert, df_test):
    """
    切分訓練集及測試集，並為訓練集的帳戶標上警示label (0為非警示、1為警示)
    
    備註:
        1. 測試集為待預測帳戶清單，你需要預測它們
        2. 此切分僅為範例，較標準的做法是基於訓練集再且分成train和validation，請有興趣的參賽者自行切分
        3. 由於待預測帳戶清單僅為玉山戶，所以我們在此範例僅使用玉山帳戶做訓練
    """  
    X_train = df[(~df['acct'].isin(df_test['acct'])) & (df['is_esun']==1)].drop(columns=['is_esun']).copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)
    X_test = df[df['acct'].isin(df_test['acct'])].drop(columns=['is_esun']).copy()
    
    print(f"(Finish) Train-Test-Split")
    return X_train, X_test, y_train

def Modeling(X_train, y_train, X_test):
    """
    Decision Tree的範例程式，參賽者可以在這裡實作自己需要的方法
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train.drop(columns=['acct']), y_train)
    y_pred = model.predict(X_test.drop(columns=['acct']))   
    
    print(f"(Finish) Modeling")
    return y_pred

def OutputCSV(path, df_test, X_test, y_pred):
    """
    根據測試資料集及預測結果，產出預測結果之CSV，該CSV可直接上傳於TBrain    
    """
    df_pred = pd.DataFrame({
        'acct': X_test['acct'].values,
        'label': y_pred
    })
    
    df_out = df_test[['acct']].merge(df_pred, on='acct', how='left')
    df_out.to_csv(path, index=False)    
    
    print(f"(Finish) Output saved to {path}")

if __name__ == "__main__":
    dir_path = "./preliminary_data/"
    df_txn, df_alert, df_test = LoadCSV(dir_path)
    df_X = PreProcessing(df_txn)
    X_train, X_test, y_train = TrainTestSplit(df_X, df_alert, df_test)
    y_pred = Modeling(X_train, y_train, X_test)
    out_path = "result.csv"
    OutputCSV(out_path, df_test, X_test, y_pred)