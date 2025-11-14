import joblib
import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 靜音警告，因帶入的dataframe feature跟原生ColumnTransformer裡面的features_list不一致
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.columns if isinstance(X, pd.DataFrame) else [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        return cyclical_encode(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "columns_", [])
        df_temp = pd.DataFrame({c: [0] for c in input_features})
        df_encoded = cyclical_encode(df_temp)
        return df_encoded.columns.to_numpy()

def cyclical_encode(df):
    df = df.copy()
    #out_cols = []
    for c in df.columns:
        if c.endswith("_hour"):
            df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 24)
            df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 24)
            #out_cols += [f"{c}_sin", f"{c}_cos"]
        elif c.endswith("_dow"):
            df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 7)
            df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 7)
            #out_cols += [f"{c}_sin", f"{c}_cos"]
        elif c.endswith("_month"):
            df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 12)
            df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 12)
            #out_cols += [f"{c}_sin", f"{c}_cos"]
    return df


def parse_dates(df: pd.DataFrame, date_cols):
    out = df.copy()
    for c in date_cols:
        try:
            out[c] = pd.to_datetime(out[c], errors="coerce")
        except Exception:
            out[c] = pd.NaT
    return out


def add_date_features(df: pd.DataFrame, date_col: str):
    out = df.copy()
    col = date_col
    out[f"{col}_year"] = out[col].dt.year
    out[f"{col}_month"] = out[col].dt.month
    out[f"{col}_day"] = out[col].dt.day
    out[f"{col}_dow"] = out[col].dt.dayofweek
    out[f"{col}_hour"] = out[col].dt.hour
    return out

# 載入最佳模型
best_model = joblib.load("ml/models/final_rf_without_part_num_divide_v2.pkl")


preprocessor = best_model.named_steps["preprocess"]
feature_names = preprocessor.get_feature_names_out()
print(feature_names)
print("特徵數量:", len(feature_names))

# print(preprocessor.transformers)

# feature_names = preprocessor.get_feature_names_out()
# print("✅ 模型實際吃到的特徵數:", len(feature_names))
# print("🔹 前 10 個特徵名稱:", feature_names.tolist())


features = ['金厚下限','板子類型','歸屬班別','金厚上限','線別','槽次1','鎳','數量','檢查型態','鎳厚下限','MTO1','鎳厚上限','電流值1','電流值2','MTO2','項目']
label_col = "金"


# 準備新的資料（x_test）
# dirname 上一層
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw", "化金線自主檢查表_all.csv")
x_test = pd.read_csv(data_path, encoding="big5")


used_time_col = None

used_time_col = "生產日期"

if used_time_col:
    x_test = parse_dates(x_test, [used_time_col])
    x_test = add_date_features(x_test, used_time_col)
    #time_features = []
    time_features = [
        f"{used_time_col}_year",
        f"{used_time_col}_month",
        f"{used_time_col}_day",
        f"{used_time_col}_dow", # day of week
        f"{used_time_col}_hour"
    ]
    features.extend(time_features)
else:
    time_features = []


# 直接預測 — 不需要再手動做 ColumnTransformer！
opt_x_test = x_test.iloc[[-1]]

y_pred = best_model.predict(opt_x_test)

print("預測結果：", y_pred)

y_test = opt_x_test[label_col]
print("真實標籤：", y_test)

# === Evaluate
# mae  = mean_absolute_error(y_test, y_pred)
# mse  = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2   = r2_score(y_test, y_pred)
# print(f" MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")



# === 特徵重要度
# importances = best_model.named_steps["model"].feature_importances_
# features = best_model.named_steps["preprocess"].get_feature_names_out()

# feat_imp = pd.DataFrame({
#     "feature": features,
#     "importance": importances
# }).sort_values("importance", ascending=False)

# print(feat_imp.head(20))

print(opt_x_test[['MTO1','MTO2','歸屬班別','生產日期','數量']])

import optuna
import numpy as np

TARGET = 2.1  # 目標厚度 μm

# mto1_low, mto1_high = 0, 3
# mto2_low, mto2_high = 0, 2.5

current1_low, current1_high = 0, 1.05
current2_low, current2_high = 0, 0.35


# ==== 目標函數：只優化 MTO1 / MTO2，其他欄位固定使用 row_Test 的值 ====
def objective(trial: optuna.Trial):
    current11 = trial.suggest_float("電流值1", current1_low, current1_high)
    current12 = trial.suggest_float("電流值2", current2_low, current2_high)

    # 取 row_Test 的第一筆（或你可改成 iloc[-1] 取最後一筆）
    base_row = opt_x_test.iloc[0].copy()

    # 替換 MTO1/MTO2，其餘欄位維持不變
    base_row['電流值1'] = current11
    base_row['電流值2'] = current12

    # 建立單筆特徵 DataFrame（欄位順序與訓練一致）
    X_new = pd.DataFrame([base_row])

    # 由已訓練模型推論金厚度
    y_hat = float(best_model.predict(X_new)[0])

    # 單目標：使預測厚度貼近 TARGET（可改成平方誤差或 Huber）
    loss = abs(y_hat - TARGET)
    #loss = (y_hat - TARGET) ** 2


    # （選配）你也可加入製程風險/成本懲罰，例如偏離名義值：
    # loss += 0.001 * max(0, mto1 - 名義上限) + 0.001 * max(0, 名義下限 - mto1)
    
    return loss

# ==== 執行最佳化 ====
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=False)

# ==== 最佳解輸出 ====
best_params = study.best_trial.params
best_row = opt_x_test.iloc[0].copy()
best_row['電流值1'] = best_params['電流值1']
best_row['電流值2'] = best_params['電流值2']

X_best = pd.DataFrame([best_row])
pred_best = float(best_model.predict(X_best)[0])

print("\n=== Optimization Result ===")
print("Best params:", best_params)                 # {'MTO1': ..., 'MTO2': ...}
print("Predicted thickness:", pred_best)           # 最佳參數下的預測金厚度
print("Abs error to TARGET:", abs(pred_best - TARGET))