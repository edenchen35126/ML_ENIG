import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV,train_test_split
from collections import defaultdict
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import shap
from joblib import Memory
from xgboost import XGBRegressor
from xgboost import XGBRegressor, callback as xcb
from xgboost.callback import EarlyStopping
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn import set_config
set_config(transform_output="pandas")  # ✅ 讓 ColumnTransformer 自動輸出 DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
import time
import polars as pl

# =========================
# 視覺化字型（可留可去）
# =========================
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 全域設定
# =========================
RANDOM_STATE = 42
OUTER_FOLDS = 5        # 外層 KFold（最終評估）
INNER_FOLDS = 5        # 內層 KFold（參數搜尋用）
N_ITER = 30            # RandomizedSearchCV 抽樣次數（可視算力調整）  從參數分佈 (param_distributions) 中隨機抽取 60 組不同的參數組合來訓練與交叉驗證。

# =========================
# 讀檔
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw", "化金線自主檢查表_all.csv")
df = pd.read_csv(data_path, encoding="big5")

df = pd.read_csv(data_path, encoding="big5")

print("=== 欄位缺值數量 ===")
print(df.isnull().sum())


# =========================
# 目標與特徵宣告（原始欄位名）
# =========================
label_col = "金"

# features = ['歸屬班別','料號','短批','子批','批號','數量','MTO1','MTO2','檢查型態','項目','鎳','金厚下限','金厚上限','鎳厚下限','鎳厚上限','板子類型','電流值1','電流值2',
#             '槽次1','槽次2','線別']
features = ['金厚下限','板子類型','歸屬班別','金厚上限','線別','槽次1','鎳','數量','檢查型態','鎳厚下限','MTO1','鎳厚上限','電流值1','電流值2','MTO2','項目']
#features = ['電流值1','電流值2','MTO1','MTO2','子批','料號','短批','批號']


# =========================
# XGBOOST 參數搜尋空間
# =========================
# def get_param_dist_xgb():
#     """XGBoost 高維度資料適用的 RandomizedSearch 參數空間（已移除 'auto'）"""
#     return {
#         # === 樹的結構參數（偏保守設定，避免高維特徵過擬合） ===
#         "model__n_estimators": np.linspace(300, 900, 7, dtype=int),  # 樹數量
#         "model__max_depth": [3, 5, 7, 9, 12],                       # 避免太深導致高維過擬合
#         "model__min_child_weight": [1, 3, 5, 7, 10],                 # 節點最小權重和
        
#         # === 學習率與收斂控制（learning rate 較低以提升穩定性） ===
#         "model__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],        # Boosting 步長，0.05~0.1 較穩定
#         "model__gamma": [0, 0.5, 1, 2, 5],                           # 分裂懲罰項，增加可泛化性
        
#         # === 子樣本與特徵取樣（高維特徵需更高隨機性以控制方差） ===
#         "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],               # 行取樣比例（樣本層級）
#         "model__colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7],        # 列取樣比例（特徵層級，特別重要）
        
#         # === 正則化控制（避免高維權重爆炸） ===
#         "model__reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],              # L2 正則化
#         "model__reg_alpha": [0, 0.1, 0.3, 0.5, 1.0],                 # L1 正則化

#         # === 模型結構 ===
#         "model__booster": ["gbtree", "dart"],                        # dart 支援 dropout，降低過擬合
#         "model__tree_method": ["hist"],                              # 適合中大型資料集
#     }

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

# 可列出輸出特徵名稱的週期轉換器
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """可列出輸出特徵名稱的週期轉換器"""
    def fit(self, X, y=None):
        self.columns_ = X.columns if isinstance(X, pd.DataFrame) else [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)
        return cyclical_encode(X)

    def get_feature_names_out(self, input_features=None):
        # 用空資料框跑一遍 cyclical_encode() 取得新欄位名
        if input_features is None:
            input_features = getattr(self, "columns_", [])
        df_temp = pd.DataFrame({c: [0] for c in input_features})
        df_encoded = cyclical_encode(df_temp)
        return df_encoded.columns.to_numpy()

def cyclical_encode(df):
    df = df.copy()

    for c in df.columns:
        if c.endswith("_hour"):
            df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 24)
            df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 24)
            # time_features.append(f"{c}_sin")
            # time_features.append(f"{c}_cos")
            #out_cols += [f"{c}_sin", f"{c}_cos"]
        elif c.endswith("_dow"):
            df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 7)
            df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 7)
            #out_cols += [f"{c}_sin", f"{c}_cos"]
            # time_features.append(f"{c}_sin")
            # time_features.append(f"{c}_cos")
        elif c.endswith("_month"):
            df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 12)
            df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 12)
            #out_cols += [f"{c}_sin", f"{c}_cos"]
            # time_features.append(f"{c}_sin")
            # time_features.append(f"{c}_cos")

    return df

# def cyclical_encode(df,time_features):
#     df = df.copy()
#     #out_cols = []
#     for c in df.columns:
#         if c.endswith("_hour"):
#             df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 24)
#             df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 24)
#             time_features.append(f"{c}_sin")
#             time_features.append(f"{c}_cos")
#             #out_cols += [f"{c}_sin", f"{c}_cos"]
#         elif c.endswith("_dow"):
#             df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 7)
#             df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 7)
#             #out_cols += [f"{c}_sin", f"{c}_cos"]
#             time_features.append(f"{c}_sin")
#             time_features.append(f"{c}_cos")
#         elif c.endswith("_month"):
#             df[f"{c}_sin"] = np.sin(2 * np.pi * df[c] / 12)
#             df[f"{c}_cos"] = np.cos(2 * np.pi * df[c] / 12)
#             #out_cols += [f"{c}_sin", f"{c}_cos"]
#             time_features.append(f"{c}_sin")
#             time_features.append(f"{c}_cos")
#     return df


# ==== LightGBM 參數搜尋空間 ====
def get_param_dist_lgb():
    """LightGBM 的 RandomizedSearch 參數空間"""
    return {
        "model__n_estimators": [200, 400, 600, 800, 1000],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__num_leaves": [15, 31, 63, 127],
        "model__max_depth": [-1, 5, 10, 15],
        "model__min_child_samples": [5, 10, 20, 40],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0, 0.1, 1, 5, 10],   # L2 regularization
        "model__reg_alpha": [0, 0.1, 0.5, 1],      # L1 regularization
        "model__boosting_type": ["gbdt", "dart"],  # dart 支援 dropout boosting
    }


def get_param_dist_dt():
    """Decision Tree 專用的 RandomizedSearchCV 搜尋空間"""
    return {
        "model__max_depth": [3, 5, 7, 10, 15, 20, None],  # 樹的最大深度
        "model__min_samples_split": [2, 5, 10, 20],       # 分裂所需的最小樣本數
        "model__min_samples_leaf": [1, 2, 4, 8, 10],      # 葉節點最小樣本數
        "model__max_features": ["sqrt", "log2", None],    # 分裂時考慮的特徵數
        "model__criterion": ["squared_error", "friedman_mse"],  # 損失函數
        "model__splitter": ["best", "random"]             # 分裂策略
    }

def get_param_dist_xgb():
    # 收斂後的「資源友善版」搜尋空間
    return {
        # "model__n_estimators": np.linspace(200, 500, 7, dtype=int),  # 小一點，交給 early stopping
        # "model__max_depth": [3, 5, 7],                               # 限制樹深
        # "model__min_child_weight": [3, 5, 7, 10],                    # 增加節點最小權重，抑制過擬合/計算量
        # "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
        # "model__gamma": [0, 0.5, 1, 2],
        # "model__subsample": [0.6, 0.7, 0.8],
        # "model__colsample_bytree": [0.3, 0.4, 0.5],
        # "model__reg_lambda": [0.8, 1.0, 1.5, 2.0],
        # "model__reg_alpha": [0, 0.1, 0.3, 0.5],
        # "model__booster": ["gbtree", "dart"],
        # "model__tree_method": ["hist"],  # 有 GPU 再改 "gpu_hist"
        # # "model__max_bin": [128, 256]   # 如記憶體吃緊可打開（對 hist 有效）

        "model__n_estimators": randint(1495, 1500),  # 小一點，交給 early stopping
        #"model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [3, 5, 7],                               # 限制樹深
        "model__min_child_weight": [3, 5, 7, 10],                    # 增加節點最小權重，抑制過擬合/計算量
        "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
        "model__gamma": [0, 0.5, 1, 2],
        "model__subsample": [0.6, 0.7, 0.8],
        "model__colsample_bytree": [0.3, 0.4, 0.5],
        "model__reg_lambda": [0.8, 1.0, 1.5, 2.0],
        "model__reg_alpha": [0, 0.1, 0.3, 0.5],
        "model__booster": ["gbtree"],
        "model__tree_method": ["hist"],  # 有 GPU 再改 "gpu_hist"
        # "model__max_bin": [128, 256]   # 如記憶體吃緊可打開（對 hist 有效）
    }


# =========================
# RF 參數搜尋空間
# =========================
def get_param_dist():
    """RandomizedSearch 的參數空間（已移除 'auto'，避免新版 sklearn 錯誤）"""
    return {
        "model__n_estimators": np.linspace(200, 800, 7, dtype=int),
        "model__max_depth": [None, 10, 15, 20, 25, 30],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],  #高維度較適合 (搜索空間收斂、方差較小)
        "model__bootstrap": [True, False],
    }
def get_rf_param_distributions(): #不適合高維度資料量跟維度
    return {
        "model__n_estimators": randint(300, 1201),     # 300~1200
        "model__max_depth":   [None] + list(range(5, 51)),
        # 若希望用「整數特徵數」而非比例，改成：randint(2, 1 + len(features))
        "model__max_features": uniform(0.3, 0.6),      # 0.3~0.9 的比例    #不適合高維度數據，因為會讓 RF 在每個節點看到太多特徵，在樣本量有限、訊號稀疏時會使模型不穩
        "model__min_samples_split": randint(2, 21),    # 2~20
        "model__min_samples_leaf":  randint(1, 11),    # 1~10
        "model__bootstrap": [True, False],
    }

def ramdom_forest_model(x_train,y_train):
    rf = RandomForestRegressor(
        random_state=42,
        n_estimators=600,
        oob_score=False,
        min_samples_split = 10,
        min_samples_leaf = 1,
        max_features = 7,
        max_depth = None,
        bootstrap = True
    )

    rf.fit(x_train, y_train)

    return rf

def iqr_clip(series, q1=0.25, q3=0.75, k=3.0):
    """Robustly clip outliers; default k=3 (比 1.5 寬鬆、較穩定)"""
    s = pd.Series(series)
    q_low, q_hi = s.quantile(q1), s.quantile(q3)
    iqr = q_hi - q_low
    lo = q_low - k * iqr
    hi = q_hi + k * iqr
    return s.clip(lower=lo, upper=hi)

def Std(df_num):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=num_cols,
        index=df_num.index
    )

    return X_scaled_df

def one_hot(df_cat):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(df_cat)

    X_encoded_df = pd.DataFrame(
        X_encoded,
        columns=encoder.get_feature_names_out(cat_cols),  # 這裡一定要用 encoder，而不是 function
        index=df_cat.index
    )

    return X_encoded_df

# ==== [Preprocess & Model Builders] ==========================================
def build_preprocessor(df: pd.DataFrame, features: list, time_features: list):
    """建構 ColumnTransformer 前處理管線"""
    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols and c not in time_features]
    # num_cols = [
    #     c for c in features
    #     if c not in cat_cols
    #     and c not in time_features
    #     and not np.issubdtype(df[c].dtype, np.datetime64)
    # ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    #time_transformer = FunctionTransformer(cyclical_encode, validate=False)
    

    # 👉 新增一個「time passthrough」分支
    #passthrough_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
            #("time", passthrough_transformer, time_features),  # ✅ 直接帶入不變動
            #("time", time_transformer, time_features),  # 分支：時間特徵轉週期編碼
            ("time", CyclicalEncoder(), time_features)
        ]
    )
    return preprocessor, cat_cols, num_cols

# =========================
# 外層 KFold（Nested CV）
# =========================
outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = []
fi_collector = defaultdict(list)  # 跨折特徵重要度彙整


# ---- Date features ----
used_time_col = None

used_time_col = "生產日期"

if used_time_col:
    df = parse_dates(df, [used_time_col])
    df = add_date_features(df, used_time_col)
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

#df = cyclical_encode(df,time_features)  # 這裡直接呼叫你的函式


for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(df), start=1):
    # cat_cols = [c for c in features if df[c].dtype == "object"]
    # num_cols = [c for c in features if c not in cat_cols]

    # # ---- Numeric cleanup ----
    # for c in num_cols:
    #     try:
    #         df[c] = pd.to_numeric(df[c], errors="coerce")
    #         df[c] = iqr_clip(df[c])
    #     except Exception:
    #         pass

    # target = "金"
    # X = df[features].copy()
    # y = df[target].astype(float)
    # # ---- Train/Test split ----
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    
    cat_cols = [c for c in features if df_train[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]


    # used_time_col = "生產日期"

    # if used_time_col:
    #     df_train = parse_dates(df_train, [used_time_col])
    #     df_train = add_date_features(df_train, used_time_col)
    #     #time_features = []
    #     time_features = [
    #         f"{used_time_col}_year",
    #         f"{used_time_col}_month",
    #         f"{used_time_col}_day",
    #         f"{used_time_col}_dow", # day of week
    #         f"{used_time_col}_hour"
    #     ]
    #     #features.extend(time_features)
    # else:
    #     time_features = []
    
    # df_train = cyclical_encode(df_train,time_features)  # 這裡直接呼叫你的函式

    # if used_time_col:
    #     df_test = parse_dates(df_test, [used_time_col])
    #     df_test = add_date_features(df_test, used_time_col)
    # else:
    #     time_features = []

    # df_test = cyclical_encode_v2(df_test,time_features)  # 這裡直接呼叫你的函式




    # if num_cols:
    #     df_num = df[num_cols].fillna(df_train[num_cols].median())
    #     X_scaled_df = Std(df_num)
    # else:
    #     X_scaled_df = pd.DataFrame()

    # if cat_cols:
    #     # 1) 先算出每個類別欄位的眾數（第一列）
    #     cat_modes = df_train[cat_cols].mode(dropna=True).iloc[0]
    #     # 2) 對目前要處理的 df（可能是 train 或 test）做缺失值補眾數
    #     df_cat = df[cat_cols].fillna(cat_modes)
    #     X_encoded_df  = one_hot(df_cat)
    # else:
    #     X_encoded_df = pd.DataFrame()


    # # 3) 合併成一個 DataFrame
    # X_final = pd.concat([X_scaled_df, X_encoded_df,df[label_col]], axis=1)

    # df_train = X_final.iloc[train_idx].copy()
    # df_test  = X_final.iloc[test_idx].copy()




    # # 1) 缺值補值（僅用訓練 fold 的統計值）
    # for col in impute_cols_with_mode:
    #     if col in df_train.columns:
    #         mode_val = df_train[col].mode(dropna=True)
    #         if not mode_val.empty:
    #             mode_val = mode_val[0]  #如果有多個數值是眾數，選擇第一個
    #             df_train[col] = df_train[col].fillna(mode_val)
    #             df_test[col]  = df_test[col].fillna(mode_val)

    # # 2) 類別欄位 Ordinal 編碼（僅用訓練 fold fit，避免洩漏）
    # enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    # enc.fit(df_train[categorical_cols_for_ordinal].astype(str))

    # # 一次性 transform 訓練/測試集的所有類別欄位
    # train_enc = enc.transform(df_train[categorical_cols_for_ordinal].astype(str))
    # test_enc  = enc.transform(df_test[categorical_cols_for_ordinal].astype(str))

    # # 將各欄結果回填成 *_Code 欄位（保持與 categorical_cols_for_ordinal 的順序一致）
    # for i, c in enumerate(categorical_cols_for_ordinal):
    #     df_train[f"{c}_Code"] = train_enc[:, i].astype(np.int64)
    #     df_test[f"{c}_Code"]  = test_enc[:, i].astype(np.int64)

    # # 3) 料號頻率編碼（僅用訓練 fold 的分佈）
    # features = []
    # for f in base_raw_features:
    #     if f in categorical_cols_for_ordinal:
    #         features.append(f"{f}_Code")  # 用編碼後欄位
    #     else:
    #         features.append(f)

    # if '料號' in df.columns:
    #     freq_map = df_train['料號'].value_counts()
    #     df_train['料號_FreqEnc'] = df_train['料號'].map(freq_map).fillna(0).astype(int)
    #     df_test['料號_FreqEnc']  = df_test['料號'].map(freq_map).fillna(0).astype(int)
    #     features = features + ['料號_FreqEnc']

    # exclude_cols = categorical_cols_for_ordinal + ['料號_FreqEnc']
    # iqr_clip_feature = [f for f in features if f not in exclude_cols and "_Code" not in f]

    # print(iqr_clip_feature)
    
    # for c in iqr_clip_feature:
    #     try:
    #         df[c] = pd.to_numeric(df[c], errors="coerce")
    #         df[c] = iqr_clip(df[c])
    #     except Exception:
    #         pass
    
    # exclude_cols = ['']

    # iqr_clip_feature = [f for f in base_features if f not in exclude_cols and "_Code" not in f]

    # === 過濾掉時間衍生特徵，只留給 preprocessor 用的欄位 ===
    features_for_preproc = [f for f in features if f not in time_features]

    # === 傳給 build_preprocessor 的就是排除後的版本 ===
    preprocessor, cat_cols, num_cols = build_preprocessor(df, features_for_preproc , time_features)
    # print(features_for_preproc)
    # print(time_features)
    # exit()

    all_features = features_for_preproc + time_features


    X_train = df_train[all_features]
    y_train = df_train[label_col].values.ravel()
    X_test  = df_test[all_features]
    y_test  = df_test[label_col].values.ravel()


    #preprocessor, cat_cols, num_cols = build_preprocessor(df, features)

    inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ==== LightGBM 模型 ====
    # lgb = LGBMRegressor(
    #     objective="regression",
    #     random_state=RANDOM_STATE,
    #     n_jobs=-1,
    #     verbose=-1,
    #     force_col_wise=True,   # ✅ 對 ColumnTransformer 輸出最相容
    # )

    # # ==== Pipeline ====
    # pipe = Pipeline(steps=[
    #     ("preprocess", preprocessor),  # 保留你的前處理流程
    #     ("model", lgb)
    # ])

    # # ==== RandomizedSearchCV ====
    # search = RandomizedSearchCV(
    #     estimator=pipe,
    #     param_distributions=get_param_dist_lgb(),
    #     n_iter=N_ITER,
    #     cv=inner_cv,
    #     scoring="neg_mean_absolute_error",
    #     verbose=1,
    #     n_jobs=-1,
    #     random_state=RANDOM_STATE,
    #     refit=True,
    # )


    # 測試 ColumnTransformer 輸出欄位
    X_train_pre = preprocessor.fit_transform(X_train)
    print("✅ ColumnTransformer 輸出 shape:", X_train_pre.shape)
    print("✅ 欄位名稱前10:", X_train_pre.columns.tolist())

    # exit()
    # exit()

    # ==== 訓練 ====
    # lgb_model = search.fit(X_train, y_train)

    # # ==== Decision Tree 模型 ====
    # dt = DecisionTreeRegressor(
    #     random_state=RANDOM_STATE,
    # )

    # # ==== Pipeline ====
    # pipe = Pipeline(steps=[
    #     ("preprocess", preprocessor),  # 前處理（縮放、編碼等）
    #     ("model", dt)
    # ])

    # # ==== RandomizedSearchCV ====
    # search = RandomizedSearchCV(
    #     estimator=pipe,
    #     param_distributions=get_param_dist_dt(),
    #     n_iter=N_ITER,
    #     cv=inner_cv,
    #     scoring="neg_mean_absolute_error",   # 可改 r2 / neg_root_mean_squared_error
    #     verbose=1,
    #     n_jobs=-1,
    #     random_state=RANDOM_STATE,
    #     refit=True,
    # )

    # # ==== 訓練 ====
    # dt_model = search.fit(X_train, y_train)


    # # ==== XGBoost 模型 ====
    # xgb = XGBRegressor(
    #     objective="reg:squarederror",   # 迴歸任務
    #     random_state=RANDOM_STATE,
    #     tree_method="gpu_hist"       # 有 GPU 改 "gpu_hist"
        
    # )

    # # ==== Pipeline ====
    # pipe = Pipeline(steps=[
    #     ("preprocess", preprocessor),   # 前處理（例如縮放、編碼等）
    #     ("model", xgb)
    # ])

    # # ==== RandomizedSearchCV ====
    # search = RandomizedSearchCV(
    #     estimator=pipe,
    #     param_distributions=get_param_dist_xgb(),
    #     n_iter=N_ITER,
    #     cv=inner_cv,
    #     scoring="neg_mean_absolute_error",  # 可改為 r2 / neg_root_mean_squared_error
    #     verbose=1,
    #     n_jobs=-1,
    #     random_state=RANDOM_STATE,
    #     refit=True,   # 以最佳參數重訓
    # )

    # # # ==== 訓練 ====
    # xgb_model = search.fit(X_train, y_train)

    # 5) 內層 RandomizedSearchCV + KFold（在訓練 fold 上尋參）
    rf = RandomForestRegressor(random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=get_param_dist(), #要搜尋的「超參數空間（parameter space）」
        n_iter=N_ITER, # 組合數量
        cv=inner_cv,  # 交叉驗證的設定
        scoring="neg_mean_absolute_error", # 模型評分指標
        verbose=1, # 訓練過程的詳細程度
        n_jobs=-1, # CPU 平行化數量，-1指全CPU
        random_state=RANDOM_STATE, # 固定隨機性來源，確保實驗可重現
        refit=True,  # 以最佳參數在整個訓練 fold 重訓
    )
    rf_model = search.fit(X_train, y_train)

    print(f"[Fold {fold_idx}] Best params:", search.best_params_)
    print(f"[Fold {fold_idx}] Best MAE (cv, neg): {search.best_score_:.6f} | Best MAE: {-search.best_score_:.6f}")

    best_model = search.best_estimator_
    rf_model = best_model.named_steps["model"]     # 已 fit 的 RandomForestRegressor
    preprocessor = best_model.named_steps["preprocess"]   # 已 fit
    #preprocessor = best_model.named_steps["preprocess"]

    # 6) 外層測試集評估
    y_pred = best_model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    fold_metrics.append({
        "fold": fold_idx,
        "MAE":  mae,
        "MSE":  mse,
        "RMSE": rmse,
        "R2":   r2,
        "best_params": search.best_params_,
        "best_model": search.best_estimator_
    })
    print(f"[Fold {fold_idx}] MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # 7) 特徵重要度（可選）
    if hasattr(best_model, "feature_importances_"):
        for f, imp in zip(features, best_model.feature_importances_):
            fi_collector[f].append(imp)

    # ---- 9.1 準備資料（取一個代表性的樣本，避免全量太慢） ----------------------
    # SHAP 繪圖對樣本數較敏感，數萬筆以上可抽樣 2000~5000 筆即可呈現趨勢
    #rf_model = pipe.named_steps["model"] 
    #xgb_model = pipe.named_steps["model"] 
    #############################################################
    # 抽樣（避免過慢）
    # def sample_frame(X, n=3000, random_state=42):
    #     return X if len(X) <= n else X.sample(n=n, random_state=random_state)

    # X_shap_raw = sample_frame(X_test, n=3000, random_state=42)

    # # ★ 必須：用已 fit 的前處理器把原始特徵轉成數值矩陣
    # X_shap_tx = preprocessor.transform(X_shap_raw)
    # if hasattr(X_shap_tx, "toarray"):   # sparse → dense
    #     X_shap_tx = X_shap_tx.toarray()

    # # 取最終特徵名稱（對 OneHot 非常重要）
    # try:
    #     feat_names_out = preprocessor.get_feature_names_out()
    # except Exception:
    #     feat_names_out = np.array([f"f{i}" for i in range(X_shap_tx.shape[1])])

    # # ---- 9.2 計算 SHAP 值 -------------------------------------------------------
    # # 對於樹模型，使用 TreeExplainer（效能/相容性較佳）
    # explainer = shap.TreeExplainer(lgb_model)
    # #shap_values = explainer.shap_values(X_shap_tx, check_additivity=False)   # regression: (n_samples, n_features)
    # shap_values = explainer.shap_values(X_shap_tx)   # regression: (n_samples, n_features)
    # # 基準值（期望輸出）
    # expected_value = explainer.expected_value

    # X_shap_df = pd.DataFrame(X_shap_tx, columns=feat_names_out)

    # # ---- 9.3 匯出 mean(|SHAP|) 作為特徵影響力指標 -------------------------------
    # mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # shap_importance_df = (
    #     pd.DataFrame({"feature": feat_names_out, "mean_abs_shap": mean_abs_shap})
    #     .sort_values("mean_abs_shap", ascending=False)
    # )
    # print("\n[SHAP] Top 20（mean |SHAP|）")
    # print(shap_importance_df.head(20))

    # # ======================================================
    # # ✅ 新增：Group OneHot → 原始特徵層級
    # # ======================================================
    # import re
    # from collections import defaultdict

    # def base_name_without_prefix(feat: str) -> str:
    #     """去掉 ColumnTransformer 的前綴（例如 'cat__', 'num__'）"""
    #     return re.sub(r'^[^_]+__', '', feat)  # 刪掉最前面的 '<prefix>__'

    # def map_to_original_feature(feat: str, cat_cols, num_cols) -> str:
    #     """
    #     將 One-Hot 後欄位對應回原始欄位：
    #     - 類別：cat__料號_A / cat__料號[A] → 料號
    #     - 數值：num__數量 → 數量
    #     """
    #     name = base_name_without_prefix(feat)

    #     # 先檢查是否為數值欄位（完全相等即可）
    #     for c in num_cols:
    #         if name == c:
    #             return c

    #     # 再檢查是否為類別欄位（名稱以「原欄位名_」或「原欄位名[」開頭）
    #     for c in cat_cols:
    #         if name.startswith(c + "_") or name.startswith(c + "["):
    #             return c

    #     # 萬一兩者都沒命中，就嘗試更保守地還原：去掉類別值尾巴（_xxx 或 [xxx]）
    #     name2 = re.sub(r'\[.*\]$', '', name)   # 去掉 [xxx]
    #     name2 = re.sub(r'_(?!.*_).*$', '', name2)  # 去掉最後一個 '_' 後的字
    #     return name2

    # # 產生「One-Hot 輸出欄位」→「原始欄位」的對應表
    # feat_group_map = {}
    # for f in feat_names_out:
    #     feat_group_map[f] = map_to_original_feature(f, cat_cols=cat_cols, num_cols=num_cols)

    # # 依原始欄位彙總 mean(|SHAP|)
    # grouped_importance = defaultdict(float)
    # for f_name, shap_val in zip(feat_names_out, mean_abs_shap):
    #     orig_name = feat_group_map.get(f_name, f_name)
    #     grouped_importance[orig_name] += shap_val  # 也可改成 .mean()；此處用加總代表整體影響力

    # grouped_shap_importance_df = (
    #     pd.DataFrame(list(grouped_importance.items()), columns=["feature", "mean_abs_shap"])
    #     .sort_values("mean_abs_shap", ascending=False)
    #     .reset_index(drop=True)
    # )

    # print("\n[SHAP] Top 20（mean |SHAP|，合併回原始欄位）")
    # print(grouped_shap_importance_df)

    # # 視覺化（列出各個原始特徵）
    # #topN = 20
    # plt.figure(figsize=(10, 0.4 * len(grouped_shap_importance_df)))  # 根據特徵數量自動調整高度
    # plt.barh(
    #     grouped_shap_importance_df["feature"][::-1],         # 所有特徵（反轉讓最重要的在最上面）
    #     grouped_shap_importance_df["mean_abs_shap"][::-1]
    # )
    # plt.xlabel("Mean |SHAP value|")
    # plt.title("Grouped SHAP Feature Importance (All Features)")
    # plt.tight_layout()
    # plt.show()

    # x = np.arange(len(y_test))

    # plt.figure(figsize=(12, 6))
    # plt.plot(x, y_pred, label="True (y_test)", color="blue", linewidth=2)
    # plt.plot(x, y_test, label="Predicted (y_pred)", color="red", linestyle="--", linewidth=2)

    # plt.title("Prediction vs True Value", fontsize=16)
    # plt.xlabel("Sample Index", fontsize=14)
    # plt.ylabel("Value", fontsize=14)
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #############################################################


# =========================
# 結果彙整
# =========================
metrics_df = pd.DataFrame(fold_metrics)
print("\n=== K-Fold 成績（逐折） ===")
print(metrics_df)

print("\n=== K-Fold 平均成績 ===")
print(metrics_df[["MAE", "MSE", "RMSE", "R2"]].mean())

# 跨折平均特徵重要度
if len(fi_collector) > 0:
    fi_avg = pd.DataFrame({
        "feature": list(fi_collector.keys()),
        "importance_mean": [np.mean(vals) for vals in fi_collector.values()],
        "importance_std":  [np.std(vals)  for vals in fi_collector.values()],
    }).sort_values("importance_mean", ascending=False)
    print("\n=== 平均特徵重要度（跨折） ===")
    print(fi_avg.to_string(index=False))


### 最佳化參數儲存，並重訓後將模型儲存

# from collections import Counter
# import joblib

# # 統計每次 fold 的最佳參數
# param_counter = Counter([tuple(sorted(d["best_params"].items())) for d in fold_metrics])
# final_params = dict(param_counter.most_common(1)[0][0])  # 出現次數最多的組合

# # ✅ 移除前綴 "model__"
# model_params = {k.replace("model__", ""): v for k, v in final_params.items() if k.startswith("model__")}


# # 重建 Pipeline（用同樣的 preprocessor）
# final_model = Pipeline(steps=[
#     ("preprocess", preprocessor),  # 或重建新的 fit_transform
#     ("model", RandomForestRegressor(random_state=42, **model_params))
# ])

# train_len = int(len(df)*0.8)
# model_train = df[:train_len]
# model_test = df[train_len:]

# final_model.fit(model_train[all_features], model_train[label_col])
# joblib.dump(final_model, "models/final_rf_without_part_num_divide_v2.pkl")

# print("✅ Final model retrained with all data and best params.")
