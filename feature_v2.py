import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV,train_test_split
from collections import defaultdict
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import shap

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

print("=== 欄位缺值數量 ===")
print(df.isnull().sum())

# =========================
# 目標與特徵宣告（原始欄位名）
# =========================
label_col = "金"

# 原始特徵（注意：此處不做全域編碼；在 fold 內才做）
# 類別欄位在每個 fold 內用 OrdinalEncoder 產生 *_Code 取代
base_raw_features = ['金厚下限', '板子類型', '線別', 'MTO2', '電流值2', '槽次2']
# base_raw_features = ['電流值1', '電流值2', 'MTO1','MTO2','料號','短批','子批','批號']

# 每個 fold 內用訓練資料的眾數補值
impute_cols_with_mode = ['子批', '金厚上限', '槽次2']
# impute_cols_with_mode = ['子批']

# 在每個 fold 內做 Ordinal 編碼（避免洩漏） # label encoder
categorical_cols_for_ordinal = ['板子類型', '線別', '槽次2']

# features = ['歸屬班別','料號','短批','子批','批號','數量','MTO1','MTO2','檢查型態','項目','鎳','金厚下限','金厚上限','鎳厚下限','鎳厚上限','板子類型','電流值1','電流值2',
#             '槽次1','槽次2','線別']
features = ['金厚下限','板子類型','料號','歸屬班別','批號','短批','金厚上限','線別','槽次1','鎳','數量']
#features = ['電流值1','電流值2','MTO1','MTO2','子批','料號','短批','批號']

rene_features = ['電流值1','電流值2','MTO1','MTO2','子批','料號','短批','批號']

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
def build_preprocessor(df: pd.DataFrame, features: list):
    """建構 ColumnTransformer 前處理管線"""
    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols

# =========================
# 外層 KFold（Nested CV）
# =========================
outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_metrics = []
fi_collector = defaultdict(list)  # 跨折特徵重要度彙整


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

    # 4) 組合資料
    all_features = df_train.columns.to_list()
    all_features = [f for f in features if f not in label_col]

    X_train = df_train[all_features]
    y_train = df_train[label_col].values.ravel()
    X_test  = df_test[all_features]
    y_test  = df_test[label_col].values.ravel()

    preprocessor, cat_cols, num_cols = build_preprocessor(df, features)

    # 5) 內層 RandomizedSearchCV + KFold（在訓練 fold 上尋參）
    rf = RandomForestRegressor(random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", rf)])

    inner_cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=get_param_dist(),
        n_iter=N_ITER,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
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
        "R2":   r2
    })
    print(f"[Fold {fold_idx}] MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # 7) 特徵重要度（可選）
    if hasattr(best_model, "feature_importances_"):
        for f, imp in zip(features, best_model.feature_importances_):
            fi_collector[f].append(imp)

    # ---- 9.1 準備資料（取一個代表性的樣本，避免全量太慢） ----------------------
    # SHAP 繪圖對樣本數較敏感，數萬筆以上可抽樣 2000~5000 筆即可呈現趨勢
    #rf_model = pipe.named_steps["model"] 

    #############################################################
    # # 抽樣（避免過慢）
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
    # explainer = shap.TreeExplainer(rf_model)
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
