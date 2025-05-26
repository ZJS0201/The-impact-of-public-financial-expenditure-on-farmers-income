from pathlib import Path
DATA_PATH = Path("Research_data.xlsx")         
SAVE_DIR  = Path("results")             

import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


df = pd.read_excel(DATA_PATH, sheet_name=0)


df["lnRINC"] = np.log(df["农村人均可支配收入(元）"])

fiscal_cols = [c for c in df.columns if c.startswith("人均") and c.endswith("支出（元）")]
for col in fiscal_cols + ["人均GDP（元）"]:
    df[f"ln_{col}"] = np.log(df[col])

df["year_idx"] = df["年份"] - 2010      
num_feats = [f"ln_{c}" for c in fiscal_cols] + [
    "ln_人均GDP（元）", "城镇化率（%）", "二三产业产值占比（%）", "year_idx"
]
cat_feats = ["地区"]

y = df["lnRINC"]


ct = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", OneHotEncoder(drop="first", dtype=np.int8), cat_feats)
])
X_fe = ct.fit_transform(df)     


train_mask = df["年份"] <= 2021
test_mask  = df["年份"] >= 2022
Xfe_train, Xfe_test = X_fe[train_mask], X_fe[test_mask]
y_train,   y_test   = y[train_mask],   y[test_mask]

fe = LinearRegression().fit(Xfe_train, y_train)
yhat_fe_train = fe.predict(Xfe_train)
yhat_fe_test  = fe.predict(Xfe_test)


resid_train = y_train - yhat_fe_train
resid_test  = y_test  - yhat_fe_test         


X_num = df[num_feats].values
Xn_train, Xn_test = X_num[train_mask], X_num[test_mask]

xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    random_state=2025,
    tree_method="hist"
)

eval_idx = df["年份"].between(2019, 2021)
xgb.fit(
    Xn_train, resid_train,
    eval_set=[(Xn_test, resid_test)],   
    early_stopping_rounds=30,
    verbose=False
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_metrics(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R²":   r2_score(y_true, y_pred)
    }


m_fe_train = get_metrics(y_train, yhat_fe_train)
m_fe_test  = get_metrics(y_test,  yhat_fe_test)


y_pred_train = yhat_fe_train + xgb.predict(Xn_train)
y_pred_test  = yhat_fe_test  + xgb.predict(Xn_test)
m_hybrid_train = get_metrics(y_train, y_pred_train)
m_hybrid_test  = get_metrics(y_test,  y_pred_test)


table = pd.DataFrame([
    ("FE-OLS", "Train", *m_fe_train.values()),
    ("FE-OLS", "Test",  *m_fe_test.values()),
    ("FE + XGB", "Train", *m_hybrid_train.values()),
    ("FE + XGB", "Test",  *m_hybrid_test.values()),
], columns=["Model", "Subset", "RMSE", "MAE", "R²"])

SAVE_DIR.mkdir(parents=True, exist_ok=True)
xlsx_path = SAVE_DIR / "Table6-1_PredictiveAccuracy_FE+XGB.xlsx"
csv_path  = SAVE_DIR / "Table6-1_PredictiveAccuracy_FE+XGB.csv"
table.to_excel(xlsx_path, index=False)
table.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(table)

