import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("Research_data.xlsx")    
SAVE_DIR  = Path("results/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)



df = pd.read_excel(DATA_PATH, sheet_name=0)
df["lnRINC"] = np.log(df["农村人均可支配收入(元）"])
fiscal_cols = [c for c in df.columns if c.startswith("人均") and c.endswith("支出（元）")]
for col in fiscal_cols + ["人均GDP（元）"]:
    df[f"ln_{col}"] = np.log(df[col])

num_feats = [f"ln_{c}" for c in fiscal_cols] + ["ln_人均GDP（元）", "城镇化率（%）", "二三产业产值占比（%）",]
cat_feats = ["地区"]


ct = ColumnTransformer([
    ("num", "passthrough", num_feats + ["year_idx"]),
    ("cat", OneHotEncoder(drop="first", dtype=np.int8), cat_feats)
])

df["year_idx"] = df["年份"] - 2010
X_fe = ct.fit_transform(df)
y    = df["lnRINC"].values
X_num = df[num_feats + ["year_idx"]].values


train_mask = df["年份"] <= 2021
test_mask  = df["年份"] >= 2022
Xfe_train, Xfe_test = X_fe[train_mask], X_fe[test_mask]
Xn_train, Xn_test   = X_num[train_mask], X_num[test_mask]
y_train, y_test     = y[train_mask], y[test_mask]


fe = LinearRegression().fit(Xfe_train, y_train)
yhat_fe_test = fe.predict(Xfe_test)
resid_train  = y_train - fe.predict(Xfe_train)
resid_test   = y_test  - yhat_fe_test


xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                   objective="reg:squarederror", random_state=2025,
                   tree_method="hist")
xgb.fit(Xn_train, resid_train, eval_set=[(Xn_test, resid_test)],
        early_stopping_rounds=30, verbose=False)
y_pred_test  = yhat_fe_test + xgb.predict(Xn_test)


rmse_fe  = np.sqrt(mean_squared_error(y_test, yhat_fe_test))
rmse_hyb = np.sqrt(mean_squared_error(y_test, y_pred_test))

plt.figure()
plt.bar(["FE‑OLS", "FE + XGB"], [rmse_fe, rmse_hyb])
plt.ylabel("RMSE (log income)")
plt.title("Figure 6‑1  Test‑set RMSE comparison")
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-1_rmse_bar.png", dpi=300)
plt.close()


tscv = TimeSeriesSplit(n_splits=5, test_size=2)
rmse_folds = []
for train_idx, test_idx in tscv.split(Xfe_train):
    fe_fold = LinearRegression().fit(Xfe_train[train_idx], y_train[train_idx])
    resid_fold = y_train[train_idx] - fe_fold.predict(Xfe_train[train_idx])
    xgb_fold = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                            objective="reg:squarederror", random_state=2025,
                            tree_method="hist")
    xgb_fold.fit(Xn_train[train_idx], resid_fold, verbose=False)
    y_pred_fold = fe_fold.predict(Xfe_train[test_idx]) + xgb_fold.predict(Xn_train[test_idx])
    rmse_folds.append(np.sqrt(mean_squared_error(y_train[test_idx], y_pred_fold)))

plt.figure()
plt.plot(range(1, len(rmse_folds)+1), rmse_folds, marker='o')
plt.xlabel("CV fold (rolling)")
plt.ylabel("RMSE")
plt.title("Figure 6‑2  Rolling CV RMSE of FE + XGB")
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-2_cv_rmse.png", dpi=300)
plt.close()


plt.figure()
plt.scatter(y_test, y_pred_test)
lims = [y_test.min()-0.2, y_test.max()+0.2]
plt.plot(lims, lims)
plt.xlabel("Actual lnRINC (test)")
plt.ylabel("Predicted lnRINC")
plt.title("Figure 6‑3  Actual vs. Predicted (test set)")
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-3_scatter.png", dpi=300)
plt.close()

print(f"✅ 图表已保存到 {SAVE_DIR.resolve()}")

