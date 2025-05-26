
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap, warnings, os

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
try:
    import lightgbm as lgb
    has_lgb = True
except ImportError:
    has_lgb = False
warnings.filterwarnings("ignore")


DATA_PATH = Path("Research_data.xlsx")
SAVE_DIR  = Path("results/6_5_robustness")
SAVE_DIR.mkdir(parents=True, exist_ok=True)



df = pd.read_excel(DATA_PATH, sheet_name=0)
df["lnRINC"] = np.log(df["农村人均可支配收入(元）"])
fiscal_cols = [c for c in df.columns if c.startswith("人均") and c.endswith("支出（元）")]
for col in fiscal_cols + ["人均GDP（元）"]:
    df[f"ln_{col}"] = np.log(df[col])
df["lnTOT"] = np.log(df[[f"ln_{c}" for c in fiscal_cols]].sum(axis=1))
df["year_idx"] = df["年份"] - df["年份"].min()

num_base = [f"ln_{c}" for c in fiscal_cols] + [
    "ln_人均GDP（元）", "城镇化率（%）", "二三产业产值占比（%）", "year_idx"
]
num_total = ["lnTOT", "ln_人均GDP（元）", "城镇化率（%）", "二三产业产值占比（%）", "year_idx"]
num_ratio = [f"ln_{c}" for c in fiscal_cols]  
for c in num_ratio:
    df[c+"_ratio"] = df[c] - df["ln_人均GDP（元）"]
num_ratio = [c+"_ratio" for c in num_ratio] + ["城镇化率（%）", "二三产业产值占比（%）", "year_idx"]

train = df["年份"] <= 2021
test  = df["年份"] >= 2022
y_train, y_test = df.loc[train, "lnRINC"], df.loc[test, "lnRINC"]


def fe_x_resid(X_train, X_test, model):
    fe = LinearRegression().fit(X_train, y_train)
    res_train = y_train - fe.predict(X_train)
    model.fit(X_train.values if hasattr(X_train,'values') else X_train, res_train)
    pred_test = fe.predict(X_test) + model.predict(X_test.values if hasattr(X_test,'values') else X_test)
    return pred_test, model

def shap_rank(model, X_test, var_names):
    shap_vals = shap.TreeExplainer(model).shap_values(X_test.values if hasattr(X_test,'values') else X_test)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    rank = mean_abs.argsort()[::-1]
    return rank, shap_vals

def tp_from_dependence(shap_vals, X_test, idx):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    x   = X_test.iloc[:, idx] if hasattr(X_test,'iloc') else X_test[:, idx]
    sm  = lowess(shap_vals[:, idx], x, frac=0.4, return_sorted=True)
    grad = np.gradient(sm[:,1])
    cross = np.where(np.diff(np.sign(grad))<0)[0]
    if cross.size:
        tp = sm[cross[0],0]
        return np.exp(tp)
    return np.nan


specs = [
    ("Baseline",        num_base,  XGBRegressor(n_estimators=300, learning_rate=0.03,
                                               max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                               reg_lambda=5, objective="reg:squarederror")),
    ("TotalSpending",   num_total, XGBRegressor(n_estimators=300, learning_rate=0.03,
                                               max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                               reg_lambda=5, objective="reg:squarederror")),
    ("Spend/GDP ratio", num_ratio, XGBRegressor(n_estimators=300, learning_rate=0.03,
                                               max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                               reg_lambda=5, objective="reg:squarederror")),
    ("RandomForest",    num_base,  RandomForestRegressor(n_estimators=500, max_depth=6,
                                                         random_state=2025)),
]

if has_lgb:
    specs.append(("LightGBM", num_base,
                  lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05,
                                    n_estimators=300, subsample=0.8,
                                    colsample_bytree=0.8, random_state=2025)))


rows=[]
for name, feat, model in specs:
    Xtr, Xte = df.loc[train, feat], df.loc[test, feat]
    pred_test, fitted_model = fe_x_resid(Xtr, Xte, model)

    R2  = np.corrcoef(pred_test, y_test)[0,1]**2
    rmse= np.sqrt(((pred_test - y_test)**2).mean())

    if hasattr(fitted_model, "feature_importances_") or "xgboost" in str(type(fitted_model)).lower():
        rank, shap_vals = shap_rank(fitted_model, Xte, feat)
        var_list = feat
        idx_edu = var_list.index('ln_人均教育支出（元）') if 'ln_人均教育支出（元）' in var_list else var_list.index('lnEDU') if 'lnEDU' in var_list else None
        idx_hea = var_list.index('ln_人均卫生健康支出（元）') if 'ln_人均卫生健康支出（元）' in var_list else var_list.index('lnHEA') if 'lnHEA' in var_list else None
        idx_inf = var_list.index('ln_人均城乡社区事务支出（元）') if 'ln_人均城乡社区事务支出（元）' in var_list else var_list.index('lnINF') if 'lnINF' in var_list else None

        tp_edu = tp_from_dependence(shap_vals, Xte, idx_edu) if idx_edu is not None else np.nan

        sign_edu_low = np.sign(shap_vals[:,idx_edu][Xte.iloc[:,idx_edu]<np.log(tp_edu)].mean()) if idx_edu is not None else np.nan
        sign_edu_high= np.sign(shap_vals[:,idx_edu][Xte.iloc[:,idx_edu]>np.log(tp_edu)].mean()) if idx_edu is not None else np.nan

        sign_hea_low = np.sign(shap_vals[:,idx_hea].mean()) if idx_hea is not None else np.nan
        sign_hea_high= np.nan  

        rank_inf = np.where(rank==idx_inf)[0][0]+1 if idx_inf is not None else np.nan
    else:
        tp_edu = sign_edu_low = sign_edu_high = sign_hea_low = rank_inf = np.nan

    rows.append([name, R2, rmse, sign_edu_low, sign_edu_high,
                 sign_hea_low, rank_inf, tp_edu])

robust = pd.DataFrame(rows, columns=["Spec","Test R²","RMSE",
                                     "EDU sign <TP","EDU sign >TP",
                                     "HEA overall sign","INF rank","EDU TP ¥"])
robust.to_excel(SAVE_DIR/"Table6-5_RobustnessSummary.xlsx", index=False)
robust.to_csv(SAVE_DIR/"Table6-5_RobustnessSummary.csv", index=False, encoding="utf-8-sig")


mask  = robust["EDU TP ¥"].notna()
vals  = robust.loc[mask, "EDU TP ¥"]
base  = vals.iloc[0]                    
delta = vals - base

if len(delta) >= 2:                      
    plt.figure(figsize=(6,3))
    plt.barh(robust.loc[mask, "Spec"], delta)
    plt.axvline(0, color="k")
    plt.xlabel("ΔEDU TP (¥) relative to Baseline")
    plt.title("Figure 6-16  Sensitivity of education turning point")
    plt.tight_layout()
    plt.savefig(SAVE_DIR/"figure6-16_tornado_tp_lnEDU.png", dpi=300)
    plt.close()
else:
    print("⚠️  仅 Baseline 有转折点，跳过 Tornado 图。")


