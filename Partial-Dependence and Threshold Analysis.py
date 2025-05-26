from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap, statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
try:                                            
    from linearmodels.panel.threshold import Threshold
except ImportError:                            
    from linearmodels.threshold import Threshold





DATA_PATH = Path("Research_data.xlsx")            
SAVE_DIR  = Path("results/6_3_figures")     
SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name=0)
map_cn2en = {
    "人均教育支出（元）":       "lnEDU",
    "人均卫生健康支出（元）":   "lnHEA",
    "人均农林水事务支出（元）": "lnAGR",
    "人均城乡社区事务支出（元）": "lnINF",
    "人均社会保障和就业支出（元）": "lnSOC",
    "人均一般公共服务支出（元）": "lnGFS",
    "人均科学技术支出（元）":       "lnTECH",
    "人均文化体育与传媒支出（元）": "lnCULT",
    "人均节能保护支出（元）":   "lnECO",
    "人均交通运输支出（元）":   "lnTRA",
    "人均住房保障支出（元）":   "lnHOU",
    "人均GDP（元）":       "lnPGDP",
    "城镇化率（%）":"URB",
    "二三产业产值占比（%）":"IND",
    "人均公共服务支出（元）":"INFS",
    "常住人口（万人）":"PRP",
    "农村人均可支配收入(元）":"lnRINC",
}


df["lnRINC"] = np.log(df["农村人均可支配收入(元）"])
fiscal_cols = [c for c in df.columns if c.startswith("人均") and c.endswith("支出（元）")]
for col in fiscal_cols + ["人均GDP（元）"]:
    df[f"ln_{col}"] = np.log(df[col])

df["year_idx"] = df["年份"] - df["年份"].min()

num_feats = [f"ln_{c}" for c in fiscal_cols] + [
    "ln_人均GDP（元）", "城镇化率（%）", "二三产业产值占比（%）", "year_idx"
]
cat_feats = ["地区"]


ct = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", OneHotEncoder(drop="first", dtype=np.int8), cat_feats)
])
X_fe = ct.fit_transform(df)
X_num = df[num_feats].values
y     = df["lnRINC"].values

feature_names_cn = num_feats
feature_names_en = []
for n in feature_names_cn:
    if n.startswith("ln_"):
        cn = n[3:]
        feature_names_en.append(map_cn2en.get(cn, n))
    elif n == "year_idx":
        feature_names_en.append("YEAR")
    elif n == "城镇化率（%）":
        feature_names_en.append("URB")
    elif n == "二三产业产值占比（%）":
        feature_names_en.append("IND")
    else:
        feature_names_en.append(n)
en2idx = {en: i for i, en in enumerate(feature_names_en)}


train_mask = df["年份"] <= 2021
test_mask  = df["年份"] >= 2022
Xfe_train, Xfe_test = X_fe[train_mask], X_fe[test_mask]
Xn_train, Xn_test   = X_num[train_mask], X_num[test_mask]
y_train, y_test     = y[train_mask], y[test_mask]

fe = LinearRegression().fit(Xfe_train, y_train)
resid_train = y_train - fe.predict(Xfe_train)
resid_test  = y_test  - fe.predict(Xfe_test)

xgb = XGBRegressor(
    n_estimators=300, learning_rate=0.03, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
    objective="reg:squarederror", random_state=2025,
    tree_method="hist"
)
xgb.fit(Xn_train, resid_train,
        eval_set=[(Xn_test, resid_test)],
        early_stopping_rounds=30, verbose=False)


explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(Xn_test)

from statsmodels.nonparametric.smoothers_lowess import lowess

def turning_point(x, y):
    """Return log‑space turning point using Lowess derivative sign change;
       if no change, return NaN."""
    sm = lowess(y, x, frac=0.3, return_sorted=True)
    grad = np.gradient(sm[:, 1])
    sign_change = np.where(np.diff(np.sign(grad)) < 0)[0]
    if len(sign_change):
        idx = sign_change[0]
        return sm[idx, 0]
    return np.nan

results = []
var_list = [("lnEDU", "figure6-9_dep_lnEDU.png"),
            ("lnHEA", "figure6-10_dep_lnHEA.png"),
            ("lnINF", "figure6-11_dep_lnINF.png"),
            ("lnSOC", "figure6-12_dep_lnSOC.png")]

for var_en, fname in var_list:
    i = en2idx[var_en]
    x = Xn_test[:, i]
    y_val = shap_vals[:, i]

    # Dependence plot (with color by lnCULT, if exists)
    plt.figure()
    shap.dependence_plot(i, shap_vals, Xn_test,
                         feature_names=feature_names_en, show=False,
                         interaction_index=None)
    # Lowess curve
    smoothed = lowess(y_val, x, frac=0.3, return_sorted=True)
    plt.plot(smoothed[:,0], smoothed[:,1], color="black", linewidth=1)
    tp_ln = turning_point(x, y_val)
    if not np.isnan(tp_ln):
        plt.axvline(tp_ln, color="red", linestyle="--")
    plt.title(f"Dependence plot of {var_en}")
    plt.tight_layout()
    plt.savefig(SAVE_DIR/fname, dpi=300)
    plt.close()

    tp_amt = np.exp(tp_ln) if not np.isnan(tp_ln) else np.nan
    results.append({"Variable": var_en, "SHAP TP (ln)": tp_ln, "SHAP TP (¥)": tp_amt})

try:
    from linearmodels import PanelOLS
    from linearmodels.threshold import Threshold
    panel = df.copy()
    panel["entity"] = panel["地区"]
    panel = panel.set_index(["entity", "年份"])
    panel["lnEDU"] = X_num[:, en2idx["lnEDU"]]
    panel["lnHEA"] = X_num[:, en2idx["lnHEA"]]
    panel["lnINF"] = X_num[:, en2idx["lnINF"]]
    panel["lnSOC"] = X_num[:, en2idx["lnSOC"]]

    for var_en in ["lnEDU","lnHEA","lnINF","lnSOC"]:
        mod = PanelOLS.from_formula("lnRINC ~ 1 + EntityEffects + TimeEffects", data=panel)
        ptr = Threshold(mod, threshold_var=panel[var_en])
        fit = ptr.fit(search_method='grid', grid=200)
        tp = fit.threshold
        for d in results:
            if d["Variable"] == var_en:
                d["PTR TP (ln)"] = tp
                d["PTR TP (¥)"]  = np.exp(tp)
except ImportError:
    for d in results:
        d["PTR TP (ln)"] = "NA"
        d["PTR TP (¥)"]  = "NA"

table63 = pd.DataFrame(results)
table63.to_excel(SAVE_DIR/"Table6-3_TurningPoints.xlsx", index=False)
table63.to_csv(SAVE_DIR/"Table6-3_TurningPoints.csv", index=False, encoding="utf-8-sig")

