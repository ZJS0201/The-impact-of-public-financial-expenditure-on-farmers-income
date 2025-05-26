
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess


DATA_PATH = Path("Research_data.xlsx")                 
SAVE_DIR  = Path("results/6_4_figures")         
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


if {"一般公共预算本级收入", "一般公共预算支出"}.issubset(df.columns):
    df["self_cap"] = df["一般公共预算本级收入"] / df["一般公共预算支出"]
    high_mask = df["self_cap"] > 0.6
    group_label = "FiscalCapacity"
else:  
    median_pgdp = df["ln_人均GDP（元）"].median()
    high_mask = df["ln_人均GDP（元）"] >= median_pgdp
    group_label = "EconomicScale"

df["Group"] = np.where(high_mask, "High", "Low")


num_feats = [f"ln_{c}" for c in fiscal_cols] + [
    "ln_人均GDP（元）", "城镇化率（%）", "二三产业产值占比（%）", "year_idx"
]
cat_feats = ["地区"]

ct = ColumnTransformer([
    ("num", "passthrough", num_feats),
    ("cat", OneHotEncoder(drop="first", dtype=np.int8), cat_feats)
])

feature_names_en = []
for n in num_feats:
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

def train_group(dataframe):
    X_fe = ct.fit_transform(dataframe)
    X_num = dataframe[num_feats].values
    y     = dataframe["lnRINC"].values

    fe = LinearRegression().fit(X_fe, y)
    resid = y - fe.predict(X_fe)

    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
        objective="reg:squarederror", random_state=2025, tree_method="hist"
    ).fit(X_num, resid)

    shap_vals = shap.TreeExplainer(xgb).shap_values(X_num)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    return shap_vals, X_num, mean_abs

shap_high, X_high, mean_high = train_group(df[df["Group"]=="High"])
shap_low,  X_low,  mean_low  = train_group(df[df["Group"]=="Low"])


import matplotlib.pyplot as plt

top_vars = np.argsort(mean_high)[::-1][:10]
labels = [feature_names_en[i] for i in top_vars]
y_high = mean_high[top_vars]
y_low  = mean_low[top_vars]

x = np.arange(len(labels))
w = 0.35
plt.figure(figsize=(6,4))
plt.barh(x-w/2, y_high, height=w, label="High")
plt.barh(x+w/2, y_low,  height=w, label="Low")
plt.yticks(x, labels, fontsize=8)
plt.xlabel("Mean |SHAP|")
plt.title(f"Figure 6-13  Top-10 importance ({group_label} split)")
plt.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-13_shap_bar_high_low.png", dpi=300)
plt.close()

def overlay_plot(var_en, fname):
    idx = en2idx[var_en]
    # High
    x_h, y_h = X_high[:, idx], shap_high[:, idx]
    sm_h = lowess(y_h, x_h, frac=0.4, return_sorted=True)
    # Low
    x_l, y_l = X_low[:, idx], shap_low[:, idx]
    sm_l = lowess(y_l, x_l, frac=0.4, return_sorted=True)

    plt.figure(figsize=(5,4))
    plt.plot(sm_h[:,0], sm_h[:,1], label="High", color="tab:blue")
    plt.plot(sm_l[:,0], sm_l[:,1], label="Low",  color="tab:red")
    plt.xlabel(var_en)
    plt.ylabel(f"SHAP value for {var_en}")
    plt.title(f"Figure 6-14  {var_en} dependence by group" if var_en=="lnEDU"
              else f"Figure 6-15  {var_en} dependence by group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR/fname, dpi=300)
    plt.close()

overlay_plot("lnEDU", "figure6-14_lnEDU_overlay.png")
overlay_plot("lnHEA", "figure6-15_lnHEA_overlay.png")


rank_high = pd.Series(mean_high).rank(ascending=False).astype(int)
rank_low  = pd.Series(mean_low).rank(ascending=False).astype(int)

table64 = pd.DataFrame({
    "Variable": feature_names_en,
    "Rank_High": rank_high,
    "Rank_Low":  rank_low,
    "ΔRank": rank_high - rank_low,
    "Mean|SHAP|_High": mean_high,
    "Mean|SHAP|_Low":  mean_low,
})

table64_top = table64.sort_values("Rank_High").head(15)  
table64_top.to_excel(SAVE_DIR/"Table6-4_GroupComparison.xlsx", index=False)
table64_top.to_csv(SAVE_DIR/"Table6-4_GroupComparison.csv", index=False, encoding="utf-8-sig")

print("✅ 6.4 图表与表格已生成：", SAVE_DIR.resolve())
