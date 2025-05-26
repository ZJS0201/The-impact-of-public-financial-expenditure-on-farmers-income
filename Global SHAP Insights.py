
from pathlib import Path
DATA_PATH = Path("Research_data.xlsx")          
SAVE_DIR  = Path("results/6_2_figures")      
SAVE_DIR.mkdir(parents=True, exist_ok=True)


import pandas as pd, numpy as np, matplotlib.pyplot as plt, shap
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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


train_mask = df["年份"] <= 2021
test_mask  = df["年份"] >= 2022
Xfe_train, Xfe_test = X_fe[train_mask], X_fe[test_mask]
Xn_train, Xn_test   = X_num[train_mask], X_num[test_mask]
y_train, y_test     = y[train_mask], y[test_mask]


fe = LinearRegression().fit(Xfe_train, y_train)
yhat_fe_train = fe.predict(Xfe_train)
yhat_fe_test  = fe.predict(Xfe_test)
resid_train   = y_train - yhat_fe_train
resid_test    = y_test  - yhat_fe_test


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
shap_values = explainer.shap_values(Xn_test)


feature_names_cn = num_feats
feature_names_en = []
for name in feature_names_cn:
    if name.startswith("ln_"):
        cn = name[3:]  
        feature_names_en.append(map_cn2en.get(cn, name))  
    elif name == "year_idx":
        feature_names_en.append("YEAR")
    elif name == "城镇化率（%）":
        feature_names_en.append("URB")
    elif name == "二三产业产值占比（%）":
        feature_names_en.append("IND")
    else:
        feature_names_en.append(name)


en2idx = {en: i for i, en in enumerate(feature_names_en)}


mean_abs = np.abs(shap_values).mean(axis=0)
total = mean_abs.sum()
top_idx = np.argsort(mean_abs)[::-1][:10]
table_top10 = pd.DataFrame({
    "Rank": range(1, 11),
    "Variable": [feature_names_en[i] for i in top_idx],
    "Mean |SHAP|": mean_abs[top_idx],
    "Share (%)": 100 * mean_abs[top_idx] / total
})
table_top10.to_excel(SAVE_DIR/"Table6-2_SHAP_Top10.xlsx", index=False)


plt.figure()
shap.summary_plot(shap_values, Xn_test, feature_names=feature_names_en, show=False)
plt.title("Figure 6-4  SHAP summary (FE + XGB residual)")
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-4_shap_summary.png", dpi=300)
plt.close()


plt.figure(figsize=(6,4))
vals = mean_abs[top_idx][::-1]
labels = [feature_names_en[i] for i in top_idx][::-1]
plt.barh(range(10), vals)
plt.yticks(range(10), labels, fontsize=8)
plt.xlabel("Mean |SHAP|")
plt.title("Figure 6-5  Top‑10 feature importance")
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-5_shap_bar.png", dpi=300)
plt.close()


inter = explainer.shap_interaction_values(Xn_test)
plt.figure()
shap.summary_plot(inter, Xn_test, feature_names=feature_names_en,
                  plot_type="bar", max_display=15, show=False)
plt.title("Figure 6-6  SHAP interaction summary")
plt.tight_layout()
plt.savefig(SAVE_DIR/"figure6-6_shap_interaction.png", dpi=300)
plt.close()


for var_en, fname in [("lnEDU", "figure6-7_dep_lnEDU.png"),
                      ("lnHEA", "figure6-8_dep_lnHEA.png")]:
    idx = en2idx[var_en]
    shap.dependence_plot(idx, shap_values, Xn_test,
                         feature_names=feature_names_en, show=False)
    plt.title(f"Dependence plot of {var_en}")
    plt.tight_layout()
    plt.savefig(SAVE_DIR/fname, dpi=300)
    plt.close()


