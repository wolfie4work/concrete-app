import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Prédiction Résistance Béton", layout="wide")
st.title("🧱 Prédiction de la Résistance du Béton")
st.markdown("Application de *Machine Learning* pour estimer la résistance en compression du béton à partir de sa composition.")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url)
    df.columns = ["Ciment", "Laitier", "Cendres", "Eau", "Superplastifiant",
                  "Gros_Granulats", "Sable", "Age", "Resistance_MPa"]
    return df

@st.cache_resource
def train_models():
    df = load_data()
    X = df.drop("Resistance_MPa", axis=1)
    y = df["Resistance_MPa"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    return X, y, X_test, y_test, lr, rf, lr_pred, rf_pred

X, y, X_test, y_test, lr, rf, lr_pred, rf_pred = train_models()

# Layout: sidebar for input, main area for results
with st.sidebar:
    st.header("🧪 Composition du mélange")
    ciment = st.number_input("Ciment (kg/m³)", min_value=0, value=300)
    laitier = st.number_input("Laitier (kg/m³)", min_value=0, value=0)
    cendres = st.number_input("Cendres volantes (kg/m³)", min_value=0, value=0)
    eau = st.number_input("Eau (kg/m³)", min_value=0, value=150)
    superplast = st.number_input("Superplastifiant (kg/m³)", min_value=0.0, value=5.0, step=0.1)
    gros_agg = st.number_input("Gros granulats (kg/m³)", min_value=0, value=1000)
    sable = st.number_input("Sable (kg/m³)", min_value=0, value=700)
    age = st.number_input("Âge (jours)", min_value=1, value=28)
    predict_btn = st.button("🚀 Prédire la résistance")

tab1, tab2, tab3 = st.tabs(["📊 Performance des modèles", "⭐ Importance des variables", "🔮 Prédiction personnalisée"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Régression Linéaire")
        st.metric("R²", f"{r2_score(y_test, lr_pred):.3f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, lr_pred)):.2f} MPa")
    with col2:
        st.subheader("Random Forest")
        st.metric("R²", f"{r2_score(y_test, rf_pred):.3f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, rf_pred)):.2f} MPa")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_test, lr_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel("Valeurs réelles (MPa)")
    axes[0].set_ylabel("Prédictions (MPa)")
    axes[0].set_title("Régression Linéaire")

    axes[1].scatter(y_test, rf_pred, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1].set_xlabel("Valeurs réelles (MPa)")
    axes[1].set_ylabel("Prédictions (MPa)")
    axes[1].set_title("Random Forest")
    st.pyplot(fig)

with tab2:
    importances = rf.feature_importances_
    features = X.columns
    indices = np.argsort(importances)[::-1]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.barh(range(len(indices)), importances[indices][::-1], color="steelblue")
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([features[i] for i in indices[::-1]])
    ax2.set_xlabel("Importance")
    ax2.set_title("Importance des variables (Random Forest)")
    st.pyplot(fig2)

    st.subheader("Classement des variables")
    for i in range(X.shape[1]):
        st.write(f"{i+1}. **{features[indices[i]]}** — {importances[indices[i]]*100:.1f}%")

with tab3:
    st.subheader("Prédire un nouveau mélange")
    nouveau = pd.DataFrame([[ciment, laitier, cendres, eau, superplast, gros_agg, sable, age]],
                           columns=X.columns)
    pred_lr = lr.predict(nouveau)[0]
    pred_rf = rf.predict(nouveau)[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Régression Linéaire", f"{pred_lr:.2f} MPa")
    with col_b:
        st.metric("Random Forest (recommandé)", f"{pred_rf:.2f} MPa",
                  delta=f"{pred_rf - pred_lr:.2f}" if pred_rf != pred_lr else None)

    st.markdown("---")
    st.caption("Données : [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)")
