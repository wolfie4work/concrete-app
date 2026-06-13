import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Prédiction Béton", page_icon="🏗️", layout="wide")

# 2. Load Model (Cached so it doesn't reload on every slider move)
@st.cache_resource
def load_model():
    return joblib.load('model_beton.pkl')

model = load_model()

# 3. Header
st.title("🏗️ Prédiction de la Résistance du Béton")
st.markdown("Ajustez les composants du mélange pour estimer la résistance à la compression (en MPa) à l'aide d'un modèle d'Intelligence Artificielle (Random Forest).")

# 4. Input Layout (Using columns for a clean UI)
st.sidebar.header("Paramètres du Mélange (kg/m³)")

ciment = st.sidebar.slider("Ciment", min_value=100.0, max_value=540.0, value=300.0, step=5.0)
laitier = st.sidebar.slider("Laitier de haut fourneau", min_value=0.0, max_value=360.0, value=0.0, step=5.0)
cendres = st.sidebar.slider("Cendres volantes", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
eau = st.sidebar.slider("Eau", min_value=120.0, max_value=250.0, value=150.0, step=5.0)
superplastifiant = st.sidebar.slider("Superplastifiant", min_value=0.0, max_value=32.0, value=5.0, step=0.5)
gros_granulats = st.sidebar.slider("Gros Granulats", min_value=800.0, max_value=1145.0, value=1000.0, step=10.0)
sable = st.sidebar.slider("Sable", min_value=590.0, max_value=990.0, value=700.0, step=10.0)
age = st.sidebar.slider("Âge (Jours)", min_value=1, max_value=365, value=28, step=1)

# 5. Format inputs for the model
input_data = pd.DataFrame({
    "Ciment": [ciment],
    "Laitier": [laitier],
    "Cendres": [cendres],
    "Eau": [eau],
    "Superplastifiant": [superplastifiant],
    "Gros_Granulats": [gros_granulats],
    "Sable": [sable],
    "Age": [age]
})

# 6. Prediction and Results Display
prediction = model.predict(input_data)[0]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Résultat")
    # Display the result in a large, visually appealing metric box
    st.metric(label="Résistance à la compression prédite", value=f"{prediction:.2f} MPa")

with col2:
    st.subheader("Composition Actuelle")
    st.dataframe(input_data, hide_index=True)

st.divider()
st.caption("Modèle entraîné sur le dataset 'Concrete Compressive Strength' de l'UC Irvine.")