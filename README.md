# Prédiction de la Résistance du Béton

Application Streamlit pour prédire la résistance en compression du béton à partir de sa composition, en utilisant la Régression Linéaire et Random Forest.

## 🚀 Déploiement

### Option 1 — Streamlit Community Cloud (recommandé)
1. Pushez ce repo sur GitHub
2. Allez sur https://streamlit.io/cloud
3. Connectez votre compte GitHub
4. Cliquez sur **New app** → sélectionnez ce dépôt → branche `main` → fichier `main.py`
5. Déployez !

### Option 2 — Hugging Face Spaces
1. Allez sur https://huggingface.co/new-space
2. Donnez un nom, choisissez SDK → **Streamlit**
3. Pushez le code sur le Space
4. L'application sera automatiquement déployée

### Option 3 — Docker
```bash
docker build -t beton-prediction .
docker run -p 8501:8501 beton-prediction
```

## 📦 Structure
- `main.py` — application Streamlit
- `requirements.txt` — dépendances Python
- `Dockerfile` — image Docker
- `.streamlit/config.toml` — thème Streamlit

## 📊 Données
[UCI Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
# concrete-app
