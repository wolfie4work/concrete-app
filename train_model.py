import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
df = pd.read_excel(url)

df.columns = ["Ciment", "Laitier", "Cendres", "Eau", "Superplastifiant", 
              "Gros_Granulats", "Sable", "Age", "Resistance_MPa"]

X = df.drop("Resistance_MPa", axis=1)
y = df["Resistance_MPa"]

print("Training Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

print("Saving model to model_beton.pkl...")
joblib.dump(rf_model, 'model_beton.pkl')
print("Done! You can now run your Streamlit app.")