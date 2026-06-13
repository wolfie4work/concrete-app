import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import requests
import io
import xlrd

print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
response = requests.get(url)

workbook = xlrd.open_workbook(file_contents=response.content)
sheet = workbook.sheet_by_index(0)

data = []
for row_idx in range(sheet.nrows):
    row = sheet.row_values(row_idx)
    data.append(row)

df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype(float)

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