import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression

# Step 1: Load the CSV file
df = pd.read_csv(r'D:\DownloAD\3ae033f50fa345051652.csv')

# Step 2: Preprocess the data
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])  # Drop rows where date conversion failed
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Clean numeric columns (remove commas and convert to int)
numeric_columns = [
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total',
    'Non-Electric Vehicle Total',
    'Total Vehicles',
    'Percent Electric Vehicles'
]
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# Step 3: Create output directory for models
os.makedirs("county_models", exist_ok=True)

# Step 4: Train and save a model for each county
grouped = df.groupby('County')

for county_name, county_df in grouped:
    county_df = county_df.sort_values('Days')
    X = county_df[['Days']]
    y = county_df['Electric Vehicle (EV) Total']

    if len(X) > 1 and y.nunique() > 1:  # Require at least 2 data points and variation
        model = LinearRegression()
        model.fit(X, y)

        # Save the model
        filename = f"county_models/{county_name.replace(' ', '_')}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(model, f)

print("✅ All county models trained and saved in 'county_models/' folder.")
