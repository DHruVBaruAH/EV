import pandas as pd
import os
import re
import pickle
from sklearn.linear_model import LinearRegression
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from streamlit_option_menu import option_menu
import base64

# 🖼️ Set background image
def set_bg():
    encoded_url = "https://img.freepik.com/free-photo/assembly-line-production-new-car-automated-welding-car-body-production-line-is-working_645730-580.jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{encoded_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        .block-container {{
            backdrop-filter: blur(6px);
            background-color: rgba(0, 0, 0, 0.4);
            padding: 2rem;
            border-radius: 10px;
        }}
        h1, h2, h3 {{
            color: #ffffff !important;
        }}
        footer {{
            position: fixed;
            bottom: 0;
            width: 100%;
            color: white;
            text-align: center;
            padding: 10px;
        }}
        </style>
        <footer>Built with ❤️ by Dhruv | 2025</footer>
        """,
        unsafe_allow_html=True,
    )


set_bg()

# Page config
st.set_page_config(page_title="EV Forecasting Dashboard", layout="wide")

# Navbar using option_menu
selected = option_menu(
    menu_title=None,
    options=["Home", "About", "Contact"],
    icons=["house", "info-circle", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "Home":
    st.title("⚡ Electric Vehicle Forecasting (County-wise)")

    # Load data
    csv_path = "3ae033f50fa345051652 (1).csv"
    df = pd.read_csv(csv_path)

    # Clean numeric columns
    columns_to_clean = [
        'Battery Electric Vehicles (BEVs)',
        'Plug-In Hybrid Electric Vehicles (PHEVs)',
        'Electric Vehicle (EV) Total',
        'Non-Electric Vehicle Total',
        'Total Vehicles',
        'Percent Electric Vehicles'
    ]

    for col in columns_to_clean:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # Preprocess date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    # Model storage
    os.makedirs("county_models", exist_ok=True)

    # Train and save model
    counties = df['County'].unique()
    for county in counties:
        county_df = df[df['County'] == county]
        if len(county_df) < 2:
            continue
        X = county_df[['Days']]
        y = county_df['Electric Vehicle (EV) Total']
        model = LinearRegression()
        model.fit(X, y)
        safe_county = re.sub(r'[^a-zA-Z0-9_]', '_', county.strip())
        with open(f'county_models/{safe_county}.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Sidebar filter
    st.sidebar.header("🔎 Filter Data")
    selected_county = st.sidebar.selectbox("Select County", sorted(df['County'].dropna().unique()))
    selected_year = st.sidebar.selectbox("Select Year", sorted(df['Year'].unique()))

    # Filter data
    county_df = df[(df['County'] == selected_county) & (df['Year'] == selected_year)]
    model_file = f'county_models/{re.sub(r"[^a-zA-Z0-9_]", "_", selected_county)}.pkl'

    # Forecast till 2030
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        last_date = df['Date'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=30), end="2030-12-01", freq='MS')
        forecast_days = (forecast_dates - df['Date'].min()).days.values.reshape(-1, 1)
        forecast_values = model.predict(forecast_days)

        # Plotting
        plt.figure(figsize=(12, 6))
        sns.set(style="whitegrid")
        actual = county_df.sort_values('Date')

        plt.plot(actual['Date'], actual['Electric Vehicle (EV) Total'], marker='o', label='Actual EV Count')
        plt.plot(forecast_dates, forecast_values, linestyle='--', color='orange', label='Forecast (to 2030)')

        plt.title(f"EV Forecast for {selected_county} ({selected_year} + Future)", fontsize=18)
        plt.xlabel("Date")
        plt.ylabel("EV Count")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        st.pyplot(plt)
    else:
        st.warning("Model not found for the selected county.")

elif selected == "About":
    st.header("📘 About This Project")
    st.write("""
    This dashboard provides a futuristic overview of electric vehicle (EV) adoption across various counties. 
    By leveraging historical data and machine learning models, it forecasts the expected growth of EVs up to the year 2030.

    **Key Features:**
    - Interactive county-wise selection
    - Forecast graphs until 2030
    - Stylish and responsive design

    **Tech Stack:**
    - Streamlit
    - Pandas, Scikit-Learn
    - Matplotlib, Seaborn

    This project aims to assist government bodies, manufacturers, and researchers in understanding trends in EV adoption.
    """)

