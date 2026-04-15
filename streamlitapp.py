
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Bike Rental Demand Prediction")

# --- Helper Function to Load Data/Model ---
@st.cache_data
def load_resources():
    try:
        df = pd.read_csv('df_final.csv')
        model = pickle.load(open('best_gradient_boosting_model.pkl', 'rb'))
        return df, model
    except FileNotFoundError:
        st.error("Required data files not found.")
        st.stop()

df_final, model = load_resources()

model_features = ['yr', 'mnth', 'hr', 'weekday', 'temp', 'atemp', 'hum', 'windspeed', 'is_rush_hour',                  'season_springer', 'season_summer', 'season_winter',                  'weathersit_Light Rain/Snow', 'weathersit_Mist',                  'weathersit_Moderate Rain/Snow', 'workingday_Working Day',                  'holiday_Yes', 'temp_type_Cold', 'temp_type_Hot']

for feature in model_features:
    if feature not in df_final.columns:
        df_final[feature] = 0

st.sidebar.title("Bike Rental Dashboard")
page = st.sidebar.radio("Navigation", ["Insights & Model Performance", "Interactive Prediction"])

if page == "Insights & Model Performance":
    st.title("🚴‍♀️ Bike Sharing Demand Prediction: Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Records", f"{len(df_final):,}")
    with col2: st.metric("Avg Rentals", f"{df_final['cnt'].mean():.2f}")
    with col3: st.metric("Max Demand", f"{df_final['cnt'].max():,}")

    st.subheader("Hourly Demand by Day Type")
    hourly_demand = df_final.groupby(['hr', 'workingday_Working Day'])['cnt'].mean().reset_index()
    hourly_demand['Day Type'] = hourly_demand['workingday_Working Day'].apply(lambda x: 'Working Day' if x == 1 else 'No work')
    fig1 = px.line(hourly_demand, x='hr', y='cnt', color='Day Type', title='Hourly Demand')
    st.plotly_chart(fig1, use_container_width=True)

    df_final['season_category'] = 'fall'
    if 'season_springer' in df_final.columns: df_final.loc[df_final['season_springer'] == 1, 'season_category'] = 'springer'
    if 'season_summer' in df_final.columns: df_final.loc[df_final['season_summer'] == 1, 'season_category'] = 'summer'
    if 'season_winter' in df_final.columns: df_final.loc[df_final['season_winter'] == 1, 'season_category'] = 'winter'

    st.subheader("Average Demand by Season")
    seasonal_demand = df_final.groupby('season_category')['cnt'].mean().reset_index()
    fig2 = px.bar(seasonal_demand, x='season_category', y='cnt', color='season_category', color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig2, use_container_width=True)

    df_final['weathersit_category'] = 'Clear'
    if 'weathersit_Mist' in df_final.columns: df_final.loc[df_final['weathersit_Mist'] == 1, 'weathersit_category'] = 'Mist'
    if 'weathersit_Light Rain/Snow' in df_final.columns: df_final.loc[df_final['weathersit_Light Rain/Snow'] == 1, 'weathersit_category'] = 'Light Rain/Snow'
    if 'weathersit_Moderate Rain/Snow' in df_final.columns: df_final.loc[df_final['weathersit_Moderate Rain/Snow'] == 1, 'weathersit_category'] = 'Moderate Rain/Snow'

    st.subheader("Demand Distribution by Weather")
    fig3 = px.box(df_final, x='weathersit_category', y='cnt', color='weathersit_category', color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Impact of Temperature")
    fig4 = px.scatter(df_final, x='temp', y='cnt', opacity=0.3)
    st.plotly_chart(fig4, use_container_width=True)

elif page == "Interactive Prediction":
    st.title("🧠 Interactive Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        yr = st.slider("Year", 2011, 2012, 2012)
        mnth = st.slider("Month", 1, 12, 7)
        hr = st.slider("Hour", 0, 23, 17)
        wk = st.slider("Weekday", 0, 6, 1)
    with col2:
        te = st.slider("Temp", 0.0, 1.0, 0.6)
        at = st.slider("ATemp", 0.0, 1.0, 0.55)
        hu = st.slider("Hum", 0.0, 1.0, 0.6)
        wi = st.slider("Wind", 0.0, 1.0, 0.2)
    with col3:
        se = st.selectbox("Season", ['springer', 'summer', 'fall', 'winter'], index=2)
        we = st.selectbox("Weather", ['Clear', 'Mist', 'Light Rain/Snow', 'Moderate Rain/Snow'], index=0)
        wd = st.selectbox("Working Day?", ['Working Day', 'No work'], index=0)
        ho = st.selectbox("Holiday?", ['No', 'Yes'], index=0)
        tt = st.selectbox("Temp Type", ['Cold', 'Mild', 'Hot'], index=1)

    irh = 1 if wd == 'Working Day' and ((7 <= hr <= 9) or (16 <= hr <= 19)) else 0
    
    # Create input DataFrame and ensure column order matches model expectations exactly
    input_data_dict = {'yr':yr,'mnth':mnth,'hr':hr,'weekday':wk,'temp':te,'atemp':at,'hum':hu,'windspeed':wi,'is_rush_hour':irh}
    input_df = pd.DataFrame(columns=model_features)
    input_df.loc[0] = 0
    input_df.update(pd.Series(input_data_dict))
    
    if se == 'springer': input_df['season_springer'] = 1
    elif se == 'summer': input_df['season_summer'] = 1
    elif se == 'winter': input_df['season_winter'] = 1
    if we == 'Mist': input_df['weathersit_Mist'] = 1
    elif we == 'Light Rain/Snow': input_df['weathersit_Light Rain/Snow'] = 1
    elif we == 'Moderate Rain/Snow': input_df['weathersit_Moderate Rain/Snow'] = 1
    if wd == 'Working Day': input_df['workingday_Working Day'] = 1
    if ho == 'Yes': input_df['holiday_Yes'] = 1
    if tt == 'Cold': input_df['temp_type_Cold'] = 1
    elif tt == 'Hot': input_df['temp_type_Hot'] = 1

    # Reorder columns to match the training data feature order exactly
    input_df = input_df[model_features]

    if st.button("Predict"):
        res = model.predict(input_df)[0]
        st.success(f"Predicted Demand: {int(res)}")
