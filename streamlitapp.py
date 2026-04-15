
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

# Dynamically get the exact feature names from the loaded preprocessed data
# This ensures consistency with the model's training features
# Recreate X from df_final to get the exact feature names used for training
X_temp_for_features = df_final.drop(['cnt', 'casual', 'registered'], axis=1)
model_features = X_temp_for_features.columns.tolist()
del X_temp_for_features # Clean up temporary variable

for feature in model_features:
    if feature not in df_final.columns:
        df_final[feature] = 0

st.sidebar.title("Bike Rental Dashboard")
page = st.sidebar.radio("Navigation", ["Insights & Model Validation", "Interactive Prediction"])

if page == "Insights & Model Validation":
    st.title("🚴\u200D♀️ Model Insights & Performance Validation")

    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Model Type", "Gradient Boosting")
    with col2: st.metric("Best R2 Score", "0.9506")
    with col3: st.metric("Avg Error (MAE)", "22.63")
    with col4: st.metric("RMSE", "37.46")
    with col5: st.metric("Avg Hourly Rentals", f"{df_final['cnt'].mean():.0f}")

    tab1, tab2, tab3 = st.tabs(["Demand Patterns", "Environmental Impact", "Model Validation"])
    
    with tab1:
        st.subheader("Hourly Bike Rental Demand: Working Day vs No Work Day")
        hourly_demand = df_final.groupby(['hr', 'workingday_Working Day'])['cnt'].mean().reset_index()
        hourly_demand['Day Type'] = hourly_demand['workingday_Working Day'].apply(lambda x: 'Working Day' if x == 1 else 'No work')
        fig1 = px.line(hourly_demand, x='hr', y='cnt', color='Day Type', title='Hourly Demand Patterns', color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Demand by Month")
        monthly_demand = df_final.groupby('mnth')['cnt'].mean().reset_index()
        fig_month = px.bar(monthly_demand, x='mnth', y='cnt', title='Average Bike Rentals by Month', color='cnt', color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig_month, use_container_width=True)

        st.subheader("Demand by Weekday")
        weekday_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
        df_final['weekday_name'] = df_final['weekday'].map(weekday_map)
        weekday_demand = df_final.groupby('weekday_name')['cnt'].mean().reindex(weekday_map.values()).reset_index()
        fig_weekday = px.bar(weekday_demand, x='weekday_name', y='cnt', title='Average Bike Rentals by Weekday', color='cnt', color_continuous_scale=px.colors.sequential.Sunset)
        st.plotly_chart(fig_weekday, use_container_width=True)

    with tab2:
        st.subheader("Impact of Temperature on Bike Rentals")
        fig_temp = px.scatter(df_final, x='temp', y='cnt', opacity=0.3, color_discrete_sequence=['coral'], title='Impact of Temperature on Bike Rentals')
        st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("Demand Distribution across Weather Conditions")
        df_plot_weather = df_final.copy()
        # Initialize weathersit_original with 'Clear' as it's the implied base category
        df_plot_weather['weathersit_original'] = 'Clear'

        # Conditionally overwrite for other weather conditions if their respective OHE columns are 1
        if 'weathersit_Mist' in df_plot_weather.columns:
            df_plot_weather.loc[df_plot_weather['weathersit_Mist'] == 1, 'weathersit_original'] = 'Mist'
        if 'weathersit_Light Rain/Snow' in df_plot_weather.columns:
            df_plot_weather.loc[df_plot_weather['weathersit_Light Rain/Snow'] == 1, 'weathersit_original'] = 'Light Rain/Snow'
        if 'weathersit_Moderate Rain/Snow' in df_plot_weather.columns:
            df_plot_weather.loc[df_plot_weather['weathersit_Moderate Rain/Snow'] == 1, 'weathersit_original'] = 'Moderate Rain/Snow'
        
        fig_weather = px.box(df_plot_weather, x='weathersit_original', y='cnt', title='Demand Distribution across Weather Conditions', color='weathersit_original', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_weather, use_container_width=True)

    with tab3:
        st.subheader("Model Performance Overview")
        st.write("The Gradient Boosting Regressor was chosen as the optimal model for bike rental demand prediction based on its superior performance across key metrics. It achieved an R-squared score of 0.9506, indicating that it explains approximately 95% of the variance in bike rental demand. Its low Mean Absolute Error (22.63) and Root Mean Squared Error (37.46) demonstrate high accuracy in predicting rental counts.")
        st.write("Hyperparameter tuning was performed, and while some models showed slight improvements, the default parameters of the ensemble methods proved highly effective, particularly for Random Forest. The Gradient Boosting model, after tuning, provided the best balance of accuracy and robustness.")

        st.subheader("Feature Importance (What drives the model?)")
        # Based on training results, replace with actual if available from loaded model
        feat_imp = pd.DataFrame({
            'Feature': ['Hour', 'Rush Hour', 'Year', 'Temp', 'atemp', 'Humidity', 'Working Day', 'Month', 'Season'],
            'Importance': [0.485, 0.137, 0.086, 0.083, 0.044, 0.036, 0.043, 0.018, 0.018] # Example values from the notebook's best GB model
        }).sort_values('Importance', ascending=True)
        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title='Feature Importance for Gradient Boosting Model', color='Importance', color_continuous_scale=px.colors.sequential.Magma)
        st.plotly_chart(fig_imp, use_container_width=True)

elif page == "Interactive Prediction":
    st.title("🧠 Demand Prediction Engine")
    st.write("The year is preset to 2012 for the most current context.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
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
    
    input_data = {
        'yr': 2012.0, 'mnth': float(mnth), 'hr': int(hr), 'weekday': int(wk), 
        'temp': float(te), 'atemp': float(at), 'hum': float(hu), 'windspeed': float(wi), 
        'is_rush_hour': int(irh)
    }
    
    input_df = pd.DataFrame([input_data])
    for f in model_features: 
        if f not in input_df.columns: input_df[f] = 0
    
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

    # FINAL ALIGNMENT: Match order AND force numeric type (converts Booleans to 1.0/0.0)
    input_df = input_df.reindex(columns=model_features).fillna(0).astype(float)

    if st.button("Run Prediction"):
        res = model.predict(input_df)[0]
        st.success(f"Predicted Hourly Rental Demand: {int(res)} units")
