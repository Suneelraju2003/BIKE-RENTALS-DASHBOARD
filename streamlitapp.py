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

# Dynamically get the exact feature names
X_temp_for_features = df_final.drop(['cnt', 'casual', 'registered'], axis=1)
model_features = X_temp_for_features.columns.tolist()
del X_temp_for_features

for feature in model_features:
    if feature not in df_final.columns:
        df_final[feature] = 0

st.sidebar.title("Bike Rental Dashboard")
page = st.sidebar.radio("Navigation", ["Insights & Model Validation", "Interactive Prediction"])

if page == "Insights & Model Validation":
    st.title("🚴‍♀️ Model Insights & Performance Validation")

    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Model Type", "Gradient Boosting")
    with col2: st.metric("Best R2 Score", "0.9506")
    with col3: st.metric("Avg Error (MAE)", "22.63")
    with col4: st.metric("RMSE", "37.46")
    with col5: st.metric("Avg Hourly Rentals", f"{df_final['cnt'].mean():.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Demand Patterns", "Environmental Impact", "Model Validation", "Correlation Analysis"])

    with tab1:
        st.subheader("Hourly Bike Rental Demand: Working Day vs No Work Day")
        hourly_demand = df_final.groupby(['hr', 'workingday_Working Day'])['cnt'].mean().reset_index()
        hourly_demand['Day Type'] = hourly_demand['workingday_Working Day'].apply(lambda x: 'Working Day' if x == 1 else 'No work')
        fig1 = px.line(hourly_demand, x='hr', y='cnt', color='Day Type', title='Hourly Demand Patterns', color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("Demand Distribution across Weather Conditions")
        df_plot_weather = df_final.copy()
        df_plot_weather['weathersit_category'] = 'Clear'
        if 'weathersit_Mist' in df_plot_weather.columns: df_plot_weather.loc[df_plot_weather['weathersit_Mist'] == 1, 'weathersit_category'] = 'Mist'
        if 'weathersit_Light Snow' in df_plot_weather.columns: df_plot_weather.loc[df_plot_weather['weathersit_Light Snow'] == 1, 'weathersit_category'] = 'Light Snow'
        fig_weather = px.box(df_plot_weather, x='weathersit_category', y='cnt', title='Weather Impact', color='weathersit_category')
        st.plotly_chart(fig_weather, use_container_width=True)

    with tab3:
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame({
            'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
            'MAE': [28.38, 23.13, 22.63],
            'RMSE': [46.52, 39.19, 37.46],
            'R2_Score': [0.9239, 0.9460, 0.9506]
        }).set_index('Model')
        st.dataframe(comparison_df)

    with tab4:
        st.subheader("Feature Correlation Heatmap")

        # ✅ FIXED PART (ONLY CHANGE)
        df_corr_data = df_final.select_dtypes(include=[np.number]).copy()
        df_corr_data = df_corr_data.apply(pd.to_numeric, errors='coerce')
        df_corr_data = df_corr_data.dropna(axis=1, how='all')

        corr_matrix = df_corr_data.corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r'
            )
        )

        st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Interactive Prediction":
    st.title("🧠 Demand Prediction Engine")
    col1, col2, col3 = st.columns(3)
    with col1:
        mnth = st.slider("Month", 1, 12, 7)
        hr = st.slider("Hour", 0, 23, 17)
    with col2:
        te = st.slider("Temp", 0.0, 1.0, 0.6)
        hu = st.slider("Hum", 0.0, 1.0, 0.6)
    with col3:
        se = st.selectbox("Season", ['springer', 'summer', 'fall', 'winter'])
        we = st.selectbox("Weather", ['Clear', 'Mist', 'Light Snow'])
    
    if st.button("Run Prediction"):
        input_data = pd.DataFrame([{'yr': 2012, 'mnth': mnth, 'hr': hr, 'temp': te, 'hum': hu}])
        for f in model_features: 
            if f not in input_data.columns: input_data[f] = 0
        input_data = input_data.reindex(columns=model_features).fillna(0).astype(float)
        res = model.predict(input_data)[0]
        st.success(f"Predicted Hourly Rental Demand: {int(res)} units")
