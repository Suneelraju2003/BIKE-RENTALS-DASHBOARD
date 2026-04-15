
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
        st.error("Required data files (df_final.csv or best_gradient_boosting_model.pkl) not found. Please ensure they are in the same directory as this script.")
        st.stop()

df_final, model = load_resources()

# Features used by the model (excluding target and derived targets like casual/registered)
# This list must match the columns and their order used during model training (X_train.columns)
model_features = ['yr', 'mnth', 'hr', 'weekday', 'temp', 'atemp', 'hum', 'windspeed', 'is_rush_hour',
                  'season_springer', 'season_summer', 'season_winter',
                  'weathersit_Light Rain/Snow', 'weathersit_Mist',
                  'weathersit_Moderate Rain/Snow', 'workingday_Working Day',
                  'holiday_Yes', 'temp_type_Cold', 'temp_type_Hot']

# --- Streamlit UI Layout ---
st.sidebar.title("Bike Rental Dashboard")
page = st.sidebar.radio("Navigation", ["Insights & Model Performance", "Interactive Prediction"])

if page == "Insights & Model Performance":
    st.title("🚴‍♀️ Bike Sharing Demand Prediction: Insights & Model Performance")
    st.write("This dashboard provides insights into bike rental demand and evaluates the performance of different machine learning models.")

    st.markdown("""---
    <h2 style='text-align: center; color: #4CAF50;'>Key Performance Indicators</h2>""", unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Records", value=f"{len(df_final):,}", delta_color="off")
    with col2:
        st.metric(label="Average Hourly Rentals", value=f"{df_final['cnt'].mean():.2f}", delta_color="off")
    with col3:
        st.metric(label="Highest Hourly Demand", value=f"{df_final['cnt'].max():,}", delta_color="off")

    st.markdown("""---
    <h2 style='text-align: center; color: #4CAF50;'>Demand Patterns Visualizations</h2>""", unsafe_allow_html=True)
    st.write("Explore how various factors influence bike rental demand.")

    # Plot 1: Hourly Demand: Working Day vs Weekend
    st.subheader("Hourly Bike Rental Demand: Working Day vs No Work Day")
    hourly_demand = df_final.groupby(['hr', 'workingday'])['cnt'].mean().reset_index()
    fig1 = px.line(hourly_demand, x='hr', y='cnt', color='workingday',
                   labels={'hr': 'Hour of the Day', 'cnt': 'Average Rental Count'},
                   title='Average Hourly Bike Rentals by Day Type',
                   color_discrete_map={'Working Day': '#1f77b4', 'No work': '#ff7f0e'})
    fig1.update_layout(hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Average Demand by Season
    st.subheader("Average Bike Rentals per Season")
    seasonal_demand = df_final.groupby('season')['cnt'].mean().reset_index()
    fig2 = px.bar(seasonal_demand, x='season', y='cnt',
                   labels={'season': 'Season', 'cnt': 'Average Rental Count'},
                   title='Average Bike Rentals per Season',
                   color='season', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Demand Distribution by Weather Situation
    st.subheader("Demand Distribution across Weather Conditions")
    fig3 = px.box(df_final, x='weathersit', y='cnt',
                  labels={'weathersit': 'Weather Situation', 'cnt': 'Rental Count'},
                  title='Demand Distribution across Weather Conditions',
                  color='weathersit', color_discrete_sequence=px.colors.qualitative.Dark)
    st.plotly_chart(fig3, use_container_width=True)

    # Plot 4: Correlation: Normalized Temperature vs Total Rentals
    st.subheader("Impact of Temperature on Bike Rentals")
    fig4 = px.scatter(df_final, x='temp', y='cnt',
                     labels={'temp': 'Normalized Temperature', 'cnt': 'Total Rentals'},
                     title='Impact of Temperature on Bike Rentals',
                     opacity=0.3, color_discrete_sequence=['#ef553b'])
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""---
    <h2 style='text-align: center; color: #4CAF50;'>Model Performance & Comparison</h2>""", unsafe_allow_html=True)
    st.write("A comparative analysis of Decision Tree, Random Forest, and Gradient Boosting models.")

    # Hardcoded comparison data from the notebook's final comparison table
    comparison_df_display = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'MAE':   [28.38, 23.13, 22.63],
        'RMSE':  [46.52, 39.19, 37.46],
        'R2_Score': [0.9239, 0.9460, 0.9506],
    }).set_index('Model')
    st.dataframe(comparison_df_display.style.highlight_max(axis=0, subset=['R2_Score'], color='lightgreen')                                          .highlight_min(axis=0, subset=['MAE', 'RMSE'], color='lightcoral'))

    # Plot Comparison Visualizations (Bar Charts)
    st.subheader("All Models — Key Performance Comparison")
    df_plot_comp = comparison_df_display.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')

    fig_comp = go.Figure()
    metrics = ['MAE', 'RMSE', 'R2_Score']
    colors = px.colors.qualitative.D3

    for i, metric in enumerate(metrics):
        bar_data = df_plot_comp[df_plot_comp['Metric'] == metric]
        fig_comp.add_trace(go.Bar(name=metric, x=bar_data['Model'], y=bar_data['Value'],
                                  text=[f'{v:.2f}' if metric != 'R2_Score' else f'{v:.4f}' for v in bar_data['Value']],
                                  textposition='outside',
                                  marker_color=colors[i]))

    fig_comp.update_layout(barmode='group', title='Model Performance: MAE, RMSE, and R2 Score',
                           yaxis_title='Metric Value', xaxis_title='Model', legend_title="Metric")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""---
    <h2 style='text-align: center; color: #4CAF50;'>Feature Importance for Best Model (Gradient Boosting)</h2>""", unsafe_allow_html=True)
    st.write("Top features driving the Gradient Boosting model's predictions.")

    # Hardcoded feature importance from the notebook's gb_feature_importance_df
    gb_feature_importance_data = {
        'Feature': ['hr', 'is_rush_hour', 'yr', 'temp', 'atemp', 'workingday_Working Day', 'hum', 'season_springer', 'mnth', 'weathersit_Light Snow'],
        'Importance': [0.485564, 0.137633, 0.086657, 0.083901, 0.044885, 0.043393, 0.036166, 0.018740, 0.018089, 0.014992]
    }
    gb_feature_importance_df = pd.DataFrame(gb_feature_importance_data)

    fig_fi = px.bar(gb_feature_importance_df, x='Importance', y='Feature', orientation='h',
                    title='Top 10 Features: Gradient Boosting (Optimized)',
                    labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                    color='Importance', color_continuous_scale=px.colors.sequential.Sunset)
    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_fi, use_container_width=True)

elif page == "Interactive Prediction":
    st.title("🧠 Interactive Bike Rental Demand Prediction")
    st.write("Adjust the parameters below to predict bike rental demand. The model uses the most important features to make predictions.")

    st.markdown("""---
    <h2 style='text-align: center; color: #4CAF50;'>Input Features</h2>""", unsafe_allow_html=True)

    # Input widgets for features
    col1, col2, col3 = st.columns(3)

    with col1:
        yr = st.slider("Year", min_value=2011, max_value=2012, value=2012, step=1, help="Year of rental (2011 or 2012)")
        mnth = st.slider("Month", min_value=1, max_value=12, value=7, step=1, help="Month of the year")
        hr = st.slider("Hour of Day", min_value=0, max_value=23, value=17, step=1, help="Hour of the day (0-23)")
        weekday = st.slider("Weekday", min_value=0, max_value=6, value=1, step=1, help="Day of the week (0=Sunday, 6=Saturday)")

    with col2:
        temp = st.slider("Normalized Temperature", min_value=0.0, max_value=1.0, value=0.6, step=0.01, help="Normalized temperature in Celsius (0-1 range)")
        atemp = st.slider("Normalized Feeling Temperature", min_value=0.0, max_value=1.0, value=0.55, step=0.01, help="Normalized 'feeling' temperature in Celsius (0-1 range)")
        hum = st.slider("Normalized Humidity", min_value=0.0, max_value=1.0, value=0.6, step=0.01, help="Normalized humidity (0-1 range)")
        windspeed = st.slider("Normalized Windspeed", min_value=0.0, max_value=1.0, value=0.2, step=0.01, help="Normalized windspeed (0-1 range)")

    with col3:
        season = st.selectbox("Season", ['springer', 'summer', 'fall', 'winter'], index=2, help="Season of the year") # 'fall' is the base in drop_first=True
        weathersit = st.selectbox("Weather Situation", ['Clear', 'Mist', 'Light Rain/Snow', 'Moderate Rain/Snow'], index=0, help="Weather condition") # 'Clear' is the base
        workingday = st.selectbox("Working Day?", ['Working Day', 'No work'], index=0, help="Is it a working day or a weekend/holiday?") # 'No work' is the base
        holiday = st.selectbox("Holiday?", ['No', 'Yes'], index=0, help="Is it a public holiday?") # 'No' is the base
        temp_type = st.selectbox("Temperature Type", ['Cold', 'Mild', 'Hot'], index=1, help="Categorized temperature (Cold, Mild, Hot)") # 'Mild' is the base

    # Feature Engineering 'is_rush_hour' based on inputs
    is_rush_hour = 0
    if workingday == 'Working Day':
        if (7 <= hr <= 9) or (16 <= hr <= 19):
            is_rush_hour = 1

    # Create input DataFrame for prediction, matching model_features order and one-hot encoding
    input_df = pd.DataFrame(columns=model_features) # Initialize with all expected columns
    input_df.loc[0] = 0 # Fill with zeros initially

    # Populate numerical features
    input_df['yr'] = yr
    input_df['mnth'] = mnth
    input_df['hr'] = hr
    input_df['weekday'] = weekday
    input_df['temp'] = temp
    input_df['atemp'] = atemp
    input_df['hum'] = hum
    input_df['windspeed'] = windspeed
    input_df['is_rush_hour'] = is_rush_hour

    # Populate one-hot encoded categorical features
    if season == 'springer': input_df['season_springer'] = 1
    elif season == 'summer': input_df['season_summer'] = 1
    elif season == 'winter': input_df['season_winter'] = 1

    if weathersit == 'Light Rain/Snow': input_df['weathersit_Light Rain/Snow'] = 1
    elif weathersit == 'Mist': input_df['weathersit_Mist'] = 1
    elif weathersit == 'Moderate Rain/Snow': input_df['weathersit_Moderate Rain/Snow'] = 1

    if workingday == 'Working Day': input_df['workingday_Working Day'] = 1

    if holiday == 'Yes': input_df['holiday_Yes'] = 1

    if temp_type == 'Cold': input_df['temp_type_Cold'] = 1
    elif temp_type == 'Hot': input_df['temp_type_Hot'] = 1

    st.markdown("""---""")
    if st.button("Predict Bike Demand", type="primary"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Bike Rental Demand: **{int(prediction):,}** bikes")
        st.balloons()

    st.markdown("""---
    <h2 style='text-align: center; color: #4CAF50;'>References and Explanation</h2>""", unsafe_allow_html=True)
    st.write("This interactive prediction tool utilizes a Gradient Boosting Regressor model, fine-tuned and trained on an extensive bike rental dataset. The model has demonstrated robust performance with an R-squared score of approximately 0.95, indicating its strong capability to explain variance in rental demand.")
    st.write("Key drivers for the prediction include temporal features (hour, year, month, rush hour status), environmental conditions (temperature, humidity, windspeed), and categorical factors (season, weather situation, working day, holiday). All numerical inputs are normalized, and categorical choices are internally transformed using one-hot encoding to match the model's training parameters.")
    st.write("This tool is designed to offer an estimate of demand under specified conditions, providing valuable insights for operational planning.")

