import streamlit as st
import joblib
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Rain Predictor AU", page_icon="🌦️", layout="centered")

# Function to load the pre-trained pipeline
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    # Ensure the path matches your repository structure
    model_path = os.path.join(base_path, 'models', 'rain_model.pkl')
    pipeline = joblib.load(model_path)
    return pipeline

# Initialize the model
try:
    pipeline = load_model()
    st.sidebar.success("✓ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Australia Rain Forecast 🇦🇺🌦️")
st.markdown("""
This application uses a **Machine Learning model** (Logistic Regression/Random Forest) to predict the probability of rainfall tomorrow based on current meteorological data.
""")

# Sidebar for user input
st.sidebar.header("Weather Input Features")

def user_input_features():
    inputs = {}
    
    # Numerical features
    st.sidebar.subheader("Numerical Metrics")
    inputs['MinTemp'] = st.sidebar.slider("Min Temperature (°C)", -10.0, 40.0, 12.0)
    inputs['MaxTemp'] = st.sidebar.slider("Max Temperature (°C)", -5.0, 50.0, 25.0)
    inputs['Rainfall'] = st.sidebar.number_input("Rainfall Today (mm)", 0.0, 300.0, 0.0)
    inputs['Evaporation'] = st.sidebar.number_input("Evaporation (mm)", 0.0, 150.0, 5.0)
    inputs['Sunshine'] = st.sidebar.slider("Sunshine Hours", 0.0, 15.0, 7.0)
    inputs['WindGustSpeed'] = st.sidebar.slider("Wind Gust Speed (km/h)", 0, 150, 40)
    inputs['WindSpeed9am'] = st.sidebar.slider("Wind Speed at 9am (km/h)", 0, 130, 15)
    inputs['WindSpeed3pm'] = st.sidebar.slider("Wind Speed at 3pm (km/h)", 0, 130, 20)
    inputs['Humidity9am'] = st.sidebar.slider("Humidity at 9am (%)", 0, 100, 60)
    inputs['Humidity3pm'] = st.sidebar.slider("Humidity at 3pm (%)", 0, 100, 50)
    inputs['Pressure9am'] = st.sidebar.number_input("Pressure at 9am (hPa)", 900.0, 1100.0, 1017.0)
    inputs['Pressure3pm'] = st.sidebar.number_input("Pressure at 3pm (hPa)", 900.0, 1100.0, 1015.0)
    inputs['Cloud9am'] = st.sidebar.slider("Cloud Cover at 9am (oktas)", 0, 9, 4)
    inputs['Cloud3pm'] = st.sidebar.slider("Cloud Cover at 3pm (oktas)", 0, 9, 4)
    inputs['Temp9am'] = st.sidebar.slider("Temperature at 9am (°C)", -10.0, 45.0, 18.0)
    inputs['Temp3pm'] = st.sidebar.slider("Temperature at 3pm (°C)", -10.0, 45.0, 23.0)

    # Categorical features
    st.sidebar.subheader("Categorical Context")
    
    locations = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
                 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
                 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
                 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
                 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
                 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
                 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
                 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
                 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
                 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
    
    inputs['Location'] = st.sidebar.selectbox("Location", sorted(locations))
    
    wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    inputs['WindGustDir'] = st.sidebar.selectbox("Wind Gust Direction", wind_directions)
    inputs['WindDir9am'] = st.sidebar.selectbox("Wind Direction at 9am", wind_directions)
    inputs['WindDir3pm'] = st.sidebar.selectbox("Wind Direction at 3pm", wind_directions)
    
    inputs['RainToday'] = st.sidebar.selectbox("Did it rain today?", [0, 1], 
                                               format_func=lambda x: "Yes" if x == 1 else "No")

    return pd.DataFrame([inputs])

# Get user inputs
input_df = user_input_features()

st.subheader("Current Input Summary")
# Displaying with readable labels
display_df = input_df.copy()
display_df['RainToday'] = display_df['RainToday'].map({0: 'No', 1: 'Yes'})
st.write(display_df)

# Prediction execution
if st.button("🌦️ Predict Rainfall", type="primary"):
    try:
        # The pipeline handles scaling and encoding automatically
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("### 🌧️ YES")
                st.write("**Rain is expected tomorrow.**")
            else:
                st.success("### ☀️ NO")
                st.write("**Dry weather is expected tomorrow.**")
        
        with col2:
            st.metric("Probability of Rain", f"{probability[1]:.1%}")
            st.metric("Probability of Sun", f"{probability[0]:.1%}")
        
        # Visual representation
        st.write("### Confidence Level")
        st.progress(float(probability[1]))
        
        # Interpretation logic
        st.divider()
        st.write("### 💡 AI Insights")
        if probability[1] > 0.7:
            st.info("🌧️ High probability of rain. Don't forget your umbrella!")
        elif probability[1] > 0.4:
            st.warning("⛅ Moderate probability. It might be cloudy with occasional showers.")
        else:
            st.success("☀️ Low probability of rain. Perfect weather for outdoor activities!")
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Developed as part of an End-to-End ML Deployment project. Model source: WeatherAUS Dataset.")
