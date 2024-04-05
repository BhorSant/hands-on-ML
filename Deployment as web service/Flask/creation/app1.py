import pandas as pd
import numpy as np
import joblib 
import streamlit
from sklearn.preprocessing import OneHotEncoder

model = open("Deployment as web service\Flask\creation\model.pkl","rb")
lr_model = joblib.load(model)

# Function to one-hot encode the categorical variable
def encode_ocean_proximity(ocean_proximity):
    categories = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
    encoder = OneHotEncoder(categories=[categories], sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform([[ocean_proximity]])
    return encoded


def lr_prediction(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    # Convert input values to float
    longitude = float(longitude)
    latitude = float(latitude)
    housing_median_age = float(housing_median_age)
    total_rooms = float(total_rooms)
    total_bedrooms = float(total_bedrooms)
    population = float(population)
    households = float(households)
    median_income = float(median_income)
    # Encode ocean_proximity
    encoded_ocean_proximity = encode_ocean_proximity(ocean_proximity)
    # Combine all features into a single array
    pred_arr = np.array([longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income])
    preds = np.concatenate((pred_arr, encoded_ocean_proximity), axis=None).reshape(1, -1)
    # Predict using the model
    model_prediction = lr_model.predict(preds)
    return model_prediction

def run():
    streamlit.title("Housing Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Housing Price Prediction </h2>
    </div>
    """
    streamlit.markdown(html_temp, unsafe_allow_html=True)
    longitude = streamlit.text_input("longitude", "Type Here")
    latitude = streamlit.text_input("latitude", "Type Here")
    housing_median_age = streamlit.text_input("housing_median_age", "Type Here")
    total_rooms = streamlit.text_input("total_rooms", "Type Here")
    total_bedrooms = streamlit.text_input("total_bedrooms", "Type Here")
    population = streamlit.text_input("population", "Type Here")
    households = streamlit.text_input("households", "Type Here")
    median_income = streamlit.text_input("median_income", "Type Here")
    ocean_proximity = streamlit.text_input("ocean_proximity", "Type Here")
    result = ""
    if streamlit.button("Predict"):
        result = lr_prediction(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity)
    streamlit.success('The output is {}'.format(result))
    if streamlit.button("About"):
        streamlit.text("Developed by Santosh")
        streamlit.text("Email: santoshbhor2001@gmail.com")
        streamlit.text("Github: https://github.com/BhorSant")
        

if __name__ == "__main__":
    run()



