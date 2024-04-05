# importing the all libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
import sklearn

app = Flask(__name__)
@app.route("/")

def home():
    return render_template("template/index.html", prediction=model_prediction)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        print(request.form.get("longitude"))
        print(request.form.get("latitude"))
        print(request.form.get("housing_median_age"))
        print(request.form.get("total_rooms"))
        print(request.form.get("total_bedrooms"))
        print(request.form.get("population"))
        print(request.form.get("households"))
        print(request.form.get("median_income"))
        print(request.form.get("ocean_proximity"))
        longitude = float(request.form.get("longitude"))
        latitude = float(request.form.get("latitude"))
        housing_median_age = float(request.form.get("housing_median_age"))
        total_rooms = float(request.form.get("total_rooms"))
        total_bedrooms = float(request.form.get("total_bedrooms"))
        population = float(request.form.get("population"))
        households = float(request.form.get("households"))
        median_income = float(request.form.get("median_income"))
        ocean_proximity = request.form.get("ocean_proximity")
        try:
            longitude = float(request.form.get("longitude"))
            latitude = float(request.form.get("latitude"))
            housing_median_age = float(request.form.get("housing_median_age"))
            total_rooms = float(request.form.get("total_rooms"))
            total_bedrooms = float(request.form.get("total_bedrooms"))
            population = float(request.form.get("population"))
            households = float(request.form.get("households"))
            median_income = float(request.form.get("median_income"))
            ocean_proximity = request.form.get("ocean_proximity")

            pred_args =[longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity]
            pred_arr = np.array(pred_args)
            preds = pred_arr.reshape(1,-1)
            model = open("linear_regression_model.pkl","rb")
            lr_model = joblib.load(model)
            model_prediction = lr_model.predict(preds)
            model_prediction = round(float(model_prediction),2)
        except ValueError:
            return "Please Enter the values"    

    return render_template("template/index.html", prediction=model_prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
    