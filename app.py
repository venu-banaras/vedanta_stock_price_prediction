from flask import Flask, render_template, session, url_for, request
import numpy as np

import logging

# from flask_wtf import FlaskForm
# from wtforms import TextField, SubmitField

import tensorflow as tf

keras = tf.keras
from keras.models import load_model
import joblib

# Load the model and the PKL files for prediction
init_model = load_model("initial_model.h5")
init_scaler = joblib.load("Imp Objects/ini_scaler.pkl")
init_look_back = joblib.load("Imp Objects/look_back.pkl")
init_scaled_data = joblib.load("Imp Objects/scaled_data.pkl")


def return_predictions(model, scaler, num_days):

    n_features = init_scaled_data.shape[1]

    forecast = []

    first_eval_batch = init_scaled_data[-init_look_back:]

    current_batch = first_eval_batch.reshape((1, init_look_back, n_features))

    for i in range(num_days):
        current_pred = init_model.predict(current_batch)[0]

        forecast.append(current_pred)

        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    true_forecast = init_scaler.inverse_transform(forecast)
    return true_forecast


app = Flask(__name__, template_folder="Templates")

app.config["SECRET_KEY"] = "mysecretkey"

# i have my own form
# class StockForm(FlaskForm):
#     n_days = TextField("Number of days: ")


@app.route("/home", methods=["GET", "POST"])
def home():
    # return "<h1>Flask app is running</h1>"
    return render_template("index.html")


@app.route("/stock_pred", methods=["GET", "POST"])
def stock_pred():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        n_days = request.form.get("num")
        # n_days = 2
        print(n_days)
        n_days = int(n_days)
        results = return_predictions(model=init_model, scaler=init_scaler, num_days=n_days)
        app.logger.info(results)

        # result = request.form["test"]
        return render_template("prediction.html", results=results, days=n_days)


if __name__ == "__main__":
    app.run(debug=True)
