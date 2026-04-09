from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64

app = Flask(__name__)

# Load model
model = joblib.load("house_price_stacking_model.pk1")
rf_model = joblib.load("house_price_rf_model.pk1")
print("MODEL LOADED")

# Home page
@app.route('/')
def home():
    print("Home page loaded")
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        area = float(request.form.get('calculatedfinishedsquarefeet',0))
        bedrooms = float(request.form.get('bedroomcnt',0))
        bathrooms = float(request.form.get('bathroomcnt',0))
        garage = float(request.form.get('garagecarcnt',0))
        yearbuilt = float(request.form.get('yearbuilt',0))

        # Prepare FULL feature set (same as training)
        data = {
            'calculatedfinishedsquarefeet': area,
            'bedroomcnt': bedrooms,
            'bathroomcnt': bathrooms,
            'yearbuilt': yearbuilt,
            'garagecarcnt': garage,

            # Default values (IMPORTANT)
            'garagetotalsqft': 400,
            'latitude': 34.05,
            'longitude': -118.25,
            'lotsizesquarefeet': 6000,
            'unitcnt': 1,
            'taxamount': 5000,
            'regionidcity': 12447,
            'propertylandusetypeid': 261,
            'month': 6
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure SAME ORDER as training
        input_df = input_df[[
            'calculatedfinishedsquarefeet',
            'bedroomcnt',
            'bathroomcnt',
            'yearbuilt',
            'garagecarcnt',
            'garagetotalsqft',
            'latitude',
            'longitude',
            'lotsizesquarefeet',
            'unitcnt',
            'taxamount',
            'regionidcity',
            'propertylandusetypeid',
            'month'
        ]]

        # Predict
        prediction = model.predict(input_df)[0]

        # Safety fix (avoid negative values)
        if prediction < 0:
            prediction = abs(prediction)
        lower = prediction * 0.9
        upper = prediction * 1.1
        confidence_text = f"Estimated Range: ${round(lower,2)} - ${round(upper,2)}"

        # Feature importance plot
        plot_url = None
        try:
            importances = rf_model.feature_importances_
            features = input_df.columns

            plt.figure()
            plt.barh(features, importances)
            plt.title("Feature Importance")
            plt.tight_layout()

            plot_path = os.path.join("static", "feature_importance.png")
            plt.savefig(plot_path)
            plt.close()

            plot_url = "feature_importance.png"

        except Exception as e:
            print("Plot error:", e)

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${round(prediction,2)}",
            confidence_text=confidence_text,
            plot_url=plot_url
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    bedrooms = 5

    if request.method == 'POST':
        bedrooms = int(request.form.get('bedrooms', 5))

    # Sample data
    import random

    bedroom_list = list(range(1, bedrooms + 1))
    price_list = [i * random.randint(80000, 150000) for i in bedroom_list]

    graph_data = {
        "bedrooms": bedroom_list,
        "prices": price_list
    }

    return render_template(
        "dashboard.html",
        graph_data=graph_data,
        bedrooms=bedrooms
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
