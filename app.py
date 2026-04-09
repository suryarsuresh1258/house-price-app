from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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

        plot_path = "static/feature_importance.png"
        plot_url = "feature_importance.png"

        try:
            importances = rf_model.feature_importances_
            features = input_df.columns
            plt.figure()
            plt.barh(features,importances)
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except:
            plot_path = None  # rf model not available

        return render_template("index.html",
                               prediction_text=f"Predicted House Price: ${round(prediction,2)}",
                               confidence_text=confidence_text,
                               plot_url=plot_url)

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    import os
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    app.run(host = host,port=port,debug=True)