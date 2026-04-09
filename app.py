from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load models
model = joblib.load("house_price_stacking_model.pk1")
rf_model = joblib.load("house_price_rf_model.pk1")

print("MODEL LOADED")


# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # User inputs
        area = float(request.form.get('calculatedfinishedsquarefeet', 0))
        bedrooms = float(request.form.get('bedroomcnt', 0))
        bathrooms = float(request.form.get('bathroomcnt', 0))
        garage = float(request.form.get('garagecarcnt', 0))
        yearbuilt = float(request.form.get('yearbuilt', 0))

        # Full feature set
        data = {
            'calculatedfinishedsquarefeet': area,
            'bedroomcnt': bedrooms,
            'bathroomcnt': bathrooms,
            'yearbuilt': yearbuilt,
            'garagecarcnt': garage,

            # Default values
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

        input_df = pd.DataFrame([data])

        # Feature order (IMPORTANT)
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

        # Prediction
        prediction = model.predict(input_df)[0]
        prediction = abs(prediction)

        confidence_text = f"Estimated Range: ${round(prediction*0.9,2)} - ${round(prediction*1.1,2)}"

        # Feature Importance Plot
        img = io.BytesIO()

        plt.figure()
        plt.barh(input_df.columns, rf_model.feature_importances_)
        plt.title("Feature Importance")
        plt.tight_layout()

        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: ${round(prediction,2)}",
            confidence_text=confidence_text,
            plot_url=plot_url
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# ---------------- DASHBOARD ----------------
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    try:
        bedrooms = 5

        if request.method == 'POST':
            bedrooms = int(request.form.get('bedrooms', 5))

        # Sample data
        data = pd.DataFrame({
            "bedrooms": list(range(1, bedrooms + 1)),
            "bathrooms": list(range(1, bedrooms + 1)),
            "area": [i * 500 for i in range(1, bedrooms + 1)],
            "price": [i * 100000 for i in range(1, bedrooms + 1)]
        })

        # -------- Bedrooms vs Price --------
        img1 = io.BytesIO()
        plt.figure()
        plt.bar(data["bedrooms"], data["price"])
        plt.xlabel("Bedrooms")
        plt.ylabel("Price")
        plt.title("Bedrooms vs Price")
        plt.savefig(img1, format='png')
        img1.seek(0)
        bar = base64.b64encode(img1.getvalue()).decode()
        plt.close()

        # -------- Area vs Price --------
        img2 = io.BytesIO()
        plt.figure()
        plt.scatter(data["area"], data["price"])
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.title("Area vs Price")
        plt.savefig(img2, format='png')
        img2.seek(0)
        scatter = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        # -------- Bathrooms vs Price --------
        img3 = io.BytesIO()
        plt.figure()
        plt.scatter(data["bathrooms"], data["price"])
        plt.xlabel("Bathrooms")
        plt.ylabel("Price")
        plt.title("Bathrooms vs Price")
        plt.savefig(img3, format='png')
        img3.seek(0)
        scatter2 = base64.b64encode(img3.getvalue()).decode()
        plt.close()

        return render_template(
            "dashboard.html",
            bar=bar,
            scatter=scatter,
            scatter2=scatter2,
            bedrooms=bedrooms
        )

    except Exception as e:
        return f"Dashboard Error: {str(e)}"


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
